import copy
import math

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F


class TransferHeads(nn.Module):
    def __init__(self, width, hidden=128, dropout=0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            "speech": nn.Linear(hidden, 3),
            "korean": nn.Linear(hidden, 2),
            "sm": nn.Linear(hidden, 3),
        })

    def forward(self, x, source):
        return self.heads[source](self.shared(x))


def _config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _mask(mask, count, name):
    mask = np.asarray(mask)
    if mask.shape != (count,) or mask.dtype.kind != "b":
        raise ValueError(f"{name} must be a one-dimensional boolean mask")
    if not mask.any():
        raise ValueError(f"{name} must be nonempty")
    return mask


def _data(model, x, y, source):
    if source not in model.heads:
        raise ValueError(f"unknown source: {source}")
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be two-dimensional with matching rows")
    if x.shape[0] == 0 or x.shape[1] != model.shared[0].in_features:
        raise ValueError("empty data or incorrect feature width")
    if y.shape[1] != model.heads[source].out_features:
        raise ValueError("incorrect target width for source")
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(
        y.dtype, np.number
    ):
        raise ValueError("x and y must be numeric")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must be finite")
    if ((y < 0) | (y > 1)).any():
        raise ValueError("targets must be normalized to [0, 1]")
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must remain finite after conversion")
    return x, y


def _settings(config):
    lr = float(_config_value(config, "learning_rate"))
    decay = float(_config_value(config, "weight_decay"))
    batch_size = _positive_int(_config_value(config, "batch_size", 32), "batch_size")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    return lr, decay, batch_size


def _optimizer(model, source, lr, decay):
    parameters = list(model.shared.parameters()) + list(
        model.heads[source].parameters()
    )
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=decay)


def _train_epoch(model, features, targets, indices, source, optimizer, batch_size, rng):
    model.train()
    shuffled = rng.permutation(indices)
    total_loss = 0.0
    for start in range(0, len(shuffled), batch_size):
        batch = shuffled[start:start + batch_size]
        model.zero_grad(set_to_none=True)
        output = model(features[batch], source)
        loss = F.smooth_l1_loss(output, targets[batch], beta=0.1)
        if not torch.isfinite(loss):
            raise ValueError("training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(indices)


def _cpu_state(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _scores(predictions, targets):
    correlations = []
    for dimension in range(targets.shape[1]):
        actual = targets[:, dimension]
        predicted = predictions[:, dimension]
        if (
            len(actual) < 2
            or np.all(actual == actual[0])
            or np.all(predicted == predicted[0])
        ):
            raise ValueError("Undefined inner validation Spearman")
        else:
            correlation = float(spearmanr(actual, predicted).statistic)
            if not math.isfinite(correlation):
                raise ValueError("Undefined inner validation Spearman")
        correlations.append(correlation)
    return float(np.mean(correlations)), float(np.mean(np.abs(predictions - targets)))


def fit_stage(model, x, y, train_mask, valid_mask, source, seed, config, device):
    x, y = _data(model, x, y, source)
    train_mask = _mask(train_mask, len(x), "train_mask")
    valid_mask = _mask(valid_mask, len(x), "valid_mask")
    if np.any(train_mask & valid_mask):
        raise ValueError("training and validation masks must be disjoint")
    lr, decay, batch_size = _settings(config)
    epochs = _positive_int(_config_value(config, "epochs"), "epochs")
    patience = _positive_int(_config_value(config, "patience"), "patience")
    rng = _seed(seed)

    model.to(device)
    dtype = model.shared[0].weight.dtype
    features = torch.as_tensor(x, dtype=dtype, device=device)
    targets = torch.as_tensor(y, dtype=dtype, device=device)
    train_indices = np.flatnonzero(train_mask)
    valid_indices = np.flatnonzero(valid_mask)
    optimizer = _optimizer(model, source, lr, decay)

    history = []
    best_state = None
    best_objective = (-math.inf, -math.inf)
    best_epoch = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(
            model, features, targets, train_indices, source,
            optimizer, batch_size, rng,
        )
        model.eval()
        with torch.no_grad():
            predictions = model(features[valid_indices], source).cpu().numpy()
        if not np.isfinite(predictions).all():
            raise ValueError("validation produced non-finite predictions")
        mean_spearman, mean_mae = _scores(predictions, y[valid_indices])
        objective = (mean_spearman, -mean_mae)
        history.append({
            "epoch": epoch,
            "train_loss": loss,
            "mean_spearman": mean_spearman,
            "mean_mae": mean_mae,
            "objective": objective,
        })
        if objective > best_objective:
            best_objective = objective
            best_epoch = epoch
            best_state = _cpu_state(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return best_state, {
        "best_epoch": best_epoch,
        "best_objective": best_objective,
        "history": history,
    }


def fit_fixed(model, x, y, train_mask, source, seed, config, epochs, device):
    x, y = _data(model, x, y, source)
    train_mask = _mask(train_mask, len(x), "train_mask")
    epochs = _positive_int(epochs, "epochs")
    lr, decay, batch_size = _settings(config)
    rng = _seed(seed)

    model.to(device)
    dtype = model.shared[0].weight.dtype
    features = torch.as_tensor(x, dtype=dtype, device=device)
    targets = torch.as_tensor(y, dtype=dtype, device=device)
    indices = np.flatnonzero(train_mask)
    optimizer = _optimizer(model, source, lr, decay)
    history = []
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(
            model, features, targets, indices, source,
            optimizer, batch_size, rng,
        )
        history.append({"epoch": epoch, "train_loss": loss})
    return _cpu_state(model), {"epochs": epochs, "history": history}


def predict(model, x, source, device):
    if source not in model.heads:
        raise ValueError(f"unknown source: {source}")
    x = np.asarray(x)
    if (
        x.ndim != 2
        or x.shape[1] != model.shared[0].in_features
        or not np.issubdtype(x.dtype, np.number)
        or not np.isfinite(x).all()
    ):
        raise ValueError("x must contain finite features with the expected width")
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("x must remain finite after conversion")
    model.to(device)
    model.eval()
    dtype = model.shared[0].weight.dtype
    with torch.no_grad():
        result = model(
            torch.as_tensor(x, dtype=dtype, device=device), source
        ).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("prediction produced non-finite values")
    return result


def speaker_inner_split(groups, outer_train, seed=2026):
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    outer_train = _mask(outer_train, len(groups), "outer_train")
    indices = np.flatnonzero(outer_train)
    speakers, inverse = np.unique(groups[indices], return_inverse=True)
    if len(speakers) < 2:
        raise ValueError("at least two training speakers are required")
    rng = _seed(seed)
    valid_count = min(len(speakers) - 1, math.ceil(0.2 * len(speakers)))
    chosen = rng.permutation(len(speakers))[:valid_count]
    inner_valid = np.zeros(len(groups), dtype=bool)
    inner_valid[indices] = np.isin(inverse, chosen)
    inner_train = outer_train & ~inner_valid
    return inner_train, inner_valid
