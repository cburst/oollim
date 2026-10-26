"""Versioned, public-trained scoring choices shared by CLI and desktop."""
from .common import ROOT
FAMILIES = ('speechocean','korean','aesop','koreanocean','aesopocean','koreanaesop','aesopkoreanocean')
SELECTIONS = {
    'spearman': '1-grid · construct-wise correlation',
    'damae': '1-grid · pronunciation-accuracy error',
    'historical_grid_damae': '5-grid · pronunciation-accuracy error',
    'banded_3ratio_spearman': 'Three-ratio banded · construct-wise correlation',
    'banded_3ratio_accuracy_mae': 'Three-ratio banded · pronunciation-accuracy error',
}
DEFAULT_SELECTION = 'spearman'

def bundle_path(family, selection=DEFAULT_SELECTION):
    if family not in FAMILIES or selection not in SELECTIONS:
        raise ValueError('Choose a supported family and selection rule.')
    collection = 'banded-3ratio-v1' if selection.startswith('banded_3ratio_') else 'selection-comparison-v1'
    return ROOT/'models'/collection/selection/family/'family.json'
