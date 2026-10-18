"""Download public resources with resumable partials and a local integrity ledger.

No credentials, model execution, installation, archive extraction, or paid services.
Run from any directory: python3 acquire.py [--only resource-id ...]
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def acquire(item):
    destination = ROOT / 'resources' / item['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + '.receipt.json')
    if destination.exists():
        actual = digest(destination)
        expected = item.get('sha256')
        if receipt.exists():
            old = json.loads(receipt.read_text())
            expected = expected or old['sha256']
            if old['url'] != item['url']:
                raise ValueError(f"Source changed for {item['id']}; explicit migration required")
        if not expected or actual != expected:
            raise ValueError(f"Unverified existing file {destination}; inspect before adoption")
        return {'id': item['id'], 'status': 'verified_cached', 'sha256': actual}
    partial = destination.with_name(destination.name + '.part')
    if not (item.get('bytes') and partial.exists() and partial.stat().st_size == item['bytes']):
        subprocess.run(['curl', '-fsSL', '-C', '-', '--connect-timeout', '20',
                        '--max-time', '1200', '--proto', '=https', '--proto-redir', '=https',
                        '-o', str(partial), item['url']], check=True)
    actual = digest(partial)
    if item.get('sha256') and actual != item['sha256']:
        raise ValueError(f"SHA256 mismatch for {item['id']}")
    if item.get('bytes') and partial.stat().st_size != item['bytes']:
        raise ValueError(f"Size mismatch for {item['id']}")
    with partial.open('rb') as source:
        prefix = source.read(16)
    magic = {'pdf': b'%PDF', 'gzip': b'\x1f\x8b', 'zip': b'PK'}
    if item.get('format') in magic and not prefix.startswith(magic[item['format']]):
        raise ValueError(f"Unexpected file format for {item['id']}")
    if item.get('format') == 'json':
        json.loads(partial.read_text())
    metadata = {**item, 'sha256': actual, 'bytes': partial.stat().st_size,
                'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'integrity': 'upstream_sha256' if item.get('sha256') else 'local_sha256_only'}
    partial.replace(destination)
    receipt.write_text(json.dumps(metadata, indent=2) + '\n')
    return {'id': item['id'], 'status': 'downloaded', 'sha256': actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+')
    args = parser.parse_args()
    items = json.loads((ROOT / 'resources-manifest.json').read_text())['resources']
    if args.only:
        unknown = set(args.only) - {x['id'] for x in items}
        if unknown:
            parser.error(f'Unknown resource IDs: {sorted(unknown)}')
        items = [x for x in items if x['id'] in args.only]
    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(acquire, x): x for x in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                failed = True
                print(json.dumps({'id': item['id'], 'status': 'failed', 'error': str(error)}), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
