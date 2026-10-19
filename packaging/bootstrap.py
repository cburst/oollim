"""First-run provisioning for bundled, relocatable CPython installers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def data_home():
    return Path.home() / 'oollim'


def main(argv=()):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup-only',action='store_true')
    args=parser.parse_args(argv)
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / 'payload.json').read_text())
    version = hashlib.sha256((bundle / 'payload.json').read_bytes()).hexdigest()[:16]
    home = data_home()
    home.mkdir(parents=True, exist_ok=True)
    lock = home / 'setup.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f'Another installer may be running. Close it before retrying. '
                           f'After a forced shutdown, remove {lock} only after confirming no setup is active.')
    try:
        target = home / 'versions' / version
        complete = target / '.payload-ready'
        if not complete.exists():
            print('Preparing private application runtime...', flush=True)
            target.mkdir(parents=True, exist_ok=True)
            for relative, expected in manifest.items():
                source = bundle / relative
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Installer integrity check failed: {relative}. Download a fresh installer.')
            for folder in ('app', 'python'):
                shutil.copytree(bundle / folder, target / folder, dirs_exist_ok=True, symlinks=True)
            complete.write_text(version)
        app = target / 'app'
        python = target / 'python' / ('python.exe' if sys.platform == 'win32' else 'bin/python3.11')
        env = os.environ.copy()
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        for key, folder in [('PIP_CACHE_DIR','cache/pip'),('XDG_CACHE_HOME','cache'),('HF_HOME','cache/huggingface'),('TORCH_HOME','cache/torch'),('TMPDIR','tmp'),('TEMP','tmp'),('TMP','tmp')]:
            location=home/folder
            location.mkdir(parents=True,exist_ok=True)
            env[key]=str(location)
        if not (app / '.oollim-install.json').exists():
            print('Installing dependencies and downloading speech models. Internet access is required.', flush=True)
            subprocess.run([str(python), str(app / 'install.py')], cwd=app, env=env, check=True)
    finally:
        lock.rmdir()
    if args.setup_only:
        print(f'Setup completed: {app}', flush=True)
        return 0
    output = home / 'results'
    output.mkdir(parents=True, exist_ok=True)
    print(f'Results folder: {output}', flush=True)
    return subprocess.call([str(python), str(app / 'oollim.py'), '--gui'], cwd=output, env=env)


if __name__ == '__main__':
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f'\noollim could not start: {exc}\nRelaunch to retry. Installation logs are in the application data folder.', file=sys.stderr)
        input('Press Enter to close...')
        raise SystemExit(1)
