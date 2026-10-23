"""Stage a release with an approved standalone Python, then compile a native installer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=['mac', 'windows'], required=True)
    p.add_argument('--python-runtime', type=Path, required=True, help='Unpacked redistributable CPython 3.11 install-only directory; retain its licenses')
    p.add_argument('--output',type=Path,help='Fresh build directory; existing staging is never overwritten')
    p.add_argument('--stage-only', action='store_true')
    a = p.parse_args()
    runtime = a.python_runtime.resolve()
    exe = runtime / ('python.exe' if a.target == 'windows' else 'bin/python3.11')
    if not exe.is_file():
        p.error(f'Missing bundled interpreter: {exe}')
    build = a.output.resolve() if a.output else ROOT / 'work/native' / a.target
    stage = build / 'stage'
    if stage.exists():
        p.error(f'Staging folder already exists: {stage}. Use a fresh staging folder after reviewing prior outputs.')
    stage.mkdir(parents=True)
    files = json.loads((ROOT/'packaging/source-files.json').read_text())
    for name in filter(None, files):
        if name.startswith(('.github/', 'paper/', 'tests/', 'results/', 'packaging/')):
            continue
        dest = stage / 'app' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    # Python regenerates bytecode on startup; do not ship mutable caches.
    shutil.copytree(runtime, stage / 'python', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    shutil.copy2(ROOT / 'packaging/bootstrap.py', stage / 'bootstrap.py')
    launcher = 'launch.cmd' if a.target == 'windows' else 'launch.command'
    shutil.copy2(ROOT / 'packaging' / launcher, stage / launcher)
    (stage / launcher).chmod(0o755)
    manifest = {str(f.relative_to(stage)).replace('\\','/'): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(stage.rglob('*')) if f.is_file()}
    (stage / 'payload.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if a.stage_only:
        print(stage)
        return
    if a.target == 'windows':
        shutil.copy2(ROOT / 'packaging/oollim.nsi', build / 'oollim.nsi')
        subprocess.run(['makensis', 'oollim.nsi'], cwd=build, check=True)
    else:
        subprocess.run(['platypus', '-a', 'oollim', '-o', 'None', '-p', '/bin/sh',
                        '-I', 'ai.interlanguage.oollim', '-i', str(ROOT/'assets/oollim.icns'),
                        str(ROOT / 'packaging/launch-mac.sh'), str(build / 'oollim.app')], check=True)
        shutil.copy2(ROOT/'packaging/install-mac.command', build/'Install oollim.command')
        (build/'Install oollim.command').chmod(0o755)
        subprocess.run(['platypus', '-a', 'Install oollim', '-o', 'Text Window', '-p', '/bin/sh',
                        '-I', 'ai.interlanguage.oollim.installer', '-u', 'Richard Rose',
                        '-i', str(ROOT/'assets/oollim.icns'),
                        '-f', str(stage)+'|'+str(build/'oollim.app'),
                        str(ROOT/'packaging/install-mac-app.sh'), str(build/'Install oollim.app')], check=True)
    print(f'Installer built in {build}. Native installation and signing checks are still required.')

if __name__ == '__main__':
    main()
