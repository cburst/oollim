#!/usr/bin/env python3
"""Portable launcher; automatically uses the installer-selected interpreter."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 config=ROOT/'.oollim-install.json'
 if config.exists() and not os.environ.get('OOLLiM_SELECTED'):
  try:target=Path(json.loads(config.read_text())['python'])
  except (ValueError,KeyError,OSError):
   print('Invalid installer record. Rerun install.py to restore it.',file=sys.stderr);raise SystemExit(1)
  if not target.exists():
   print('The installed Python environment is missing. Rerun install.py.',file=sys.stderr);raise SystemExit(1)
  if target.exists() and str(target.absolute())!=str(Path(sys.executable).absolute()):
   raise SystemExit(subprocess.call([str(target),str(Path(__file__).resolve()),*sys.argv[1:]],env={**os.environ,'OOLLiM_SELECTED':'1'}))
 try:
  if '--gui' in sys.argv:
   from oollim_app.gui import main
   raise SystemExit(main())
  from oollim_app.cli import main
  raise SystemExit(main())
 except ModuleNotFoundError as exc:
  print(f'Missing Python dependency: {exc.name}. Run python3.11 install.py (Mac) or py -3.11 install.py (Windows) from this folder.',file=sys.stderr);raise SystemExit(1)
