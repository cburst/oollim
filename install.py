#!/usr/bin/env python3
"""Install oollim into .venv (default) or the invoking Python (--current)."""
import argparse,json,os,platform,subprocess,sys,time,venv
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(command,log):
 with log.open('a',encoding='utf-8') as f:
  f.write('\nCOMMAND: '+repr(command)+'\n');f.flush()
  process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  try:
   while process.poll() is None:time.sleep(.2)
  except KeyboardInterrupt:
   process.terminate()
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill();process.wait()
   raise
 if process.returncode:
  tail='\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-18:])
  raise RuntimeError(f'Command failed (exit {process.returncode}).\n{tail}\nFull log: {log}\nUse a Python 3.11 virtual environment; check network access and free disk space. On Windows, DLL errors may require the Microsoft Visual C++ x64 runtime. Do not use sudo or --break-system-packages.')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',action='store_true',help='Install into this interpreter instead of creating .venv');p.add_argument('--skip-models',action='store_true',help='Install application/dependencies only; encoder download can be resumed later');p.add_argument('--plan',action='store_true',help='Print planned operations without changing the environment');p.add_argument('--log',type=Path,help=argparse.SUPPRESS);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
 from oollim_app.common import platform_check
 platform_check()
 if sys.version_info[:2]!=(3,11):raise RuntimeError(f'Python 3.11 is required for the pinned runtime; found {platform.python_version()}. Install native Python 3.11 from python.org, then run python3.11 install.py (Mac) or py -3.11 install.py (Windows).')
 target=Path(sys.executable) if a.current else ROOT/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
 if a.plan:
  print(f'Interpreter: {target}\nStages: create/select environment; bootstrap tqdm; install CPU Torch; install requirements; verify/extract ESPnet; '+('defer encoder weights' if a.skip_models else 'download/check encoder weights')+'; runtime import/FFmpeg check.\nNo data or paid API calls.');return 0
 logdir=ROOT/'work/install';logdir.mkdir(parents=True,exist_ok=True);log=a.log or logdir/(time.strftime('%Y%m%d-%H%M%S')+'.log')
 if not a.worker:
  if not a.current and not target.exists():
   print('Creating isolated .venv ...',flush=True);venv.EnvBuilder(with_pip=True).create(ROOT/'.venv')
  check=subprocess.check_output([str(target),'-c','import sys,platform,struct,json; print(json.dumps([list(sys.version_info[:2]),platform.machine().lower(),struct.calcsize("P")]))'],text=True).strip()
  if json.loads(check)!=[[3,11],platform.machine().lower(),8]:raise RuntimeError('Existing environment has an incompatible Python version or architecture: '+check+'. Move .venv aside and rerun with native 64-bit Python 3.11.')
  print('Preparing tqdm installer progress ...',flush=True);run([str(target),'-m','pip','install','tqdm==4.70.1'],log)
  command=[str(target),str(Path(__file__).resolve()),'--worker','--current','--log',str(log)]
  if a.skip_models:command.append('--skip-models')
  return subprocess.call(command,cwd=ROOT)
 from tqdm import tqdm
 pip=[sys.executable,'-m','pip','install','--disable-pip-version-check']
 torch=pip+['torch==2.11.0','torchaudio==2.11.0']
 if os.name=='nt':torch+=['--index-url','https://download.pytorch.org/whl/cpu']
 steps=[('Install CPU PyTorch',lambda:run(torch,log)),('Install application dependencies',lambda:run(pip+['-r',str(ROOT/'requirements-runtime.txt')],log)),('Install pinned ESPnet source',lambda:run([sys.executable,'-m','oollim_app.assets','--source-only'],log))]
 if not a.skip_models:steps.append(('Download verified pretrained encoders',lambda:run([sys.executable,'-m','oollim_app.assets'],log)))
 steps.append(('Verify runtime and FFmpeg',lambda:run([sys.executable,'oollim.py','--doctor','--runtime-only'],log)))
 with tqdm(total=len(steps),unit='stage',desc='Installing oollim') as progress:
  for label,action in steps:
   progress.set_description(label);action();progress.update()
 atomic=ROOT/'.oollim-install.json';temp=atomic.with_suffix('.tmp');temp.write_text(json.dumps({'python':sys.executable,'platform':platform.platform(),'models_downloaded':not a.skip_models},indent=2));temp.replace(atomic)
 print('\nDependencies installed. Start: '+subprocess.list2cmdline([sys.executable,'oollim.py'])+'\nLog: '+str(log))
 print('Encoder downloads were deferred. Run oollim.py --download-models before scoring.' if a.skip_models else 'Run oollim.py --doctor to verify scoring assets, then oollim.py to open the dashboard.')
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except KeyboardInterrupt:print('\nInstallation interrupted. Rerun the same command to continue.',file=sys.stderr);raise SystemExit(130)
 except Exception as exc:print('\nINSTALLATION FAILED\n'+str(exc),file=sys.stderr);raise SystemExit(1)
