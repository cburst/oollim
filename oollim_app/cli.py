"""Local file/batch evaluation with durable, content-checked resume records."""
import argparse,csv,hashlib,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
from functools import lru_cache
from importlib.metadata import version,PackageNotFoundError
from .common import ROOT,STAGES,OollimError,atomic_json,read_reference,sha
from .scoring import FAMILIES, SELECTIONS, DEFAULT_SELECTION, bundle_path
EXTENSIONS={'.wav','.mp3','.m4a','.mp4','.mov','.flac','.ogg','.opus','.webm','.mkv','.aac','.aiff','.aif','.wma','.amr'}

@lru_cache(maxsize=1)
def runtime_signature():
 h=hashlib.sha256()
 for folder in ('oollim_app','speech/pronunciation'):
  for path in sorted((ROOT/folder).glob('*.py')):h.update(sha(path).encode())
 for package in ('torch','torchaudio','numpy','scipy','transformers','imageio-ffmpeg','nltk'):
  try:value=version(package)
  except PackageNotFoundError:value='missing'
  h.update((package+value).encode())
 return h.hexdigest()

def identity(audio,text,model):
 h=hashlib.sha256()
 for value in (sha(audio),text,sha(model),sha(ROOT/'configs/extractor.json'),runtime_signature()):h.update(value.encode());h.update(b'\0')
 return h.hexdigest()

def save(output,records):
 atomic_json(output.with_suffix(output.suffix+'.state.json'),records)
 fields=['audio','status','family','selection','model_version','duration_seconds','accuracy','fluency','prosody','error_code','error_message','fingerprint']
 temp=output.with_name(output.name+'.tmp')
 with temp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for r in records.values():w.writerow({**r,**r.get('scores',{}),'error_code':r.get('error',{}).get('code',''),'error_message':r.get('error',{}).get('message','')})
  f.flush();os.fsync(f.fileno())
 try:temp.replace(output)
 except PermissionError as exc:raise OollimError('E_OUTPUT_BUSY',f'Cannot update {output}.','Close the CSV in Excel or other software and rerun; results are retained in the state file.') from exc

def stop(process):
 if process.poll() is not None:return
 if os.name=='nt':
  subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
 else:
  try:os.killpg(process.pid,signal.SIGTERM)
  except ProcessLookupError:pass
 try:process.wait(timeout=10)
 except subprocess.TimeoutExpired:
  if os.name=='nt':process.kill()
  else:
   try:os.killpg(process.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  process.wait()
 if os.name!='nt':
  try:os.killpg(process.pid,signal.SIGKILL)
  except ProcessLookupError:pass

def main(argv=None, event_callback=None, cancel_event=None):
 p=argparse.ArgumentParser(description='oollim: local whole-utterance pronunciation scoring')
 source=p.add_mutually_exclusive_group();source.add_argument('--audio',type=Path);source.add_argument('--folder',type=Path)
 p.add_argument('--transcript',type=Path,help='Common reference text; otherwise use a .txt beside each audio file')
 p.add_argument('--family',choices=FAMILIES,default='speechocean');p.add_argument('--verify-install',action='store_true');p.add_argument('--recursive',action='store_true');p.add_argument('--model',type=Path,default=None)
 p.add_argument('--output',type=Path,default=Path('scores.csv'));p.add_argument('--threads',type=int,default=4);p.add_argument('--timeout',type=float,default=3600)
 p.add_argument('--retry-failed',action='store_true');p.add_argument('--no-live',action='store_true');p.add_argument('--doctor',action='store_true');p.add_argument('--runtime-only',action='store_true');p.add_argument('--download-models',action='store_true')
 p.add_argument('--selection',choices=tuple(SELECTIONS),default=DEFAULT_SELECTION,help='Public-development tuning rule; default: matched Spearman. --model overrides both selectors.')
 p.add_argument('--granular',action='store_true',help='Experimental sentence construct scores and word phone-support scores')
 a=p.parse_args(argv)
 if a.verify_install:
  if not a.audio or not a.transcript:p.error('--verify-install requires --audio and --transcript')
  from .acceptance import main as verify
  return verify(['--audio',str(a.audio),'--transcript',str(a.transcript)])
 custom_model=a.model is not None
 a.model=a.model or bundle_path(a.family,a.selection)
 try:
  from .common import platform_check
  platform_check()
  if a.threads<1 or a.timeout<=0:p.error('--threads and --timeout must be positive')
  if a.download_models:
   from .assets import setup
   setup();return 0
  from .engine import runtime_check,verify_resources,load_bundle
  if a.doctor:
   print(json.dumps(runtime_check(),indent=2))
   if not a.runtime_only:verify_resources();load_bundle(a.model)
   print('Runtime checks passed.' if a.runtime_only else 'Runtime and scoring asset checks passed.');return 0
  from rich.console import Console
  from rich.prompt import Prompt
  from rich.live import Live
  from .dashboard import dashboard
  console=Console()
  if not a.audio and not a.folder:
   if not sys.stdin.isatty():p.print_help();return 2
   mode=Prompt.ask('Evaluate',choices=['file','folder'],default='file')
   path=Path(Prompt.ask('Audio file' if mode=='file' else 'Audio folder').strip('"'))
   if mode=='file':a.audio=path
   else:a.folder=path
   ref=Prompt.ask('Reference .txt (leave blank for matching sidecar files)',default='')
   if ref:a.transcript=Path(ref.strip('"'))
  if a.folder and not a.folder.is_dir():raise OollimError('E_FOLDER',f'Folder not found: {a.folder}','Supply an existing audio folder.')
  files=[a.audio.resolve()] if a.audio else sorted(x.resolve() for x in (a.folder.rglob('*') if a.recursive else a.folder.iterdir()) if x.is_file() and x.suffix.lower() in EXTENSIONS)
  if not files:raise OollimError('E_NO_AUDIO','No supported audio/video files found.','Check the folder or add --recursive for subfolders.')
  runtime_check();verify_resources();load_bundle(a.model)
  output=a.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
  lock=output.with_suffix(output.suffix+'.lock')
  try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
  except FileExistsError:raise OollimError('E_BUSY',f'Output is locked: {lock}','Stop the other run first. After a crashed run, verify no worker remains before deleting this lock file.')
  os.write(fd,str(os.getpid()).encode());os.close(fd)
  try:
   statepath=output.with_suffix(output.suffix+'.state.json')
   if output.exists() and not statepath.exists():raise OollimError('E_RESUME_STATE','CSV exists without its resume state.','Choose a new output path or restore the matching .state.json file; existing scores will not be overwritten.')
   records=json.loads(statepath.read_text()) if statepath.exists() else {}
   if set(records)-{str(f) for f in files}:raise OollimError('E_RESUME_INPUT','This output contains recordings outside the selected input set.','Use a new output path for a different folder or recording selection.')
   save(output,records)  # Rebuild CSV if the previous run stopped between atomic writes.
   logs=output.parent/(output.stem+'_logs');logs.mkdir(exist_ok=True)
   view=dict(mode='Batch' if a.folder else 'Single file',current='',family=a.model.parent.name if custom_model else f'{a.family} / {a.selection}',threads=a.threads,done=0,failed=0,skipped=0,processed=0,total=len(files),stages=[],scores={},activity=[],output=str(output))
   with _Live(Live,console,a.no_live) as live:
    def publish():
     if event_callback:event_callback(view)
     live.update(dashboard(view),refresh=True)
    for audio in files:
     if cancel_event is not None and cancel_event.is_set():raise KeyboardInterrupt
     view.update(current=audio.name,stages=[{'status':'waiting'} for _ in STAGES],scores={},error=None)
     key=str(audio);fingerprint=''
     try:
      ref=(a.transcript or audio.with_suffix('.txt')).resolve();text=read_reference(ref);fingerprint=identity(audio,text,a.model)+('-granular-v2' if a.granular else '')
      old=records.get(key,{})
      if old.get('fingerprint')==fingerprint and (old.get('status')=='completed' or (old.get('status')=='failed' and not a.retry_failed)):
       view['skipped']+=1;view['processed']+=1
       if old.get('status')=='failed':view['failed']+=1
       publish();continue
      with tempfile.TemporaryDirectory(prefix='oollim-') as directory:
       tmp=Path(directory);request=dict(audio=str(audio),transcript=str(ref),model=str(a.model.resolve()),temp=directory,events=str(tmp/'events.jsonl'),result=str(tmp/'result.json'),threads=a.threads,granular=a.granular)
       atomic_json(tmp/'request.json',request)
       log=logs/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.log');view['log']=str(log)
       with log.open('w',encoding='utf-8') as f:
        process=subprocess.Popen([sys.executable,'-m','oollim_app.worker','--request',str(tmp/'request.json')],cwd=ROOT,env={**os.environ,"PYTHONUTF8":"1"},stdout=f,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
        start=time.monotonic()
        try:
         while process.poll() is None:
          events=tmp/'events.jsonl'
          if events.exists():
           for line in events.read_text().splitlines():
            try:e=json.loads(line)
            except json.JSONDecodeError:continue
            if e['status']=='running':e['elapsed']=time.time()-e.get('at',time.time())
            view['stages'][e['stage']]=e
          publish()
          if cancel_event is not None and cancel_event.is_set():raise KeyboardInterrupt
          if time.monotonic()-start>a.timeout:raise OollimError('E_TIMEOUT',f'Inference exceeded {a.timeout:g} seconds.','Inspect the log and available memory; increase --timeout only if the process is making progress.')
          time.sleep(.3)
        finally:stop(process)
       events=tmp/'events.jsonl'
       if events.exists():
        for line in events.read_text().splitlines():
         try:e=json.loads(line)
         except json.JSONDecodeError:continue
         view['stages'][e['stage']]=e
       resultpath=tmp/'result.json'
       if not resultpath.exists():raise OollimError('E_WORKER_EXIT',f'Worker exited with code {process.returncode} without a result.','Inspect the recording log and system memory. Run --doctor before retrying.')
       result=json.loads(resultpath.read_text())
       if process.returncode!=0 and result.get('status')=='completed':raise OollimError('E_WORKER_EXIT','Worker did not exit successfully.','Inspect the log before retrying; its incomplete result was rejected.')
       if identity(audio,read_reference(ref),a.model)+('-granular-v2' if a.granular else '')!=fingerprint:raise OollimError('E_INPUT_CHANGED','Audio, reference or model changed during scoring.','Keep inputs unchanged and rerun.')
     except (OSError,ValueError,OollimError) as exc:
      result={'status':'failed','error':{'code':getattr(exc,'code','E_INPUT'),'message':str(exc),'fix':getattr(exc,'fix','Check paths, permissions and the diagnostic log.')}}
     if result['status']=='failed':
      for step in view['stages']:
       if step['status']=='running':step['status']='failed'
     records[key]={'audio':key,'fingerprint':fingerprint,**result};save(output,records)
     if result['status']=='completed':view['done']+=1;view['scores']=result['scores'];view['stages'][-1]={'status':'done','detail':'CSV and resume state saved'}
     else:view['failed']+=1;view['error']=result['error']
     view['processed']+=1
     publish()
   console.print(f'Saved {output}: {view["done"]} new results, {view["skipped"]} reused, {view["failed"]} failures. Use --retry-failed to retry.')
   return 1 if view['failed'] else 0
  finally:lock.unlink(missing_ok=True)
 except KeyboardInterrupt:print('\nStopped. Completed results are saved; rerun to resume.');return 130
 except ModuleNotFoundError as exc:print(f'Missing dependency: {exc.name}. Rerun install.py with native Python 3.11.',file=sys.stderr);return 1
 except Exception as exc:
  if event_callback:event_callback({'error':{'message':str(exc)}})
  print(str(exc),file=sys.stderr);return 1

class _Live:
 def __init__(self,cls,console,disabled):self.live=None if disabled or not console.is_terminal else cls(console=console,auto_refresh=False)
 def __enter__(self):
  if self.live:self.live.__enter__()
  return self
 def update(self,*args,**kwargs):
  if self.live:self.live.update(*args,**kwargs)
 def __exit__(self,*args):
  if self.live:return self.live.__exit__(*args)
