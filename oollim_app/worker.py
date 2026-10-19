"""One recording per subprocess; JSONL progress, full diagnostic log, no retained decoded audio."""
import argparse,json,os,time,traceback
from pathlib import Path
from .common import atomic_json,OollimError,read_reference

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True,type=Path);a=p.parse_args();r=json.loads(a.request.read_text());events=Path(r['events']);started={}
 def emit(stage,status,detail):
  if status=='running':started[stage]=time.monotonic()
  record={'at':time.time(),'stage':stage,'status':status,'detail':detail,'elapsed':time.monotonic()-started.get(stage,time.monotonic())}
  with events.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n');f.flush()
 os.environ['CUDA_VISIBLE_DEVICES']=''
 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]=str(r['threads'])
 try:
  from .engine import assess
  result=assess(r['audio'],read_reference(r['transcript']),r['model'],r['temp'],emit,r['threads'])
  if r.get('granular'):
   from .granular import assess_segments
   result['granular']=assess_segments(Path(r['temp'])/'decoded.wav',read_reference(r['transcript']),r['model'],r['temp'],emit)
  atomic_json(r['result'],{'status':'completed',**result});return 0
 except Exception as exc:
  traceback.print_exc()
  if isinstance(exc,OollimError):error={'code':exc.code,'message':exc.message,'fix':exc.fix}
  elif isinstance(exc,MemoryError) or 'out of memory' in str(exc).lower() or 'allocate memory' in str(exc).lower():error={'code':'E_MEMORY','message':'Not enough memory for this recording.','fix':'Close memory-heavy applications or use a machine with more RAM. Do not keep retrying an unchanged long recording.'}
  else:error={'code':'E_INFERENCE','message':str(exc),'fix':'Run oollim.py --doctor and inspect the per-recording log. Report its error and model version; audio need not be shared.'}
  atomic_json(r['result'],{'status':'failed','error':error});return 1
if __name__=='__main__':raise SystemExit(main())
