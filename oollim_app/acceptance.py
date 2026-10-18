#!/usr/bin/env python3
"""Run a native-platform acceptance check with your own valid audio/reference."""
import argparse,csv,json,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audio',type=Path,required=True);p.add_argument('--transcript',type=Path,required=True);a=p.parse_args(argv)
 config=ROOT/'.oollim-install.json'
 if not config.exists():p.error('Run install.py first.')
 python=json.loads(config.read_text())['python'];out=ROOT/'work/acceptance';out.mkdir(parents=True,exist_ok=True)
 stamp=time.strftime('%Y%m%d-%H%M%S');csvpath=out/(stamp+'.csv');logpath=out/(stamp+'.log')
 report={'platform':platform.platform(),'machine':platform.machine(),'python':python,'checks':[]}
 commands=[('doctor',[python,'oollim.py','--doctor']),('inference',[python,'oollim.py','--audio',str(a.audio.resolve()),'--transcript',str(a.transcript.resolve()),'--output',str(csvpath),'--no-live'])]
 commands.append(('resume',commands[-1][1]))
 with logpath.open('w',encoding='utf-8') as log:
  for name,cmd in commands:
   print('Checking '+name+' ...',flush=True);start=time.monotonic();run=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   report['checks'].append({'name':name,'exit_code':run.returncode,'seconds':round(time.monotonic()-start,2)})
   if run.returncode:break
 if csvpath.exists():
  with csvpath.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
  report['completed_rows']=sum(r['status']=='completed' for r in rows)
 report['passed']=len(report['checks'])==3 and all(r['exit_code']==0 for r in report['checks']) and report.get('completed_rows')==1
 result=out/(stamp+'.json');result.write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(('PASS' if report['passed'] else 'FAIL')+f' — report: {result}\nLog: {logpath}')
 return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
