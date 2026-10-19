"""Run one public fixture through the real GUI/controller and save evidence."""
import argparse,csv,json,platform,sys,time
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--app-root',type=Path,default=ROOT);p.add_argument('--audio',required=True);p.add_argument('--transcript',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.app_root))
from oollim_app.gui import Window
a.output.mkdir(parents=True,exist_ok=True)
app=QApplication([]);window=Window();window.show()
window.source.setText(a.audio);window.refmode.setCurrentIndex(1);window.transcript.setText(a.transcript);window.output.setText(str(a.output/'gui-scores.csv'))
started=time.monotonic();report={'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,'qt_platform':app.platformName()}
def finish(code):
 report.update(exit_code=code,seconds=round(time.monotonic()-started,2))
 path=a.output/'gui-scores.csv'
 if path.exists():
  with path.open(newline='',encoding='utf-8') as f:report['rows']=list(csv.DictReader(f))
 report['passed']=code==0 and len(report.get('rows',[]))==1 and report['rows'][0]['status']=='completed'
 window.grab().save(str(a.output/'gui-result.png'))
 (a.output/'gui-report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True)
 window.runner.finished.connect(app.quit)
 QTimer.singleShot(1000,lambda:app.quit() if not window.runner.isRunning() else None)
def begin():
 window.grab().save(str(a.output/'gui-start.png'))
 window.begin()
 if window.runner:window.runner.result.connect(finish)
 else:app.exit(2)
QTimer.singleShot(500,begin)
QTimer.singleShot(1200000,window.cancel)
app.exec()
sys.exit(0 if report.get('passed') else 1)
