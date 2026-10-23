import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication
from oollim_app.gui import Window, Runner
from oollim_app import cli

@pytest.fixture
def window():
    app=QApplication.instance() or QApplication([])
    w=Window()
    yield w
    w.close()

def test_reference_modes_and_progress(window):
    window.refmode.setCurrentIndex(0)
    assert not window.text.isHidden() and window.transcript_row.isHidden()
    window.refmode.setCurrentIndex(1)
    assert not window.transcript_row.isHidden()
    window.update_view(dict(total=5,processed=2,current='sample.wav',done=1,skipped=0,failed=1,stages=[]))
    assert window.progress.value()==2

def test_results_and_error_display(window,tmp_path):
    p=tmp_path/'scores.csv';p.write_text('audio,status,accuracy,fluency,prosody\nsample.wav,completed,8.12345,7,9\n')
    window.active_output=p;window.load_results()
    assert window.table.rowCount()==1
    assert window.table.item(0,2).text()=='8.123'
    window.update_view({'error':{'message':'Missing encoder. Download models.'}})
    assert 'Download models' in window.details.toPlainText()

def test_runner_uses_existing_pipeline_and_cancellation(monkeypatch):
    seen={}
    def fake_main(args, event_callback, cancel_event):
        seen['args']=args;seen['cancel']=cancel_event.is_set()
        event_callback({'processed':1})
        return 130
    monkeypatch.setattr(cli,'main',fake_main)
    runner=Runner(['--audio','example.wav']);runner.cancel.set()
    results=[];views=[];runner.result.connect(results.append);runner.update.connect(views.append)
    runner.run()
    assert results==[130] and views==[{'processed':1}]
    assert seen['cancel'] and seen['args']==['--audio','example.wav']


def test_cli_cancellation_releases_output_lock(tmp_path,monkeypatch):
    import sys,types,threading
    from oollim_app import common
    engine=types.ModuleType('oollim_app.engine')
    engine.runtime_check=lambda:None
    engine.verify_resources=lambda:None
    engine.load_bundle=lambda path:None
    monkeypatch.setitem(sys.modules,'oollim_app.engine',engine)
    monkeypatch.setattr(common,'platform_check',lambda:None)
    audio=tmp_path/'sample.wav';audio.write_bytes(b'fake audio')
    output=tmp_path/'scores.csv'
    event=threading.Event();event.set()
    assert cli.main(['--audio',str(audio),'--output',str(output),'--no-live'],cancel_event=event)==130
    assert not output.with_suffix('.csv.lock').exists()
    assert output.with_suffix('.csv.state.json').exists()

def test_gui_routes_selection_to_cli(window,tmp_path,monkeypatch):
    from oollim_app import gui
    audio=tmp_path/'clip.wav';audio.write_bytes(b'fake')
    ref=tmp_path/'ref.txt';ref.write_text('Please call Stella.')
    window.source.setText(str(audio));window.transcript.setText(str(ref));window.output.setText(str(tmp_path/'out.csv'))
    class Signal:
        def connect(self,*args):pass
    class FakeRunner:
        update=Signal();result=Signal()
        def __init__(self,args,parent):self.args=args
        def start(self):pass
        def isRunning(self):return False
    monkeypatch.setattr(gui,'Runner',FakeRunner)
    for family in gui.FAMILIES:
        for selection in gui.SELECTIONS:
            window.scoring.setCurrentIndex(window.scoring.findData((family,selection)))
            window.begin();args=window.runner.args
            assert args[args.index('--family')+1]==family
            assert args[args.index('--selection')+1]==selection
            window.timer.stop()
