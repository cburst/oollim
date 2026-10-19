"""Qt desktop frontend; scoring and resume use the same CLI pipeline."""
import copy
import csv
from pathlib import Path
import tempfile
import threading
import time

from PySide6.QtCore import QThread, Signal, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QComboBox,
    QPlainTextEdit, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QGridLayout, QGroupBox)
from .common import ROOT, STAGES

FAMILIES = ['speechocean','korean','aesop','koreanocean','aesopocean','koreanaesop','aesopkoreanocean']


class Runner(QThread):
    update = Signal(dict)
    result = Signal(int)

    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.args = args
        self.cancel = threading.Event()

    def run(self):
        from .cli import main
        try:
            code = main(self.args, event_callback=lambda view:self.update.emit(copy.deepcopy(view)), cancel_event=self.cancel)
        except BaseException as exc:
            self.update.emit({'error':{'message':str(exc)}})
            code = 1
        self.result.emit(code)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('oollim · Speech evaluation')
        self.setWindowIcon(QIcon(str(ROOT/'assets/oollim.png')))
        self.resize(1200, 245)
        self.runner = None
        self.reference_temp = None
        self.started = 0
        self.active_output = None
        self.active_reference = None
        self.active_audio = None
        self.report_path = None
        body = QWidget(); self.setCentralWidget(body)
        layout = QVBoxLayout(body); layout.setContentsMargins(26,22,26,22); layout.setSpacing(12)
        heading=QHBoxLayout();heading.setSpacing(12);heading.addStretch()
        logo=QLabel()
        ratio=self.devicePixelRatioF()
        logo_pixels=QPixmap(str(ROOT/'assets/oollim.png')).scaled(round(48*ratio),round(48*ratio),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
        logo_pixels.setDevicePixelRatio(ratio);logo.setPixmap(logo_pixels)
        title=QLabel('oollim');title.setStyleSheet('font-size: 30px; font-weight: 700; color: #163c58;')
        heading.addWidget(logo);heading.addWidget(title);heading.addStretch();layout.addLayout(heading)
        self.inputs = QWidget(); grid = QGridLayout(self.inputs)
        grid.setHorizontalSpacing(16); grid.setVerticalSpacing(10)
        for i,name in enumerate(['Audio','Reference text','Results','Scoring']):
            header=QLabel(name); header.setStyleSheet('font-weight: 600; padding: 5px; background: #dceefa; border-bottom: 1px solid #bad5e7;')
            grid.addWidget(header,0,i);grid.setColumnStretch(i,1)
        self.source=QLineEdit();self.source.setPlaceholderText('Choose one audio/video file')
        grid.addWidget(self.path_row(self.source,self.choose_source),1,0)
        self.refmode=QComboBox(self);self.refmode.addItems(['Paste reference text','Use a transcript file']);self.refmode.setCurrentIndex(1);self.refmode.hide()
        self.transcript=QLineEdit();self.transcript.setPlaceholderText('Reference.txt')
        self.transcript_row=self.path_row(self.transcript,self.choose_transcript);grid.addWidget(self.transcript_row,1,1)
        self.family=QComboBox();self.family.addItems(FAMILIES);grid.addWidget(self.family,1,3)
        self.output=QLineEdit(str(Path.home()/'oollim/results/scores.csv'))
        grid.addWidget(self.path_row(self.output,self.choose_output),1,2)
        self.text=QPlainTextEdit(self);self.text.setPlaceholderText('Paste the text spoken in the recording…');self.text.setMinimumHeight(130);self.text.setMaximumHeight(180)
        self.refmode.currentIndexChanged.connect(self.reference_mode);self.reference_mode()
        layout.addWidget(self.inputs)
        buttons=QHBoxLayout();buttons.setSpacing(16);buttons.addStretch()
        self.start=QPushButton('Evaluate');self.stop=QPushButton('Cancel');self.open_pdf=QPushButton('Open PDF')
        action_buttons=(self.start,self.stop,self.open_pdf)
        for button in action_buttons:button.ensurePolished()
        button_width=max(button.fontMetrics().horizontalAdvance(button.text()) for button in action_buttons)+40
        for button in action_buttons:
            button.setProperty('action',True);button.setFixedSize(button_width,42);buttons.addWidget(button)
        self.stop.setEnabled(False);self.open_pdf.setEnabled(False)
        self.start.clicked.connect(self.begin);self.stop.clicked.connect(self.cancel);self.open_pdf.clicked.connect(self.show_pdf)
        buttons.addStretch();layout.addLayout(buttons)
        for field in (self.source,self.transcript,self.output,self.refmode,self.family):field.setFixedHeight(36)
        # Retain controller state without showing the old dashboard panels.
        self.elapsed=QLabel(self);self.status=QLabel(self);self.stage=QLabel(self)
        self.progress=QProgressBar(self);self.table=QTableWidget(0,5,self)
        self.details=QPlainTextEdit(self)
        for widget in (self.elapsed,self.status,self.stage,self.progress,self.table,self.details):widget.hide()
        self.timer=QTimer(self);self.timer.timeout.connect(self.tick)
        self.setStyleSheet('QMainWindow {background: #f5f9fc;} QGroupBox {font-weight: 600;} QPushButton {padding: 8px 15px;} QPushButton[action="true"] {background: #b9e2fb; color: #163c58; border: 1px solid #84bddc; border-radius: 7px;} QPushButton[action="true"]:disabled {background: #e3edf3; color: #8b9ba6; border-color: #c9d7df;} QLineEdit, QPlainTextEdit, QComboBox {padding: 5px;}')

    def path_row(self, field, action):
        row=QWidget(); box=QHBoxLayout(row);box.setContentsMargins(0,0,0,0);box.addWidget(field)
        button=QPushButton('Browse…');button.setFixedHeight(36);button.clicked.connect(action);box.addWidget(button);return row

    def reference_mode(self):
        self.text.setVisible(self.refmode.currentIndex()==0)
        self.transcript_row.setVisible(self.refmode.currentIndex()==1)

    def choose_source(self):
        value=QFileDialog.getOpenFileName(self,'Choose one audio or video file')[0]
        if value:self.source.setText(value)

    def choose_transcript(self):
        value=QFileDialog.getOpenFileName(self,'Choose transcript',filter='Text files (*.txt);;All files (*)')[0]
        if value:self.transcript.setText(value)

    def choose_output(self):
        value=QFileDialog.getSaveFileName(self,'Save results CSV',self.output.text(),'CSV (*.csv)')[0]
        if value:self.output.setText(value)

    def begin(self):
        if self.runner and self.runner.isRunning():return
        try:
            source=Path(self.source.text()).expanduser()
            if not self.source.text().strip() or not source.is_file():
                raise ValueError('Choose an existing audio or video file.')
            if not self.output.text().strip():raise ValueError('Choose a CSV output path.')
            output=Path(self.output.text()).expanduser().resolve()
            if output.suffix.lower()!='.csv':raise ValueError('The output filename must end in .csv.')
            if output==source.resolve():raise ValueError('Results cannot overwrite an input file.')
            args=['--audio',str(source.resolve()),'--family',self.family.currentText(),'--output',str(output),'--no-live','--retry-failed','--granular']
            if self.refmode.currentIndex()==0:
                text=self.text.toPlainText().strip()
                if not text:raise ValueError('Paste the reference text or select another reference mode.')
                self.reference_temp=tempfile.TemporaryDirectory(prefix='oollim-reference-')
                ref=Path(self.reference_temp.name)/'reference.txt';ref.write_text(text,encoding='utf-8')
                args+=['--transcript',str(ref)]
            elif self.refmode.currentIndex()==1:
                ref=Path(self.transcript.text()).expanduser()
                if not self.transcript.text().strip() or not ref.is_file():raise ValueError('Choose an existing UTF-8 transcript file.')
                if output==ref.resolve():raise ValueError('Results cannot overwrite the transcript.')
                args+=['--transcript',str(ref.resolve())]
            self.active_reference=ref.read_text(encoding="utf-8").strip()
            self.active_audio=source.resolve()
            self.report_path=None;self.open_pdf.setEnabled(False)
            self.active_output=output
            self.details.clear();self.inputs.setEnabled(False);self.start.setEnabled(False);self.stop.setEnabled(True)
            self.status.setText('Checking runtime and model assets…');self.progress.setRange(0,0)
            self.start.setText('Evaluating…');self.started=time.monotonic();self.timer.start(1000)
            self.runner=Runner(args,self);self.runner.update.connect(self.update_view);self.runner.result.connect(self.finished)
            self.runner.start()
        except (OSError,ValueError) as exc:
            if self.reference_temp:self.reference_temp.cleanup();self.reference_temp=None
            QMessageBox.warning(self,'Check inputs',str(exc))

    def update_view(self,view):
        if view.get('error'):
            message=view['error'].get('message','Unknown error')
            if message not in self.details.toPlainText():self.details.appendPlainText(message)
        if 'total' not in view:return
        self.load_results()
        self.progress.setRange(0,view['total'])
        self.progress.setValue(view.get('processed',0))
        self.status.setText(f"{view['current']} · {view['done']} new · {view['skipped']} reused · {view['failed']} failed")
        stages=view.get('stages',[])
        active=[STAGES[i] for i,s in enumerate(stages) if s.get('status')=='running']
        self.stage.setText(' → '.join(active) or 'Preparing / saving results')

    def load_results(self):
        if not self.active_output or not self.active_output.exists():return
        try:
            with self.active_output.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        except OSError:return
        self.table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            for j,key in enumerate(['audio','status','accuracy','fluency','prosody']):
                value=row.get(key,'')
                if key=='audio':value=Path(value).name
                elif key in ('accuracy','fluency','prosody') and value:
                    try:value=f'{float(value):.3f}'
                    except ValueError:pass
                self.table.setItem(i,j,QTableWidgetItem(value))

    def tick(self):
        seconds=int(time.monotonic()-self.started);self.setWindowTitle(f'oollim · Evaluating · {seconds//60:02d}:{seconds%60:02d} · {self.stage.text()}')

    def cancel(self):
        if self.runner:self.runner.cancel.set();self.stop.setEnabled(False);self.status.setText('Cancelling safely; completed results will be retained…')

    def finished(self,code):
        self.timer.stop();self.start.setText('Evaluate');self.setWindowTitle('oollim · Speech evaluation');self.inputs.setEnabled(True);self.start.setEnabled(True);self.stop.setEnabled(False)
        self.load_results();self.progress.setRange(0, max(1,self.table.rowCount()));self.progress.setValue(self.table.rowCount())
        self.status.setText('Completed. Scores are saved to CSV.' if code==0 else 'Cancelled. Rerun with the same CSV to resume.' if code==130 else 'Some work failed. Read diagnostics and retry failed recordings.')
        self.stage.setText('Ready')
        if code==0:
            self.inputs.setEnabled(False);self.start.setEnabled(False)
            try:
                from .report import write_report
                with self.active_output.open(newline='',encoding='utf-8') as handle:
                    rows=[r for r in csv.DictReader(handle) if Path(r['audio']).resolve()==self.active_audio and r['status']=='completed']
                if len(rows)!=1:raise ValueError('Could not identify the completed recording for the PDF.')
                import json
                state=json.loads(self.active_output.with_suffix('.csv.state.json').read_text())
                detail=state[str(self.active_audio)].get('granular')
                self.report_path=write_report(self.active_output.with_suffix('.pdf'),rows[0],self.active_reference,detail)
                self.open_pdf.setEnabled(True)
                self.status.setText('Completed. CSV and PDF report saved.')
            except Exception as exc:
                self.status.setText('Scores saved to CSV; PDF export failed.')
                self.details.appendPlainText(f'PDF export: {exc}')
            finally:
                self.inputs.setEnabled(True);self.start.setEnabled(True)
        if code not in (0,130) or (code==0 and not self.report_path):
            self.message=QMessageBox(QMessageBox.Icon.Warning,'Evaluation needs attention',self.details.toPlainText() or self.status.text(),parent=self)
            self.message.open()
        if self.reference_temp:self.reference_temp.cleanup();self.reference_temp=None

    def show_pdf(self):
        if self.report_path:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report_path)))

    def open_folder(self):
        path=(self.active_output or Path(self.output.text()).expanduser()).parent
        if path.exists():QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def closeEvent(self,event):
        if self.runner and self.runner.isRunning():
            self.cancel();event.ignore()
            self.status.setText('Stopping the worker. Close the window again once cancellation completes.')
        else:event.accept()


def main():
    import sys
    if sys.platform=='win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('ai.interlanguage.oollim')
    app=QApplication.instance() or QApplication([])
    app.setApplicationName('oollim');app.setWindowIcon(QIcon(str(ROOT/'assets/oollim.png')))
    window=Window();window.show()
    return app.exec()
