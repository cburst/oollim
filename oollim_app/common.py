"""Dependency-free paths, hashes, atomic state, and user-facing diagnostics."""
import hashlib,json,os,platform,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESOURCES=ROOT/'speech/resources'
ESP=RESOURCES/'source/wanchichen-espnet-5b52d57'
STAGES=('Validate audio and reference','Decode and prepare audio','Extract XEUS representation','Extract XLS-R representation','Extract phone and IPA evidence','Calculate three scores','Save result and resume record')
class OollimError(Exception):
 def __init__(self,code,message,fix):
  self.code,self.message,self.fix=code,message,fix
  super().__init__(f'{code}: {message}\nFix: {fix}')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
 with temp.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
 temp.replace(path)
def platform_check(system=None,machine=None):
 system=system or platform.system();machine=(machine or platform.machine()).lower()
 if (system,machine) not in [('Darwin','arm64'),('Windows','amd64'),('Windows','x86_64')]:
  raise OollimError('E_PLATFORM',f'Unsupported platform: {system} {machine}.','Use native ARM64 Python on Apple Silicon macOS, or AMD64 Python on Windows 11. Rosetta, Intel Macs and WSL are not the initial supported targets.')
 if struct.calcsize('P')!=8:raise OollimError('E_PYTHON_ARCH','Python is not 64-bit.','Install a native 64-bit Python 3.11 interpreter.')
def imports_path():
 for path in (ROOT/'speech',ESP):
  if str(path) not in sys.path:sys.path.insert(0,str(path))
def read_reference(path):
 try:text=Path(path).read_text(encoding='utf-8-sig').strip()
 except (OSError,UnicodeError) as exc:raise OollimError('E_REFERENCE',f'Cannot read UTF-8 transcript: {path}.','Provide a readable UTF-8 .txt file with the spoken reference text.') from exc
 if not text:raise OollimError('E_REFERENCE_EMPTY',f'Transcript is empty: {path}.','Add the reference text; automatic transcription is not substituted.')
 return text
