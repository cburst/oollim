"""Pinned WeasyPrint setup and native-library discovery."""
import hashlib,os,platform,shutil,subprocess,sys,tempfile,urllib.request,zipfile
from pathlib import Path
VERSION='70.0'
URL='https://github.com/Kozea/WeasyPrint/releases/download/v70.0/weasyprint-windows-onedir.zip'
SHA256='ab1151f210b4e6bb7aa7a79e91a67e8ddb760094c107bfda55241b6aaefe7d53'
RUNTIME=Path.home()/'oollim'/'pdf-runtime'/('weasyprint-'+VERSION)


def windows_exe():
    candidates=list(RUNTIME.rglob('weasyprint.exe')) if RUNTIME.exists() else []
    if len(candidates)!=1:raise RuntimeError('WeasyPrint Windows runtime is missing. Rerun install.py to install the verified PDF renderer.')
    return candidates[0]


def environment():
    env=os.environ.copy();env['PYTHONUTF8']='1'
    if sys.platform=='darwin':
        env['DYLD_FALLBACK_LIBRARY_PATH']='/opt/homebrew/lib:/usr/local/lib:'+env.get('DYLD_FALLBACK_LIBRARY_PATH','')
    return env


def setup():
    if os.name=='nt':
        if not RUNTIME.exists():
            RUNTIME.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=RUNTIME.parent) as directory:
                root=Path(directory);archive=root/'runtime.zip'
                urllib.request.urlretrieve(URL,archive)
                if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:raise RuntimeError('WeasyPrint runtime checksum failed; download rejected.')
                extracted=root/'unpacked';extracted.mkdir()
                with zipfile.ZipFile(archive) as zipped:
                    for item in zipped.infolist():
                        if not (extracted/item.filename).resolve().is_relative_to(extracted.resolve()):raise RuntimeError('Invalid PDF runtime archive path.')
                    zipped.extractall(extracted)
                extracted.replace(RUNTIME)
        command=[str(windows_exe())]
    else:
        if sys.platform=='darwin' and not Path('/opt/homebrew/lib/libpango-1.0.dylib').exists():
            brew=shutil.which('brew') or ('/opt/homebrew/bin/brew' if Path('/opt/homebrew/bin/brew').exists() else None)
            if not brew:raise RuntimeError('PDF rendering needs Pango. Install Homebrew from https://brew.sh, then rerun the installer; it will install Pango automatically.')
            subprocess.run([brew,'install','pango'],check=True)
        command=[sys.executable,'-m','weasyprint']
    with tempfile.TemporaryDirectory() as directory:
        source=Path(directory)/'check.html';target=Path(directory)/'check.pdf'
        source.write_text('<p>oollim PDF runtime check</p>',encoding='utf-8')
        subprocess.run(command+['--encoding','utf-8',str(source),str(target)],env=environment(),check=True,timeout=60)
        if not target.read_bytes().startswith(b'%PDF'):raise RuntimeError('WeasyPrint PDF smoke test failed.')
    print('WeasyPrint '+VERSION+' PDF rendering verified.')

if __name__=='__main__':setup()
