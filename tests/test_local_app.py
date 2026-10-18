import json
from pathlib import Path
import pytest
from oollim_app.common import OollimError,platform_check,atomic_json
from oollim_app import cli

def test_platform():
 platform_check('Darwin','arm64');platform_check('Windows','AMD64')
 with pytest.raises(OollimError):platform_check('Darwin','x86_64')

def test_bundle():
 from oollim_app.engine import load_bundle
 cfg=load_bundle(Path(__file__).parents[1]/'models/speechocean/family.json')
 assert cfg['family']=='speechocean'

def test_resume_identity(tmp_path):
 audio=tmp_path/'audio';audio.write_bytes(b'a');model=tmp_path/'family.json';model.write_text('{}')
 first=cli.identity(audio,'reference',model)
 assert first==cli.identity(audio,'reference',model)
 assert first!=cli.identity(audio,'different',model)
 audio.write_bytes(b'b');assert first!=cli.identity(audio,'reference',model)

def test_csv_roundtrip(tmp_path):
 import csv
 output=tmp_path/'scores.csv'
 records={'one':{'audio':'한 글, file.wav','status':'completed','scores':{'accuracy':8.1,'fluency':7.2,'prosody':6.3}}}
 cli.save(output,records)
 assert json.loads(output.with_suffix('.csv.state.json').read_text())==records
 with output.open() as f:row=next(csv.DictReader(f))
 assert row['accuracy']=='8.1' and row['audio']=='한 글, file.wav'

def test_batch_resume_and_retry(tmp_path,monkeypatch):
 from oollim_app import common,engine
 monkeypatch.setattr(common,'platform_check',lambda:None)
 monkeypatch.setattr(engine,'runtime_check',lambda:None);monkeypatch.setattr(engine,'verify_resources',lambda:None);monkeypatch.setattr(engine,'load_bundle',lambda p:None)
 audio=tmp_path/'speech';audio.mkdir();(audio/'one.wav').write_bytes(b'a');(audio/'one.txt').write_text('Please call Stella.')
 (audio/'two.wav').write_bytes(b'b')
 model=tmp_path/'family.json';model.write_text('{}');calls=[]
 class Process:
  returncode=0
  def __init__(self,args,**kwargs):
   r=json.loads(Path(args[-1]).read_text());calls.append(r)
   atomic_json(r['result'],{'status':'completed','scores':{'accuracy':8,'fluency':7,'prosody':6}})
  def poll(self):return 0
 monkeypatch.setattr(cli.subprocess,'Popen',Process)
 out=tmp_path/'result.csv';args=['--folder',str(audio),'--model',str(model),'--output',str(out),'--no-live']
 assert cli.main(args)==1;assert len(calls)==1
 assert cli.main(args)==1;assert len(calls)==1
 (audio/'two.txt').write_text('Reference')
 assert cli.main(args)==0;assert len(calls)==2
 (audio/'one.txt').write_text('Changed reference')
 assert cli.main(args)==0;assert len(calls)==3
 assert not out.with_suffix('.csv.lock').exists()

def test_reject_bad_bundle(tmp_path):
 from oollim_app.engine import load_bundle
 path=tmp_path/'family.json';path.write_text('{}')
 with pytest.raises(OollimError,match='E_MODEL_SCHEMA'):load_bundle(path)

def test_download_bad_complete_part(tmp_path,monkeypatch):
 import hashlib,io
 from oollim_app import assets
 target=tmp_path/'file';target.with_name('file.part').write_bytes(b'bad')
 data=b'yes';item={'id':'fixture','url':'https://example.invalid/file','bytes':3,'sha256':hashlib.sha256(data).hexdigest()}
 class Response(io.BytesIO):
  status=200;headers={}
 def open_url(request,**kwargs):
  assert not request.has_header('Range')
  return Response(data)
 monkeypatch.setattr(assets.urllib.request,'urlopen',open_url)
 assets.download(item,target);assert target.read_bytes()==data

def test_archive_rejects_traversal(tmp_path):
 import tarfile,io
 from oollim_app.assets import unpack_esp
 archive=tmp_path/'bad.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  info=tarfile.TarInfo('../espnet2/test.py');info.size=1;tf.addfile(info,io.BytesIO(b'x'))
 with pytest.raises(ValueError,match='Unsafe'):unpack_esp(archive,tmp_path/'source')
 assert not (tmp_path/'source').exists()

def test_csv_restored_from_state(tmp_path,monkeypatch):
 records={'a':{'audio':'a','status':'completed','scores':{'accuracy':5}}}
 output=tmp_path/'results.csv';cli.save(output,records);output.unlink();cli.save(output,json.loads(output.with_suffix('.csv.state.json').read_text()))
 assert output.is_file()

def test_all_family_bundles():
 from oollim_app.engine import load_bundle
 paths=list((Path(__file__).parents[1]/'models').glob('*/family.json'))
 assert len(paths)==7
 for p in paths:assert load_bundle(p)['family']==p.parent.name

def test_decode_formats(tmp_path):
 ff=pytest.importorskip('imageio_ffmpeg')
 import numpy as np,soundfile as sf,subprocess
 from oollim_app.engine import decode
 source=tmp_path/'한 글.wav';sf.write(source,np.zeros(16000),16000)
 assert decode(source,tmp_path/'decoded.wav')==1
 video=tmp_path/'container.mp4';subprocess.run([ff.get_ffmpeg_exe(),'-v','error','-i',str(source),'-c:a','aac',str(video)],check=True)
 assert .9<decode(video,tmp_path/'decoded-video.wav')<1.1
 invalid=tmp_path/'bad.mp3';invalid.write_bytes(b'not audio')
 with pytest.raises(OollimError,match='E_DECODE'):decode(invalid,tmp_path/'bad.wav')
 sf.write(source,np.zeros(121*16000),16000)
 with pytest.raises(OollimError,match='E_DURATION'):decode(source,tmp_path/'long.wav')
