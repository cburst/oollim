"""Pinned acquisition using Python HTTPS, portable extraction and tqdm."""
import argparse,hashlib,json,os,shutil,tarfile,tempfile,urllib.request
from pathlib import Path,PurePosixPath
from .common import ROOT,RESOURCES,ESP,sha,atomic_json,OollimError

def items():
 return [i for i in json.loads((ROOT/'speech/resources-manifest.json').read_text())['resources'] if i['path'].startswith(('xeus/','xlsr-300m/','phoneticxeus/','text-reference-v1/')) or i['id']=='espnet-xeus-source']

def download(item,destination,progress=None):
 destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.exists() and sha(destination)==item['sha256']:
  atomic_json(destination.with_name(destination.name+'.receipt.json'),item);return 'cached'
 part=destination.with_name(destination.name+'.part');offset=part.stat().st_size if part.exists() else 0
 if offset and item.get('bytes')==offset and sha(part)==item['sha256']:
  part.replace(destination);atomic_json(destination.with_name(destination.name+'.receipt.json'),item);return 'resumed'
 if offset and item.get('bytes') and offset>=item['bytes']:
  part.unlink();offset=0
 headers={'User-Agent':'oollim-installer/0.2'}
 if offset:headers['Range']=f'bytes={offset}-'
 try:
  with urllib.request.urlopen(urllib.request.Request(item['url'],headers=headers),timeout=90) as response:
   append=offset>0 and response.status==206
   if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Server returned an invalid resume range')
   if not append:offset=0
   total=item.get('bytes') or (int(response.headers['Content-Length'])+offset if response.headers.get('Content-Length') else None)
   from tqdm import tqdm
   with tqdm(total=total,initial=offset,unit='B',unit_scale=True,desc=item['id'],leave=False) as bar,part.open('ab' if append else 'wb') as f:
    while True:
     block=response.read(1024*1024)
     if not block:break
     f.write(block);bar.update(len(block))
 except Exception as exc:
  raise OollimError('E_DOWNLOAD',f'Could not download {item["id"]}: {exc}','Check the connection/proxy and rerun. Partial downloads are retained; no unverified file is used.') from exc
 if sha(part)!=item['sha256']:
  part.unlink(missing_ok=True)
  raise OollimError('E_CHECKSUM',f'Download checksum mismatch: {item["id"]}.','Rerun the download. If it repeats, report the resource ID; do not bypass verification.')
 part.replace(destination);atomic_json(destination.with_name(destination.name+'.receipt.json'),item);return 'downloaded'

def unpack_esp(archive,target):
 target=Path(target);receipt=target/'.oollim-source.json';expected=sha(archive)
 if receipt.exists():
  record=json.loads(receipt.read_text())
  if record.get('archive_sha256')==expected and all((target/name).is_file() and sha(target/name)==digest for name,digest in record.get('files',{}).items()) and record.get('files'):return
 if target.exists():raise OollimError('E_SOURCE_CHANGED',f'ESPnet source directory is incomplete or changed: {target}','Move that directory aside and rerun installation; existing source is not silently overwritten.')
 target.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=target.parent,prefix='espnet-install-') as temp:
  staging=Path(temp)/'source';staging.mkdir()
  with tarfile.open(archive,'r:gz') as tf:
   for member in tf:
    if PurePosixPath(member.name).is_absolute() or '..' in PurePosixPath(member.name).parts:raise ValueError('Unsafe archive path')
    parts=PurePosixPath(member.name).parts
    if len(parts)<2:continue
    relative=PurePosixPath(*parts[1:])
    if relative.is_absolute() or '..' in relative.parts or '\\' in member.name or ':' in member.name:raise ValueError('Unsafe archive path')
    # Only the importable packages and notices are needed. Never extract links/devices.
    if relative.parts[0] not in ('espnet','espnet2','LICENSE','NOTICE'):continue
    if not member.isfile():
     if member.isdir():continue
     raise ValueError('Unsupported archive entry')
    path=staging/relative;path.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(member) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
  files={str(p.relative_to(staging)):sha(p) for p in staging.rglob('*') if p.is_file()}
  if not (staging/'espnet2/asr/frontend/cnn.py').exists():raise ValueError('ESPnet source layout differs')
  atomic_json(staging/'.oollim-source.json',{'archive_sha256':expected,'files':files});staging.replace(target)

def setup(source_only=False):
 from tqdm import tqdm
 selected=[i for i in items() if not source_only or i['id']=='espnet-xeus-source']
 for item in tqdm(selected,desc='Verified resources',unit='file'):
  download(item,RESOURCES/item['path'])
 unpack_esp(RESOURCES/'espnet-xeus-source.tar.gz',ESP)

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-only',action='store_true');a=p.parse_args();setup(a.source_only)
if __name__=='__main__':main()
