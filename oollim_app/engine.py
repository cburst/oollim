"""Local inference orchestration around the unchanged research encoder arithmetic."""
import gc,json,subprocess,sys
from pathlib import Path
from .common import ROOT,RESOURCES,ESP,sha,imports_path,OollimError

def load_bundle(path):
 path=Path(path).resolve()
 if not path.is_file():raise OollimError('E_MODEL_MISSING',f'Model bundle is missing: {path}','Select speechocean, or supply an authorized family.json bundle with --model.')
 cfg=json.loads(path.read_text(encoding='utf-8'))
 if cfg.get('schema_version')!='oollim-family-v1' or set(cfg.get('heads',{}))!={'accuracy','fluency','prosody'}:raise OollimError('E_MODEL_SCHEMA','Unsupported or incomplete model bundle.','Use an oollim-family-v1 bundle with all three construct heads.')
 for name,digest in cfg['files'].items():
  p=(path.parent/name).resolve()
  if p.parent!=path.parent or not p.is_file() or sha(p)!=digest:raise OollimError('E_MODEL_CHECKSUM',f'Missing or changed scoring asset: {name}','Restore the original bundle; do not edit checkpoint metadata or bypass checksums.')
 for name in cfg['heads'].values():
  if name not in cfg['files'] or str(Path(name).with_suffix('.json')) not in cfg['files']:raise OollimError('E_MODEL_SCHEMA','Head is not covered by the checksum manifest.','Obtain a complete model bundle.')
 reference=json.loads((ROOT/'configs/extractor.json').read_text())
 for name in set(cfg['heads'].values()):
  head=json.loads((path.parent/Path(name).with_suffix('.json')).read_text())
  if head.get('experiment_provenance')!=reference['experiment_provenance'] or any(reference['block_widths'].get(b)!=head['block_widths'].get(b) for b in head['blocks']):raise OollimError('E_MODEL_INCOMPATIBLE','Scoring heads require different encoder features.','Use a bundle compatible with this pinned extraction pipeline.')
 return cfg

def runtime_check():
 imports_path()
 import torch,numpy,scipy,soundfile,rich,tqdm,transformers,safetensors
 from transformers import Wav2Vec2Model
 from espnet2.asr.frontend.cnn import CNNFrontend
 from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
 from espnet2.asr.preencoder.linear import LinearProjection
 from imageio_ffmpeg import get_ffmpeg_exe
 ffmpeg=get_ffmpeg_exe();subprocess.run([ffmpeg,'-version'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,timeout=20)
 return {'python':sys.version.split()[0],'torch':torch.__version__,'ffmpeg':ffmpeg,'device':'CPU'}

def verify_resources():
 from .assets import items
 missing=[]
 for item in items():
  p=RESOURCES/item['path']
  if not p.exists():missing.append(item['id'])
  elif sha(p)!=item['sha256']:raise OollimError('E_ENCODER_CHECKSUM',f'Encoder asset changed: {item["id"]}','Run oollim.py --download-models to restore the pinned file.')
 if missing:raise OollimError('E_ENCODERS_MISSING','Missing resources: '+', '.join(missing[:6])+(' ...' if len(missing)>6 else ''),'Run oollim.py --download-models; several GB of disk space and an internet connection are required.')
 from .assets import unpack_esp
 unpack_esp(RESOURCES/'espnet-xeus-source.tar.gz',ESP)

def decode(source,destination):
 from imageio_ffmpeg import get_ffmpeg_exe
 import soundfile as sf
 args=[get_ffmpeg_exe(),'-nostdin','-hide_banner','-loglevel','error','-protocol_whitelist','file,pipe','-format_whitelist','wav,mp3,flac,ogg,mov,matroska,webm,aac,aiff,asf,amr','-i',str(source),'-map','0:a:0','-vn','-t','121','-c:a','pcm_f32le','-fs','134217728','-f','wav','-y',str(destination)]
 try:result=subprocess.run(args,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=90)
 except subprocess.TimeoutExpired as exc:raise OollimError('E_DECODE_TIMEOUT','FFmpeg did not finish decoding within 90 seconds.','Check that this is a local readable file; try exporting it as WAV.') from exc
 if result.returncode:
  detail=result.stderr.decode('utf-8',errors='replace')[-1800:]
  noaudio='matches no streams' in detail.lower() or 'does not contain any stream' in detail.lower()
  raise OollimError('E_NO_AUDIO_STREAM' if noaudio else 'E_DECODE',('No audio track was found.' if noaudio else 'FFmpeg could not decode the file.')+' '+detail,'Supply a file containing an audio track, or re-export damaged/unsupported media as WAV.')
 info=sf.info(destination);duration=info.frames/info.samplerate
 if destination.stat().st_size>=134152192:raise OollimError('E_DECODE_SIZE','Decoded audio exceeds the safe buffer size.','Export mono or stereo audio before retrying.')
 if not .25<=duration<=120:raise OollimError('E_DURATION',f'Audio duration is {duration:.2f} seconds; supported range is 0.25–120 seconds.','Supply a recording within this range; oollim does not silently trim or split it.')
 if not 1<=info.channels<=8:raise OollimError('E_CHANNELS',f'Audio has {info.channels} channels.','Export mono or stereo audio.')
 return duration

def assess(audio,reference,bundle_path,temp,event,threads=4):
 imports_path()
 import torch,numpy as np
 from pronunciation.fusion import UnifiedScorer,DIMENSIONS
 from pronunciation.inference_memory import install
 torch.set_num_threads(threads);torch.set_num_interop_threads(threads);cleanup=install()
 event(0,'running','Checking recording, transcript and scoring heads')
 cfg=load_bundle(bundle_path);audio=Path(audio)
 if not audio.is_file() or not 0<audio.stat().st_size<=400*1024**2:raise OollimError('E_AUDIO_SIZE','Audio is empty, missing, or exceeds 400 MiB.','Supply a readable local media file smaller than 400 MiB.')
 if not reference.strip():raise OollimError('E_REFERENCE_EMPTY','Reference text is empty.','Supply the intended spoken text.')
 heads={d:UnifiedScorer(Path(bundle_path).parent/name) for d,name in cfg['heads'].items()}
 from pronunciation.text_reference import text_to_phones,load_dictionary
 _,coverage=text_to_phones(reference,load_dictionary(RESOURCES/'text-reference-v1/cmudict.dict'))
 if not coverage['complete']:raise OollimError('E_REFERENCE_VOCABULARY','Reference contains words outside the pinned pronunciation dictionary.','Check spelling; write numbers and abbreviations as spoken words.')
 event(0,'done','Audio and reference supplied; head checksums verified')
 event(1,'running','FFmpeg decoding; timing and silence preserved')
 decoded=Path(temp)/'decoded.wav';duration=decode(audio,decoded);event(1,'done',f'{duration:.2f} seconds; whole recording')
 meta=json.loads((ROOT/'configs/extractor.json').read_text());needed=set(k for h in heads.values() for k in h.blocks);blocks={}
 for stage,names,label in [(2,['xeus'],'XEUS'),(3,['xlsr'],'XLS-R'),(4,['phoneticxeus','ipa'],'PhoneticXeus and reference-phone evidence')]:
  selected=[n for n in names if n in needed]
  if not selected:event(stage,'skipped','Not required by selected heads');continue
  event(stage,'running',f'Loading and extracting {label}; CPU')
  model=UnifiedScorer.__new__(UnifiedScorer);model.metadata=meta;model.blocks=selected;model.widths=meta['block_widths'];model.width=sum(model.widths[n] for n in selected)
  values,status=model.extract(decoded,reference,RESOURCES)
  if not status['text']['complete'] or (stage==4 and not status['phone']['reference_complete']):raise OollimError('E_REFERENCE_VOCABULARY','The reference contains words not covered by the pinned pronunciation dictionary.','Check spelling and write numbers/abbreviations as spoken words. Do not substitute recognized text silently.')
  offset=0
  for name in selected:blocks[name]=values[:,offset:offset+model.widths[name]];offset+=model.widths[name]
  del model,values;gc.collect();event(stage,'done',f'{label} features ready')
 event(5,'running','Applying the selected ridge for each construct');scores={}
 for j,d in enumerate(DIMENSIONS):
  head=heads[d];x=np.concatenate([blocks[n] for n in head.blocks],axis=1);scores[d]=float(head.predict_features(x)[0,j])
 event(5,'done','Raw 0–10-scale predictions; no calibration or clipping')
 cleanup.remove()
 return {'scores':scores,'duration_seconds':duration,'family':cfg['family'],'model_version':cfg['model_version'],'score_scale':'0–10 scale; raw ridge predictions may exceed the endpoints'}
