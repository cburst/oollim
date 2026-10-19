#!/bin/bash
# Standalone interactive signing assistant. Run from Terminal or double-click.
set -euo pipefail
exec /usr/bin/python3 - "$@" <<'PY'
import argparse, hashlib, json, os, plistlib, re, shutil, subprocess, sys, tempfile
from pathlib import Path

parser=argparse.ArgumentParser(description='Sign and notarize the oollim Platypus installer.')
parser.add_argument('app', nargs='?', help='Path to Install oollim.app')
parser.add_argument('--check-only', action='store_true', help='Inspect structure and tools without signing or uploading')
a=parser.parse_args()
tty=None

def ask(prompt):
    global tty
    if tty is None: tty=open('/dev/tty','r')
    with open('/dev/tty','w') as terminal:
        terminal.write(prompt);terminal.flush()
    value=tty.readline()
    if not value: raise RuntimeError('Input closed. Run this script in an interactive Terminal.')
    return value.strip()

def run(args,capture=False):
    result=subprocess.run([str(x) for x in args],stdin=tty if tty else None,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None,text=True)
    if result.returncode:
        if capture: print(result.stderr or result.stdout)
        raise RuntimeError(f'{Path(str(args[0])).name} failed (exit {result.returncode}). See the diagnostic above; no success is assumed.')
    return result.stdout if capture else ''

def manifest(stage):
    data={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(stage.rglob('*')) if p.is_file() and p.name!='payload.json'}
    (stage/'payload.json').write_text(json.dumps(data,indent=2)+'\n')

def notarize(app,tag):
    archive=work/(tag+'-submission.zip')
    run(['/usr/bin/ditto','-c','-k','--keepParent',app,archive])
    print(f'\nSubmitting {app.name} to Apple. This may take several minutes.')
    result=run(['/usr/bin/xcrun','notarytool','submit',archive,'--keychain-profile',profile,'--output-format','json'],True)
    record=json.loads(result);submission=record['id']
    (work/(tag+'-submission.json')).write_text(json.dumps(record,indent=2))
    print('Submission ID:',submission)
    print('Waiting for Apple. If interrupted, use this ID with notarytool info; do not resubmit blindly.')
    # Wait may return nonzero for rejection. Always retrieve status and the diagnostic log.
    subprocess.run(['/usr/bin/xcrun','notarytool','wait',submission,'--keychain-profile',profile],stdin=tty)
    status=json.loads(run(['/usr/bin/xcrun','notarytool','info',submission,'--keychain-profile',profile,'--output-format','json'],True))
    (work/(tag+'-status.json')).write_text(json.dumps(status,indent=2))
    run(['/usr/bin/xcrun','notarytool','log',submission,'--keychain-profile',profile,work/(tag+'-notary-log.json')])
    if status.get('status')!='Accepted':
        raise RuntimeError(f'Apple returned {status.get("status")}. Read {work/(tag+"-notary-log.json")}. Nothing is labelled ready for distribution.')
    print('Accepted. Attaching and checking the notarization ticket…')
    run(['/usr/bin/xcrun','stapler','staple',app]);run(['/usr/bin/xcrun','stapler','validate',app])

try:
    print('\noollim — Developer ID signing and notarization\n')
    print('You need Xcode command-line tools, an active Apple Developer membership,')
    print('and a Developer ID Application certificate WITH its private key in your Keychain.')
    print('This assistant signs a COPY. Your original installer is preserved.\n')
    raw=a.app or ask('Drag Install oollim.app here, then press Return: ')
    # Finder drag-and-drop into a shell escapes spaces. Quoted paths also work.
    import shlex
    if not Path(raw).expanduser().exists():
        parts=shlex.split(raw)
        if len(parts)==1: raw=parts[0]
    source=Path(raw).expanduser().resolve()
    resources=source/'Contents/Resources';stage=resources/'stage';stub=resources/'oollim.app'
    if not source.is_dir() or source.suffix!='.app' or not (stage/'payload.json').is_file() or not stub.is_dir():
        raise RuntimeError('Select the self-contained Install oollim.app, not the small installed oollim launcher.')
    for tool in ('notarytool','stapler'):run(['/usr/bin/xcrun','--find',tool],True)
    run(['/usr/bin/xcrun','--find','codesign'],True)
    print('Installer structure and Apple signing tools found.')
    if a.check_only:
        print('Check-only complete: no files signed, credentials accessed, or uploads performed.');sys.exit(0)
    identities=run(['/usr/bin/security','find-identity','-v','-p','codesigning'],True)
    choices=re.findall(r'([0-9A-Fa-f]{40}) "(Developer ID Application:[^"]+)"',identities)
    if not choices:raise RuntimeError('No usable Developer ID Application identity found. Install the certificate and matching private key through Xcode/Keychain, then rerun.')
    for i,(_,name) in enumerate(choices,1):print(f'  {i}. {name}')
    selected=ask('Choose the identity number: ')
    if not selected.isdigit() or not 1<=int(selected)<=len(choices):raise RuntimeError('Invalid identity selection.')
    identity,name=choices[int(selected)-1]
    profile=ask('Notarization Keychain profile name [oollim-notary]: ') or 'oollim-notary'
    if ask('Create/update this Keychain profile now? [y/N]: ').lower()=='y':
        print('Apple’s notarytool will prompt for credentials and save them in Keychain.')
        print('For Apple ID authentication, use an app-specific password—not your normal login password.')
        apple_id=ask('Developer Apple ID email: ')
        match=re.search(r'\(([A-Z0-9]{10})\)$',name)
        default_team=match.group(1) if match else ''
        team=ask('Developer Team ID'+(' ['+default_team+']' if default_team else '')+': ') or default_team
        if not apple_id or not re.fullmatch(r'[A-Z0-9]{10}',team):raise RuntimeError('Provide an Apple ID email and the 10-character Developer Team ID.')
        print('Create an app-specific password at https://account.apple.com under Sign-In and Security.')
        print('Enter that password at Apple’s next secure prompt; this script does not read or save it.')
        run(['/usr/bin/xcrun','notarytool','store-credentials',profile,'--apple-id',apple_id,'--team-id',team])
    print('Checking notarization authentication…')
    run(['/usr/bin/xcrun','notarytool','history','--keychain-profile',profile,'--output-format','json'],True)
    print('\nThe embedded Python runtime will permit third-party extension libraries and JIT code.')
    print('The installer and launcher retain normal hardened-runtime settings.')
    print('Both the installed launcher and the enclosing installer will be notarized.')
    if ask('Proceed with signing and uploading these apps to Apple? [y/N]: ').lower()!='y':
        print('Cancelled. No files signed.');sys.exit(0)
    work=Path(tempfile.mkdtemp(prefix='oollim-signed-',dir=source.parent));app=work/source.name
    print('\nWorking folder:',work)
    run(['/usr/bin/ditto',source,app]);run(['/usr/bin/xattr','-cr',app])
    resources=app/'Contents/Resources';stage=resources/'stage';stub=resources/'oollim.app'
    entitlements=work/'python-entitlements.plist'
    with entitlements.open('wb') as f:plistlib.dump({'com.apple.security.cs.disable-library-validation':True,'com.apple.security.cs.allow-jit':True,'com.apple.security.cs.allow-unsigned-executable-memory':True},f)
    magic={b'\xfe\xed\xfa\xce',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xcf\xfa\xed\xfe',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca',b'\xca\xfe\xba\xbf',b'\xbf\xba\xfe\xca'}
    binaries=[]
    for p in app.rglob('*'):
        if p.is_file() and not p.is_symlink():
            with p.open('rb') as f: head=f.read(4)
            if head in magic:binaries.append(p)
    for i,p in enumerate(sorted(binaries,key=lambda p:len(p.parts),reverse=True),1):
        print(f'Signing binary {i}/{len(binaries)}: {p.relative_to(app)}',flush=True)
        cmd=['/usr/bin/codesign','--force','--timestamp','--options','runtime','--sign',identity]
        if p.parent==stage/'python/bin' and p.name.startswith('python'):cmd+=['--entitlements',entitlements]
        run(cmd+[p])
    nested=sorted([p for p in app.rglob('*.app') if p.is_dir() and not p.is_symlink()],key=lambda p:len(p.parts),reverse=True)
    for bundle in nested:
        run(['/usr/bin/codesign','--force','--timestamp','--options','runtime','--sign',identity,bundle])
    notarize(stub,'launcher')
    print('\nRefreshing installer integrity hashes after runtime signing…')
    manifest(stage)
    print('Signing the enclosing installer…')
    run(['/usr/bin/codesign','--force','--timestamp','--options','runtime','--sign',identity,app])
    run(['/usr/bin/codesign','--verify','--deep','--strict','--verbose=2',app])
    notarize(app,'installer')
    for bundle in (stub,app):
        run(['/usr/bin/codesign','--verify','--deep','--strict',bundle])
        run(['/usr/sbin/spctl','--assess','--type','execute','--verbose=2',bundle])
    archive=work/'oollim-mac-arm64-signed.zip'
    run(['/usr/bin/ditto','-c','-k','--keepParent',app,archive])
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (work/'SHA256SUMS.txt').write_text(digest+'  '+archive.name+'\n')
    print('\nSUCCESS: signatures, notarization tickets, and Gatekeeper checks passed.')
    print('Distribute:',archive)
    print('SHA-256:',digest)
    print('Before publishing, test this signed build on a clean Mac with a short audio file.')
    print('Signing changes the Python runtime; the earlier unsigned audio test does not replace that check.')
except (Exception,KeyboardInterrupt) as exc:
    print('\nSTOPPED:',str(exc) or 'Interrupted by user',file=sys.stderr)
    print('Your original installer is unchanged. Keep the working folder and any submission IDs for diagnosis.',file=sys.stderr)
    sys.exit(1)
PY
