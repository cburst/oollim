# Native installer builds

Status: unsigned prototype installers have been compiled. Apple Silicon macOS
has passed native GUI scoring and PDF export. On 19 October 2026, a fresh
Windows Server 2022 installation passed native Browse dialogs, short-audio
scoring, four-page PDF export, and Open PDF in an active console session.
The installer ran silently under a standard user; the application checks used
the visible desktop. Windows 11 and signed/notarized distribution are not yet verified.

## Design

- Windows 11 AMD64: NSIS setup.exe, per-user install and desktop/Start-menu shortcuts.
- Apple Silicon: a lightweight Platypus .app stub opens the launcher; first-run setup uses Terminal, then opens the Qt GUI.
- Bundle a redistributable CPython 3.11 install-only runtime for the target platform,
  retaining the runtime's license notices. No system Python requirement. WeasyPrint PDF export on macOS requires Homebrew Pango.
- First launch copies the application and runtime into ~/oollim/versions/,
  creates .venv there, then uses install.py to install dependencies and acquire models.
- Internet required at first setup; this is not an offline 10 GB installer.
- Mac app bundles remain unmodified. All application data is under `~/oollim` on both platforms.
- Default outputs go to ~/oollim/results. Windows uninstall removes the launcher
  but preserves runtime/model data and results. Mac removal similarly preserves them.
- A setup lock prevents concurrent provisioning. After forced termination, the error
  describes how to remove the lock only once no setup process remains.
- Changing the release creates another versioned environment. Existing versions
  are retained; allow additional disk space during upgrades.

## Build prerequisites

Install NSIS (`makensis`) or Platypus and its command-line tool on the build machine.
The PDF setup step installs Pango with an existing Homebrew installation. If Homebrew is missing, setup explains how to install it and resume. Windows uses the checksum-pinned official WeasyPrint onedir runtime under `~/oollim/pdf-runtime`, including its native libraries and license notices.
Obtain an approved, checksum-verified relocatable CPython 3.11 runtime, such as an
Astral python-build-standalone install-only distribution. Record its exact release,
source URL and SHA-256 in the release record before distributing binaries. Do not
copy an ordinary system Python or an existing virtual environment.

The build currently requires that runtime as an explicit input; runtime acquisition
and publisher signing credentials are not configured.

```sh
python3 packaging/build.py --target mac --python-runtime /path/to/python
```

```powershell
py -3.11 packaging/build.py --target windows --python-runtime C:\path\python
```

`--stage-only` prepares the checked payload without calling a compiler. Builds go
under work/native; the staging directory must be fresh. Source files come from
the checked packaging/source-files.json manifest, excluding papers, test data and research result exports.

## Release acceptance still required

1. Compile each installer; inspect bundled files and runtime license notices.
2. Test native architecture and Python version, install without system Python,
   dependencies, model acquisition, and short-audio inference on both target OSes.
3. Test spaces/non-ASCII paths, interrupted setup/resume, duplicate setup, upgrades,
   and uninstall with results retained. Test Windows Visual C++ runtime availability.
4. Sign/notarize the Mac app and sign Windows installer with publisher credentials;
   test downloaded/quarantined artifacts. Unsigned local prototypes are not equivalent.
5. Publish installer checksums and the exact bundled runtime version/checksum.

References: [NSIS](https://nsis.sourceforge.io/Docs/),
[Platypus](https://github.com/sveinbjornt/Platypus), and
[standalone Python](https://github.com/astral-sh/python-build-standalone).

## Mac stub installation

Distribute the entire Mac build folder containing `Install oollim.command`, `stage/`, and `oollim.app`. Double-click `Install oollim.command`: it copies the payload to `~/oollim/launcher` and the small Platypus stub to `~/Applications`. No packed Python executable is created. The stub opens the home-folder launcher, which provisions dependencies before launching the GUI. Windows NSIS similarly extracts Python source/runtime and installs shortcuts rather than freezing Python into an executable.

Both icons use the same microphone artwork on a light-blue rounded rectangle; attribution is in `assets/NOTICE.md`. Regenerate with `packaging/make_icons.py` (PySide6 and Pillow).

## Self-contained Platypus installer and signing

The Mac build also produces **Install oollim.app**, containing the runtime payload
and lightweight installed launcher. Distribute this app as a ZIP; users no longer
need a separate `.command` file or adjacent `stage` folder. Both apps use the same
microphone icon. First-run dependency/model setup still opens in Terminal.

Run `packaging/sign-mac.command` interactively on the developer's Mac. Select
the installer and Developer ID Application identity; the assistant uses an Apple
Keychain profile for notarization credentials. It signs a fresh copy, signs the
embedded Python binaries before refreshing the bootstrap integrity manifest,
notarizes/staples the installed launcher separately, then signs/notarizes/staples
the enclosing installer and creates a final ZIP with a SHA-256 checksum.
`--check-only /path/to/Install\ oollim.app` validates prerequisites without signing.

Developer signing/notarization requires the owner's certificate and has not been
performed in this workspace. Test a short-audio run with the signed build before
publishing: hardened runtime changes must be verified independently.
