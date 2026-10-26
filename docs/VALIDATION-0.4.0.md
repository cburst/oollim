# oollim 0.4.0 validation

- 56 automated tests passed, including all 35 bundled family/selection choices.
- macOS: short public audio inference and GUI-to-PDF evaluation passed using the existing local runtime. A fresh dependency installation was not repeated for this release.
- Windows Server 2022 EC2: installation, short public audio inference, resume checks, and automated offscreen Qt GUI-to-PDF evaluation passed under a standard user account. All 35 menu choices were checked. Interactive desktop operation and Windows 11 were not retested in this run.
- The tested Windows executable was downloaded and its SHA-256 verified before cloud cleanup: `c33f7741b7ba1e1d6e55316a084c1020f212b93d1893abe254aab35385723806`.
- The Mac installer is unsigned; the distributor signs and notarizes the self-contained `Install oollim.app`.
- Ridge weights remain bundled with the application; no remote model catalog is required.
