# Release 0.4.0

The release adds three-ratio banded ridge to the unchanged S/D/H collections: 35 selectable family/criterion combinations. Dropdown names describe construct-wise correlation (Spearman) or pronunciation-accuracy error (accuracy MAE). Existing CLI keys and the matched SpeechOcean default remain compatible.

The manuscript reports descriptive leaders across all 35 configurations and paired uncertainty across 510 contrasts. All coefficient fitting and hyperparameter selection use public training/development data. Private recordings, identifiers, individual ratings and system predictions are excluded from the release.

The repository is https://github.com/cburst/oollim. This package is prepared locally; it does not automatically publish a GitHub release or update an existing DOI. The author signs/notarizes the Mac installer before public distribution. The Windows executable is unsigned. See VALIDATION.md for the scope of platform checks; Windows Server testing does not establish Windows 11 compatibility.

Derived-weight permissions were confirmed by the author. Included licenses and third-party notices remain applicable. No source corpus is bundled. The original deployed SpeechOcean model remains separately versioned from the matched refits.
