# Versioned selection-comparison heads

Exact evaluated heads for three selection conditions and seven corpus families. `registry.json` identifies all 63 construct heads, their checksums, and the API evaluation versions. Each NPZ stores the multi-output fit; use the named construct's output column for that construct. Accuracy uses XEUS + PhoneticXeus + IPA; fluency/prosody use XLS-R + XEUS.

These are additional research bundles. They do not replace the application defaults or `configs/selected-constructs.json`. `spearman/speechocean` is the matched refit used in the revised paper, whereas the original default `models/speechocean` remains historical.

All weights remain subject to [the repository's weight license](../WEIGHTS-LICENSE.md) and underlying model/data restrictions. No private evaluation data were used to fit these heads. See `results/selection-comparison-v1/README.md` for selection rules and results.

## Desktop/CLI bundles

Each condition/family directory includes a checksum-verified `family.json` for local inference. Version 0.3 exposes all 21 choices through `--family` and `--selection`; GUI selectors use the same choices. These manifests wrap the exact registered coefficients without refitting or modifying the original head metadata. Machine-specific encoder directory paths are ignored for compatibility, while semantic provenance and all model hashes remain checked.
