# Code, model, and data rights

This release intentionally bundles no corpora, audio, individual private ratings, or pretrained weights. It includes the seven ridge families with attribution. Public availability is not the same as unrestricted redistribution.

| Resource | Source / terms | Release treatment |
|---|---|---|
| SpeechOcean762 | [OpenSLR 101](https://www.openslr.org/101/), CC BY 4.0 | Obtain corpus separately; cite dataset. The derived default scoring heads are included with attribution in models/speechocean/NOTICE.md. |
| Korean learner corpus | [AIHub 71463](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71463), [usage policy](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105), [FAQ](https://www.aihub.or.kr/aihubnews/faq/list.do) | The project author confirms permission to release the derived scoring weights for this project. Corpus redistribution is excluded; source access conditions and attribution remain applicable. |
| J-AESOP | [Corpus paper](https://doi.org/10.1250/ast.e25.06), [rating metadata](https://zenodo.org/records/10907820), supplied individual agreement | Written project-specific permission covers training, evaluation, aggregate publication, code and trained-weight release, and free availability through interlanguage.ai. Weights must be non-commercial, and the provider requests notification before release. No corpus redistribution or commercial use is permitted. |
| XEUS | [Model card](https://huggingface.co/espnet/xeus), currently CC BY-NC-SA 4.0 | Acquire pinned revision separately; respect applicable noncommercial/share-alike terms. |
| PhoneticXeus | [Model card](https://huggingface.co/changelinglab/PhoneticXeus), currently CC BY-NC-SA 4.0 | Same separation; model card explicitly notes a license update. |
| XLS-R 300M | [Model card](https://huggingface.co/facebook/wav2vec2-xls-r-300m), Apache 2.0 recorded in acquisition manifest | Acquire separately with upstream notices. |
| CMUdict | [Upstream dictionary](https://github.com/cmusphinx/cmudict) | Preserve upstream copyright/license when obtaining or redistributing. |
| ESPnet / PhoneticXeus source | Pinned upstream repositories in resource manifest | Not vendored into this draft; retain upstream terms when installing. |

Release permission for the bundled derived scoring weights is confirmed by the project author as of 19 October 2026. The J-AESOP provider’s supplied written permission is project-specific; it does not license redistribution of the corpus. Private permission correspondence is not included. See [weight release terms](../models/WEIGHTS-LICENSE.md). An article's open-access license is not the audio corpus's license. Likewise, an open code license would not override pretrained-model or dataset restrictions. “Free local inference” does not mean commercial use is automatically permitted.

Original code and documentation use the owner-supplied Custom Source-Available License (Non-Production Use Only), copyright 2026 R ROSE; see `../LICENSE.md`. The text is included unchanged. A separate production license from the project author does not grant or override third-party model/data rights. Describe this release as source-available, not permissively open-source.

Model-exposure audit: the release does not assert that PhoneticXeus never saw SpeechOcean762 or L2-ARCTIC in upstream training. Our own speaker separation cannot establish that. Record subsequent evidence explicitly rather than silently changing this limitation.
