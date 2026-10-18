# Code, model, and data rights

This draft intentionally bundles no corpora, audio, individual private ratings, or pretrained weights. It includes the seven ridge families with attribution. Public availability is not the same as unrestricted redistribution.

| Resource | Source / terms | Release treatment |
|---|---|---|
| SpeechOcean762 | [OpenSLR 101](https://www.openslr.org/101/), CC BY 4.0 | Obtain corpus separately; cite dataset. The derived default scoring heads are included with attribution in models/speechocean/NOTICE.md. |
| Korean learner corpus | [AIHub 71463](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71463), [usage policy](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105), [FAQ](https://www.aihub.or.kr/aihubnews/faq/list.do) | No corpus redistribution. Check access/export conditions and attribution; confirm any ambiguous derived-weight release conditions. |
| J-AESOP | [Corpus paper](https://doi.org/10.1250/ast.e25.06), [rating metadata](https://zenodo.org/records/10907820), supplied individual agreement | Research fitting/evaluation and aggregate publication allowed; no corpus redistribution or commercial use. The supplied agreement requires case-by-case permission for derived weights. At the project author’s direction, AESOP-containing heads are included in this local release candidate while that permission remains unresolved. Inclusion is not evidence of permission. |
| XEUS | [Model card](https://huggingface.co/espnet/xeus), currently CC BY-NC-SA 4.0 | Acquire pinned revision separately; respect applicable noncommercial/share-alike terms. |
| PhoneticXeus | [Model card](https://huggingface.co/changelinglab/PhoneticXeus), currently CC BY-NC-SA 4.0 | Same separation; model card explicitly notes a license update. |
| XLS-R 300M | [Model card](https://huggingface.co/facebook/wav2vec2-xls-r-300m), Apache 2.0 recorded in acquisition manifest | Acquire separately with upstream notices. |
| CMUdict | [Upstream dictionary](https://github.com/cmusphinx/cmudict) | Preserve upstream copyright/license when obtaining or redistributing. |
| ESPnet / PhoneticXeus source | Pinned upstream repositories in resource manifest | Not vendored into this draft; retain upstream terms when installing. |

The user-supplied J-AESOP agreement is a project permission summary, not a blanket public license. An article's open-access license is not the audio corpus's license. Likewise, an open code license would not override pretrained-model or dataset restrictions. “Free local inference” does not mean commercial use is automatically permitted.

Original code and documentation use the owner-supplied Custom Source-Available License (Non-Production Use Only), copyright 2026 R ROSE; see `../LICENSE.md`. The text is included unchanged. A separate production license from the project author does not grant or override third-party model/data rights. Describe this release as source-available, not permissively open-source.

Model-exposure audit: the release does not assert that PhoneticXeus never saw SpeechOcean762 or L2-ARCTIC in upstream training. Our own speaker separation cannot establish that. Record subsequent evidence explicitly rather than silently changing this limitation.
