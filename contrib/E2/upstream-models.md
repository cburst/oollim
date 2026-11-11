# E2 audit, part 1: upstream encoders

Audit date: 2026-11-10. Scope: a documentary audit of the three pretrained encoders pinned by oollim in configs/extractor.json and speech/resources-manifest.json. Nothing was retrained. Each statement says where it was checked. Statements marked [inference] are not stated by the model authors.

## Summary

None of the three encoders declares SpeechOcean762, L2-ARCTIC, the Speech Accent Archive (SAA), J-AESOP or AIHub 71463 as training data. The configuration of the released PhoneticXeus run lists SpeechOcean762, L2-ARCTIC and SAA among its development sets, so these corpora may have informed checkpoint selection. XEUS and XLS-R do not mention any of the five corpora. The PhoneticXeus model card license changed from Apache-2.0 to CC BY-NC-SA 4.0 on 2026-09-30, and docs/RIGHTS.md already records the current license.

## Pinned revisions and licenses

| Encoder | Hugging Face repository | Pinned revision | Current HEAD on 2026-11-10 | Weights file | License on the model card |
|---|---|---|---|---|---|
| XLS-R 300M | facebook/wav2vec2-xls-r-300m | 1a640f32ac3e39899438a2931f9924c02f080a54 | Yes, last commit 2022-08-10 | pytorch_model.bin | Apache-2.0, unchanged in the history |
| XEUS | espnet/xeus | b651261280bf792df4632ccb2d1ba4eeaabeaee2 | Yes, 2025-06-17 | model/xeus_checkpoint_new.pth | CC BY-NC-SA 4.0 since at least 2024-08-03. The card said MIT through 2024-07-10, the last commit on which it appears |
| PhoneticXeus | changelinglab/PhoneticXeus | ed64f14aeb0e903a80a825cf67a079aa8f93d3ac | Yes, 2026-09-30 | model.safetensors | CC BY-NC-SA 4.0 since commit ed64f14 ("update license", 2026-09-30). Apache-2.0 up to commit 3a8d860 (2026-09-20) |

Full SHA-256 values of the downloaded weights, checked against the Hugging Face LFS records, are in checksums.txt.

Notes:
- The XEUS repository holds two checkpoints. On 2025-06-17, xeus_checkpoint_new.pth was renamed from 38epoch.pth and the previous file became xeus_checkpoint_old.pth. meta.yaml points to the new file, which is the one oollim uses. The model card does not describe the difference between them.
- The PhoneticXeus source code pinned by oollim is commit 1be0aec of the upstream source repository. Its HEAD on 2026-11-10 is 388f4c5 (2026-10-17).

## Declared training and development data

**XLS-R 300M.** Babu et al., "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale", Interspeech 2022 (arXiv 2111.09296). Pretraining uses about 436K hours: VoxPopuli (372K), MLS (about 50K, of which 44K English), CommonVoice v6.1 (7K), VoxLingua107 (6.6K) and BABEL (about 1K). Audio is cropped to 20 s. The paper does not mention any of the five corpora.

**XEUS.** Chen et al., "Towards Robust Speech Representation Learning for Thousands of Languages", EMNLP 2024, pp. 10205-10224, DOI 10.18653/v1/2024.emnlp-main.570. Appendix Table 11 lists 37 public corpora plus MMS-unlab v2, WikiTongues and Jesus Dramas, about 1.081 million hours in total. Utterances longer than 40 s are filtered out. Accented English sources include the Edinburgh International Accents of English corpus (40 h), AccentDB (20 h), NPTEL (15K h), Common Voice 13 and YODAS. KSponSpeech (970 h of native Korean speech) is included. It is a different AIHub corpus from 71463. None of the five corpora appears.

**PhoneticXeus.** Bharadwaj et al., "An Empirical Recipe for Universal Phone Recognition", arXiv 2603.29042 v2 (2026-07-13), stated on arXiv as accepted at Interspeech 2026. The model is fine-tuned from XEUS with self-conditioned CTC on IPAPack++ (about 17K hours with G2P labels), introduced by Zhu et al., "ZIPA: A family of efficient models for multilingual phone recognition", ACL 2025 (arXiv 2505.23170). IPAPack++ draws on IPA Pack, Common Voice 16.0, LibriSpeech, MLS, Aishell-1, OpenSLR sets for Javanese, Sinhala and Bengali, IISc-MILE Tamil and Kazakh corpora. In ZIPA, L2-ARCTIC and Buckeye are test sets only. Table 1 of the PhoneticXeus paper marks in gray the datasets included in training, and the L2-ARCTIC (PR-arc) and SAA (PR-saa) cells for PhoneticXeus are not gray.

Development sets of the released checkpoint. The file config_tree.log published at revision ed64f14 records train_split "train" and dev_splits dev_1k, dev_gmuaccent, dev_buckeye, dev_epadb, dev_speechoceanotth and dev_l2arctic. In configs/data/ipapack_index.yaml, present at the pinned commit 1be0aec, dev_gmuaccent reads transcripts from test_gmuaccent (the SAA), dev_speechoceanotth from test_speechoceannotth and dev_l2arctic from test_l2arctic_perceived. The utterance lists of these development sets are not public.

Checkpoint selection. The configuration sets max_steps to 40,000, saves a checkpoint every 2,000 steps and sets no monitor metric. An earlier version of the model card (2026-03-30) published checkpoint-22000.ckpt. The paper states that the model was trained until convergence and that the best checkpoint was evaluated, without stating the selection criterion. [inference] The development sets above may have informed the choice of checkpoint.

## Corpus by encoder

| Corpus | XLS-R 300M | XEUS | PhoneticXeus |
|---|---|---|---|
| SpeechOcean762 | Not declared | Not declared | Not in training. Development set dev_speechoceanotth |
| L2-ARCTIC | Not declared | Not declared | Not in training. Development set dev_l2arctic |
| Speech Accent Archive | Not declared | Not declared | Not in training. Development set dev_gmuaccent |
| J-AESOP | Not declared | Not declared | Not declared |
| AIHub 71463 | Not declared | Not declared. KSponSpeech, another AIHub corpus, is in training | Not declared |

## Limits of this audit

- YODAS (XEUS) and VoxLingua107 (XLS-R) come from YouTube without a public list of videos, so readings of the Stella passage posted there cannot be ruled out.
- Common Voice includes English speakers whose first languages include Mandarin, Korean and Japanese. This is generic exposure to L2 English, not exposure to the five corpora.

## Relation to the oollim partitions

oollim fits and selects its ridge heads on the official SpeechOcean762 training partition. The official training and test partitions have 125 speakers each (Gong et al., "Transformer-Based Multi-Aspect Multi-Granularity Non-Native English Speaker Pronunciation Assessment", ICASSP 2022, arXiv 2205.03432). [inference] If dev_speechoceanotth comes from the official test partition, as its name suggests, it shares no utterances with the partitions oollim uses. docs/RIGHTS.md asks that later evidence be recorded explicitly rather than silently changing the stated limitation. This file is that record.

## Open questions

1. Which XEUS checkpoint, old or new, the PhoneticXeus authors fine-tuned from.
2. Whether speech/resources-manifest.json should record the date of the PhoneticXeus license change (2026-09-30).

## Suggested text for the paper, for the author to rewrite

[SUGGESTION] PhoneticXeus was fine-tuned on IPAPack++, which its authors report does not contain the learner corpora used here. Its released training configuration lists SpeechOcean762, L2-ARCTIC and the Speech Accent Archive among its development sets, and the Speech Accent Archive uses the Stella passage. These sets may have informed checkpoint selection. That they did not supply training gradients is inferred from the released configuration and from the shading of Table 1, not stated by the authors.

## Sources

- XLS-R: https://arxiv.org/abs/2111.09296 and https://huggingface.co/facebook/wav2vec2-xls-r-300m
- XEUS: https://aclanthology.org/2024.emnlp-main.570/ and https://huggingface.co/espnet/xeus
- PhoneticXeus: https://arxiv.org/abs/2603.29042 and https://huggingface.co/changelinglab/PhoneticXeus (config_tree.log and commit history at revision ed64f14)
- ZIPA and IPAPack++: https://arxiv.org/abs/2505.23170
- GOPT and the SpeechOcean762 partitions: https://arxiv.org/abs/2205.03432
