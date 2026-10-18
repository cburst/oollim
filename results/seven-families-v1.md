# Matched seven-family development results

Fixed representations; included-corpus development alpha selection. These are not final tests. SpeechOcean here is a fresh refit, not the original deployed checkpoint.

## Accuracy

| Family | SpeechOcean (500) | Korean (636) | J-AESOP (36) |
|---|---:|---:|---:|
| speechocean | 0.8163 | 0.6993 | 0.9671 |
| korean | 0.7352 | 0.7591 | 0.9694 |
| aesop | 0.6592 | 0.6733 | 0.9871 |
| koreanocean | 0.8079 | 0.7554 | 0.9640 |
| aesopocean | 0.8068 | 0.7154 | 0.9704 |
| koreanaesop | 0.6829 | 0.7552 | 0.9763 |
| aesopkoreanocean | 0.8036 | 0.7483 | 0.9709 |

## Fluency

| Family | SpeechOcean (500) | Korean (636) | J-AESOP (36) |
|---|---:|---:|---:|
| speechocean | 0.8865 | 0.6683 | 0.8564 |
| korean | 0.8370 | 0.7776 | 0.9186 |
| aesop | 0.6820 | 0.5864 | 0.9665 |
| koreanocean | 0.8761 | 0.7696 | 0.9244 |
| aesopocean | 0.8811 | 0.7279 | 0.9518 |
| koreanaesop | 0.8340 | 0.7789 | 0.9591 |
| aesopkoreanocean | 0.8749 | 0.7695 | 0.9535 |

## Prosody

| Family | SpeechOcean (500) | Korean (636) | J-AESOP (36) |
|---|---:|---:|---:|
| speechocean | 0.8901 | 0.6529 | 0.9287 |
| korean | 0.8292 | 0.7776 | 0.9381 |
| aesop | 0.6851 | 0.5437 | 0.9746 |
| koreanocean | 0.8761 | 0.7714 | 0.9494 |
| aesopocean | 0.8781 | 0.7466 | 0.9712 |
| koreanaesop | 0.8298 | 0.7772 | 0.9727 |
| aesopkoreanocean | 0.8723 | 0.7699 | 0.9740 |

Values: Spearman correlation. Full MAE/RMSE/Pearson values are in the adjacent JSON.
