# X4 results: the official SpeechOcean762 test partition

Written by `contrib/X4/evaluate.py`. Every figure and every interval is in
`metrics.json`. The predictions are in `predictions.npz`, so these tables can
be rebuilt without extracting again.

Resamples 2000, seed 42. Coverage: T-all 2500, T-cov 2483.

Intervals are 95 per cent speaker cluster bootstrap intervals. The column
"exposed" marks a system whose blocks include `phoneticxeus` or `ipa`. The
released PhoneticXeus configuration lists a development set whose transcripts
come from a split named after this test partition, and `phone_evidence`
derives the `ipa` block from the same checkpoint. Those rows are potentially
optimistic.

## S1. The released heads, exactly as distributed

### T-all

| rule | construct | blocks | penalty | Pearson [95% CI] | Spearman [95% CI] | MAE [95% CI] | Pearson clipped | exposed |
|---|---|---|---|---|---|---|---|---|
| spearman | accuracy | xeus+phoneticxeus+ipa | 10 | 0.7165 [0.6541, 0.7625] | 0.7047 [0.6432, 0.7528] | 0.784 [0.723, 0.854] | 0.7165 | yes |
| spearman | fluency | xeus+xlsr | 1 | 0.7882 [0.7221, 0.8397] | 0.7882 [0.7388, 0.8271] | 0.621 [0.573, 0.681] | 0.7883 | no |
| spearman | prosody | xeus+xlsr | 1 | 0.7887 [0.7317, 0.8348] | 0.7859 [0.7305, 0.8271] | 0.632 [0.581, 0.695] | 0.7887 | no |
| damae | accuracy | xeus+phoneticxeus+ipa | 10.0 | 0.7165 [0.6541, 0.7625] | 0.7047 [0.6432, 0.7528] | 0.784 [0.723, 0.854] | 0.7165 | yes |
| damae | fluency | xeus+xlsr | 10.0 | 0.7870 [0.7183, 0.8404] | 0.7953 [0.7487, 0.8317] | 0.620 [0.567, 0.686] | 0.7870 | no |
| damae | prosody | xeus+xlsr | 10.0 | 0.7867 [0.7239, 0.8370] | 0.7923 [0.7406, 0.8309] | 0.634 [0.577, 0.703] | 0.7867 | no |
| historical_grid_damae | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7227 [0.6600, 0.7676] | 0.7066 [0.6459, 0.7544] | 0.781 [0.722, 0.847] | 0.7227 | yes |
| historical_grid_damae | fluency | xeus+xlsr | 5.0 | 0.7921 [0.7241, 0.8441] | 0.7992 [0.7525, 0.8360] | 0.611 [0.561, 0.675] | 0.7921 | no |
| historical_grid_damae | prosody | xeus+xlsr | 5.0 | 0.7914 [0.7316, 0.8395] | 0.7947 [0.7418, 0.8340] | 0.625 [0.571, 0.692] | 0.7914 | no |
| banded_3ratio_spearman | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7227 [0.6600, 0.7676] | 0.7066 [0.6459, 0.7544] | 0.781 [0.722, 0.847] | 0.7227 | yes |
| banded_3ratio_spearman | fluency | xeus+xlsr | 2.3713737056616555 | 0.7933 [0.7268, 0.8450] | 0.7979 [0.7509, 0.8346] | 0.610 [0.560, 0.671] | 0.7933 | no |
| banded_3ratio_spearman | prosody | xeus+xlsr | 2.3713737056616555 | 0.7931 [0.7351, 0.8402] | 0.7937 [0.7395, 0.8333] | 0.623 [0.571, 0.688] | 0.7931 | no |
| banded_3ratio_accuracy_mae | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7227 [0.6600, 0.7676] | 0.7066 [0.6459, 0.7544] | 0.781 [0.722, 0.847] | 0.7227 | yes |
| banded_3ratio_accuracy_mae | fluency | xeus+xlsr | 3.1622776601683795 | 0.7934 [0.7265, 0.8452] | 0.7992 [0.7526, 0.8359] | 0.610 [0.559, 0.672] | 0.7934 | no |
| banded_3ratio_accuracy_mae | prosody | xeus+xlsr | 3.1622776601683795 | 0.7929 [0.7344, 0.8406] | 0.7945 [0.7408, 0.8340] | 0.623 [0.571, 0.688] | 0.7929 | no |

### T-cov

| rule | construct | blocks | penalty | Pearson [95% CI] | Spearman [95% CI] | MAE [95% CI] | Pearson clipped | exposed |
|---|---|---|---|---|---|---|---|---|
| spearman | accuracy | xeus+phoneticxeus+ipa | 10 | 0.7176 [0.6551, 0.7636] | 0.7049 [0.6431, 0.7533] | 0.784 [0.723, 0.853] | 0.7176 | yes |
| spearman | fluency | xeus+xlsr | 1 | 0.7890 [0.7229, 0.8400] | 0.7884 [0.7393, 0.8273] | 0.620 [0.573, 0.678] | 0.7891 | no |
| spearman | prosody | xeus+xlsr | 1 | 0.7899 [0.7324, 0.8358] | 0.7867 [0.7308, 0.8280] | 0.630 [0.581, 0.692] | 0.7899 | no |
| damae | accuracy | xeus+phoneticxeus+ipa | 10.0 | 0.7176 [0.6551, 0.7636] | 0.7049 [0.6431, 0.7533] | 0.784 [0.723, 0.853] | 0.7176 | yes |
| damae | fluency | xeus+xlsr | 10.0 | 0.7881 [0.7204, 0.8407] | 0.7957 [0.7496, 0.8319] | 0.619 [0.567, 0.683] | 0.7881 | no |
| damae | prosody | xeus+xlsr | 10.0 | 0.7886 [0.7265, 0.8381] | 0.7933 [0.7415, 0.8320] | 0.632 [0.576, 0.698] | 0.7886 | no |
| historical_grid_damae | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7236 [0.6607, 0.7689] | 0.7066 [0.6455, 0.7546] | 0.780 [0.722, 0.846] | 0.7236 | yes |
| historical_grid_damae | fluency | xeus+xlsr | 5.0 | 0.7931 [0.7258, 0.8444] | 0.7995 [0.7533, 0.8359] | 0.611 [0.560, 0.672] | 0.7931 | no |
| historical_grid_damae | prosody | xeus+xlsr | 5.0 | 0.7930 [0.7331, 0.8405] | 0.7955 [0.7426, 0.8347] | 0.623 [0.570, 0.687] | 0.7930 | no |
| banded_3ratio_spearman | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7236 [0.6607, 0.7689] | 0.7066 [0.6455, 0.7546] | 0.780 [0.722, 0.846] | 0.7236 | yes |
| banded_3ratio_spearman | fluency | xeus+xlsr | 2.3713737056616555 | 0.7942 [0.7278, 0.8453] | 0.7982 [0.7508, 0.8352] | 0.609 [0.560, 0.669] | 0.7942 | no |
| banded_3ratio_spearman | prosody | xeus+xlsr | 2.3713737056616555 | 0.7944 [0.7364, 0.8409] | 0.7944 [0.7403, 0.8344] | 0.621 [0.571, 0.684] | 0.7944 | no |
| banded_3ratio_accuracy_mae | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7236 [0.6607, 0.7689] | 0.7066 [0.6455, 0.7546] | 0.780 [0.722, 0.846] | 0.7236 | yes |
| banded_3ratio_accuracy_mae | fluency | xeus+xlsr | 3.1622776601683795 | 0.7943 [0.7274, 0.8455] | 0.7995 [0.7529, 0.8361] | 0.609 [0.559, 0.669] | 0.7943 | no |
| banded_3ratio_accuracy_mae | prosody | xeus+xlsr | 3.1622776601683795 | 0.7944 [0.7359, 0.8416] | 0.7953 [0.7414, 0.8348] | 0.621 [0.569, 0.684] | 0.7944 | no |

## S2. The same blocks and penalties refitted on all 2,500 train rows

### T-all

| rule | construct | blocks | penalty | Pearson [95% CI] | Spearman [95% CI] | MAE [95% CI] | Pearson clipped | exposed |
|---|---|---|---|---|---|---|---|---|
| spearman | accuracy | xeus+phoneticxeus+ipa | 10 | 0.7164 [0.6521, 0.7645] | 0.7050 [0.6428, 0.7537] | 0.786 [0.726, 0.854] | 0.7164 | yes |
| spearman | fluency | xeus+xlsr | 1 | 0.7911 [0.7189, 0.8454] | 0.7884 [0.7390, 0.8277] | 0.614 [0.566, 0.674] | 0.7912 | no |
| spearman | prosody | xeus+xlsr | 1 | 0.7873 [0.7239, 0.8365] | 0.7817 [0.7253, 0.8243] | 0.629 [0.578, 0.693] | 0.7873 | no |
| damae | accuracy | xeus+phoneticxeus+ipa | 10.0 | 0.7164 [0.6521, 0.7645] | 0.7050 [0.6428, 0.7537] | 0.786 [0.726, 0.854] | 0.7164 | yes |
| damae | fluency | xeus+xlsr | 10.0 | 0.7880 [0.7158, 0.8422] | 0.7935 [0.7463, 0.8305] | 0.618 [0.567, 0.681] | 0.7880 | no |
| damae | prosody | xeus+xlsr | 10.0 | 0.7863 [0.7192, 0.8380] | 0.7907 [0.7389, 0.8300] | 0.629 [0.575, 0.696] | 0.7863 | no |
| historical_grid_damae | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7233 [0.6596, 0.7705] | 0.7073 [0.6460, 0.7556] | 0.781 [0.723, 0.847] | 0.7233 | yes |
| historical_grid_damae | fluency | xeus+xlsr | 5.0 | 0.7930 [0.7202, 0.8468] | 0.7976 [0.7503, 0.8343] | 0.609 [0.560, 0.671] | 0.7930 | no |
| historical_grid_damae | prosody | xeus+xlsr | 5.0 | 0.7901 [0.7245, 0.8409] | 0.7923 [0.7397, 0.8322] | 0.621 [0.569, 0.687] | 0.7901 | no |
| banded_3ratio_spearman | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7233 [0.6596, 0.7705] | 0.7073 [0.6460, 0.7556] | 0.781 [0.723, 0.847] | 0.7233 | yes |
| banded_3ratio_spearman | fluency | xeus+xlsr | 2.3713737056616555 | 0.7948 [0.7222, 0.8482] | 0.7968 [0.7494, 0.8345] | 0.606 [0.558, 0.667] | 0.7948 | no |
| banded_3ratio_spearman | prosody | xeus+xlsr | 2.3713737056616555 | 0.7912 [0.7265, 0.8405] | 0.7904 [0.7368, 0.8313] | 0.621 [0.569, 0.686] | 0.7912 | no |
| banded_3ratio_accuracy_mae | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7233 [0.6596, 0.7705] | 0.7073 [0.6460, 0.7556] | 0.781 [0.723, 0.847] | 0.7233 | yes |
| banded_3ratio_accuracy_mae | fluency | xeus+xlsr | 3.1622776601683795 | 0.7946 [0.7219, 0.8483] | 0.7978 [0.7503, 0.8349] | 0.606 [0.557, 0.667] | 0.7946 | no |
| banded_3ratio_accuracy_mae | prosody | xeus+xlsr | 3.1622776601683795 | 0.7912 [0.7262, 0.8411] | 0.7916 [0.7382, 0.8320] | 0.620 [0.568, 0.686] | 0.7912 | no |

### T-cov

| rule | construct | blocks | penalty | Pearson [95% CI] | Spearman [95% CI] | MAE [95% CI] | Pearson clipped | exposed |
|---|---|---|---|---|---|---|---|---|
| spearman | accuracy | xeus+phoneticxeus+ipa | 10 | 0.7180 [0.6536, 0.7665] | 0.7054 [0.6430, 0.7541] | 0.785 [0.726, 0.852] | 0.7180 | yes |
| spearman | fluency | xeus+xlsr | 1 | 0.7923 [0.7209, 0.8454] | 0.7888 [0.7388, 0.8283] | 0.613 [0.566, 0.671] | 0.7923 | no |
| spearman | prosody | xeus+xlsr | 1 | 0.7888 [0.7259, 0.8375] | 0.7827 [0.7263, 0.8253] | 0.627 [0.578, 0.689] | 0.7888 | no |
| damae | accuracy | xeus+phoneticxeus+ipa | 10.0 | 0.7180 [0.6536, 0.7665] | 0.7054 [0.6430, 0.7541] | 0.785 [0.726, 0.852] | 0.7180 | yes |
| damae | fluency | xeus+xlsr | 10.0 | 0.7893 [0.7187, 0.8425] | 0.7939 [0.7474, 0.8312] | 0.617 [0.567, 0.678] | 0.7893 | no |
| damae | prosody | xeus+xlsr | 10.0 | 0.7884 [0.7225, 0.8389] | 0.7919 [0.7401, 0.8308] | 0.626 [0.573, 0.691] | 0.7884 | no |
| historical_grid_damae | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7250 [0.6609, 0.7725] | 0.7076 [0.6453, 0.7564] | 0.780 [0.722, 0.845] | 0.7250 | yes |
| historical_grid_damae | fluency | xeus+xlsr | 5.0 | 0.7943 [0.7231, 0.8470] | 0.7980 [0.7512, 0.8348] | 0.608 [0.559, 0.668] | 0.7943 | no |
| historical_grid_damae | prosody | xeus+xlsr | 5.0 | 0.7921 [0.7274, 0.8416] | 0.7933 [0.7413, 0.8329] | 0.619 [0.567, 0.682] | 0.7921 | no |
| banded_3ratio_spearman | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7250 [0.6609, 0.7725] | 0.7076 [0.6453, 0.7564] | 0.780 [0.722, 0.845] | 0.7250 | yes |
| banded_3ratio_spearman | fluency | xeus+xlsr | 2.3713737056616555 | 0.7960 [0.7243, 0.8485] | 0.7972 [0.7501, 0.8348] | 0.605 [0.558, 0.663] | 0.7960 | no |
| banded_3ratio_spearman | prosody | xeus+xlsr | 2.3713737056616555 | 0.7929 [0.7288, 0.8414] | 0.7914 [0.7378, 0.8320] | 0.618 [0.568, 0.681] | 0.7929 | no |
| banded_3ratio_accuracy_mae | accuracy | xeus+phoneticxeus+ipa | 5.0 | 0.7250 [0.6609, 0.7725] | 0.7076 [0.6453, 0.7564] | 0.780 [0.722, 0.845] | 0.7250 | yes |
| banded_3ratio_accuracy_mae | fluency | xeus+xlsr | 3.1622776601683795 | 0.7958 [0.7244, 0.8485] | 0.7982 [0.7509, 0.8355] | 0.605 [0.557, 0.664] | 0.7958 | no |
| banded_3ratio_accuracy_mae | prosody | xeus+xlsr | 3.1622776601683795 | 0.7929 [0.7284, 0.8422] | 0.7926 [0.7399, 0.8326] | 0.618 [0.567, 0.680] | 0.7929 | no |

## S3. One ridge per block combination, penalty chosen on the 500 development rows

Descriptive by construction. The penalty is the only quantity selected,
and it is selected on the development split, never on the test partition.

### T-all

**accuracy**

| blocks | penalty | development Spearman | Pearson [95% CI] | Spearman [95% CI] | MAE [95% CI] | exposed |
|---|---|---|---|---|---|---|
| xeus+phoneticxeus+ipa | 10.0 | 0.8163 | 0.7165 [0.6541, 0.7625] | 0.7047 [0.6432, 0.7528] | 0.784 [0.723, 0.854] | yes |
| xeus+xlsr+phoneticxeus+ipa | 10.0 | 0.8136 | 0.7312 [0.6717, 0.7744] | 0.7133 [0.6516, 0.7618] | 0.772 [0.714, 0.836] | yes |
| xeus+ipa | 10.0 | 0.8124 | 0.7096 [0.6490, 0.7564] | 0.6998 [0.6363, 0.7501] | 0.795 [0.734, 0.863] | yes |
| xeus+xlsr+ipa | 10.0 | 0.8100 | 0.7248 [0.6659, 0.7685] | 0.7084 [0.6451, 0.7590] | 0.783 [0.725, 0.846] | yes |
| xeus | 10.0 | 0.8043 | 0.6835 [0.6175, 0.7370] | 0.6843 [0.6223, 0.7335] | 0.810 [0.745, 0.883] | no |
| xeus+xlsr | 10.0 | 0.8031 | 0.7039 [0.6407, 0.7511] | 0.6974 [0.6334, 0.7476] | 0.794 [0.732, 0.863] | no |
| xeus+phoneticxeus | 10.0 | 0.8004 | 0.6819 [0.6129, 0.7354] | 0.6834 [0.6212, 0.7324] | 0.808 [0.740, 0.884] | yes |
| xeus+xlsr+phoneticxeus | 10.0 | 0.7988 | 0.7005 [0.6358, 0.7487] | 0.6952 [0.6317, 0.7449] | 0.796 [0.734, 0.867] | yes |
| xlsr+phoneticxeus+ipa | 10.0 | 0.7964 | 0.7141 [0.6522, 0.7588] | 0.6932 [0.6281, 0.7461] | 0.794 [0.734, 0.862] | yes |
| phoneticxeus+ipa | 1.0 | 0.7944 | 0.7000 [0.6327, 0.7504] | 0.6780 [0.6143, 0.7286] | 0.809 [0.748, 0.878] | yes |
| xlsr+ipa | 1.0 | 0.7775 | 0.7121 [0.6497, 0.7581] | 0.6756 [0.6024, 0.7345] | 0.807 [0.752, 0.866] | yes |
| xlsr+phoneticxeus | 10.0 | 0.7747 | 0.6732 [0.6041, 0.7261] | 0.6632 [0.5934, 0.7174] | 0.826 [0.760, 0.903] | yes |
| phoneticxeus | 1.0 | 0.7715 | 0.6531 [0.5758, 0.7135] | 0.6441 [0.5764, 0.6987] | 0.847 [0.778, 0.928] | yes |
| xlsr | 1.0 | 0.7684 | 0.6809 [0.6124, 0.7323] | 0.6500 [0.5729, 0.7098] | 0.829 [0.768, 0.895] | no |
| ipa | 1.0 | 0.6808 | 0.5745 [0.5014, 0.6368] | 0.5511 [0.4671, 0.6249] | 0.931 [0.856, 1.013] | yes |

**fluency**

| blocks | penalty | development Spearman | Pearson [95% CI] | Spearman [95% CI] | MAE [95% CI] | exposed |
|---|---|---|---|---|---|---|
| xeus+xlsr | 1.0 | 0.8865 | 0.7882 [0.7221, 0.8397] | 0.7882 [0.7388, 0.8271] | 0.621 [0.573, 0.681] | no |
| xeus+xlsr+ipa | 1.0 | 0.8848 | 0.7969 [0.7397, 0.8425] | 0.7934 [0.7446, 0.8322] | 0.618 [0.572, 0.676] | yes |
| xeus+xlsr+phoneticxeus+ipa | 10.0 | 0.8829 | 0.8007 [0.7410, 0.8486] | 0.8051 [0.7599, 0.8404] | 0.605 [0.554, 0.669] | yes |
| xeus+xlsr+phoneticxeus | 10.0 | 0.8825 | 0.7861 [0.7123, 0.8414] | 0.7978 [0.7515, 0.8339] | 0.617 [0.562, 0.687] | yes |
| xeus+phoneticxeus+ipa | 10.0 | 0.8800 | 0.7892 [0.7194, 0.8433] | 0.7969 [0.7505, 0.8337] | 0.614 [0.560, 0.681] | yes |
| xeus+phoneticxeus | 10.0 | 0.8796 | 0.7733 [0.6861, 0.8357] | 0.7878 [0.7427, 0.8252] | 0.627 [0.568, 0.701] | yes |
| xeus | 10.0 | 0.8791 | 0.7681 [0.6804, 0.8324] | 0.7819 [0.7344, 0.8202] | 0.638 [0.580, 0.709] | no |
| xeus+ipa | 1.0 | 0.8785 | 0.7850 [0.7180, 0.8369] | 0.7866 [0.7382, 0.8258] | 0.623 [0.574, 0.686] | yes |
| xlsr+phoneticxeus+ipa | 1.0 | 0.8757 | 0.8019 [0.7466, 0.8462] | 0.7943 [0.7478, 0.8323] | 0.615 [0.568, 0.675] | yes |
| xlsr | 1.0 | 0.8749 | 0.7821 [0.7286, 0.8254] | 0.7700 [0.7183, 0.8129] | 0.647 [0.599, 0.705] | no |
| xlsr+phoneticxeus | 1.0 | 0.8748 | 0.7887 [0.7195, 0.8405] | 0.7888 [0.7422, 0.8275] | 0.621 [0.571, 0.685] | yes |
| xlsr+ipa | 1.0 | 0.8680 | 0.7909 [0.7411, 0.8324] | 0.7807 [0.7308, 0.8217] | 0.635 [0.590, 0.689] | yes |
| phoneticxeus+ipa | 1.0 | 0.8624 | 0.7758 [0.7012, 0.8311] | 0.7750 [0.7270, 0.8155] | 0.637 [0.585, 0.705] | yes |
| phoneticxeus | 1.0 | 0.8609 | 0.7549 [0.6612, 0.8221] | 0.7630 [0.7136, 0.8057] | 0.651 [0.593, 0.727] | yes |
| ipa | 0.001 | 0.6972 | 0.6067 [0.5176, 0.6784] | 0.5804 [0.4913, 0.6550] | 0.848 [0.787, 0.916] | yes |

**prosody**

| blocks | penalty | development Spearman | Pearson [95% CI] | Spearman [95% CI] | MAE [95% CI] | exposed |
|---|---|---|---|---|---|---|
| xeus+xlsr | 1.0 | 0.8901 | 0.7887 [0.7317, 0.8348] | 0.7859 [0.7305, 0.8271] | 0.632 [0.581, 0.695] | no |
| xeus+xlsr+ipa | 1.0 | 0.8894 | 0.7933 [0.7409, 0.8378] | 0.7888 [0.7329, 0.8302] | 0.626 [0.576, 0.688] | yes |
| xeus+xlsr+phoneticxeus+ipa | 10.0 | 0.8850 | 0.7983 [0.7413, 0.8457] | 0.8023 [0.7511, 0.8400] | 0.616 [0.561, 0.685] | yes |
| xeus+xlsr+phoneticxeus | 10.0 | 0.8846 | 0.7856 [0.7181, 0.8389] | 0.7954 [0.7447, 0.8335] | 0.626 [0.568, 0.699] | yes |
| xeus+ipa | 1.0 | 0.8841 | 0.7791 [0.7175, 0.8293] | 0.7804 [0.7245, 0.8214] | 0.637 [0.585, 0.703] | yes |
| xeus | 1.0 | 0.8836 | 0.7730 [0.7032, 0.8263] | 0.7756 [0.7191, 0.8168] | 0.644 [0.591, 0.713] | no |
| xeus+phoneticxeus+ipa | 1.0 | 0.8814 | 0.7893 [0.7304, 0.8365] | 0.7876 [0.7324, 0.8266] | 0.628 [0.575, 0.692] | yes |
| xeus+phoneticxeus | 1.0 | 0.8806 | 0.7756 [0.7029, 0.8301] | 0.7809 [0.7253, 0.8208] | 0.636 [0.581, 0.706] | yes |
| xlsr+phoneticxeus+ipa | 1.0 | 0.8678 | 0.8017 [0.7517, 0.8446] | 0.7939 [0.7410, 0.8344] | 0.620 [0.571, 0.684] | yes |
| xlsr+phoneticxeus | 1.0 | 0.8670 | 0.7893 [0.7292, 0.8382] | 0.7882 [0.7334, 0.8287] | 0.630 [0.577, 0.698] | yes |
| xlsr | 1.0 | 0.8624 | 0.7797 [0.7293, 0.8216] | 0.7657 [0.7082, 0.8094] | 0.660 [0.609, 0.722] | no |
| xlsr+ipa | 1.0 | 0.8582 | 0.7874 [0.7388, 0.8298] | 0.7777 [0.7203, 0.8201] | 0.639 [0.590, 0.698] | yes |
| phoneticxeus+ipa | 1.0 | 0.8554 | 0.7688 [0.6937, 0.8268] | 0.7763 [0.7233, 0.8170] | 0.644 [0.585, 0.718] | yes |
| phoneticxeus | 1.0 | 0.8525 | 0.7504 [0.6591, 0.8171] | 0.7637 [0.7094, 0.8065] | 0.658 [0.595, 0.740] | yes |
| ipa | 1.0 | 0.6896 | 0.6057 [0.5131, 0.6787] | 0.6021 [0.5162, 0.6708] | 0.835 [0.765, 0.911] | yes |

### T-cov

**accuracy**

| blocks | penalty | development Spearman | Pearson [95% CI] | Spearman [95% CI] | MAE [95% CI] | exposed |
|---|---|---|---|---|---|---|
| xeus+phoneticxeus+ipa | 10.0 | 0.8163 | 0.7176 [0.6551, 0.7636] | 0.7049 [0.6431, 0.7533] | 0.784 [0.723, 0.853] | yes |
| xeus+xlsr+phoneticxeus+ipa | 10.0 | 0.8136 | 0.7321 [0.6728, 0.7752] | 0.7135 [0.6512, 0.7624] | 0.771 [0.713, 0.835] | yes |
| xeus+ipa | 10.0 | 0.8124 | 0.7109 [0.6501, 0.7579] | 0.7000 [0.6361, 0.7507] | 0.794 [0.733, 0.861] | yes |
| xeus+xlsr+ipa | 10.0 | 0.8100 | 0.7260 [0.6668, 0.7694] | 0.7086 [0.6449, 0.7595] | 0.782 [0.724, 0.844] | yes |
| xeus | 10.0 | 0.8043 | 0.6841 [0.6170, 0.7369] | 0.6838 [0.6210, 0.7331] | 0.810 [0.745, 0.883] | no |
| xeus+xlsr | 10.0 | 0.8031 | 0.7045 [0.6409, 0.7513] | 0.6970 [0.6328, 0.7473] | 0.795 [0.733, 0.863] | no |
| xeus+phoneticxeus | 10.0 | 0.8004 | 0.6824 [0.6132, 0.7364] | 0.6828 [0.6206, 0.7320] | 0.809 [0.741, 0.886] | yes |
| xeus+xlsr+phoneticxeus | 10.0 | 0.7988 | 0.7010 [0.6361, 0.7495] | 0.6947 [0.6305, 0.7446] | 0.797 [0.735, 0.868] | yes |
| xlsr+phoneticxeus+ipa | 10.0 | 0.7964 | 0.7154 [0.6530, 0.7596] | 0.6936 [0.6284, 0.7467] | 0.794 [0.733, 0.861] | yes |
| phoneticxeus+ipa | 1.0 | 0.7944 | 0.7020 [0.6342, 0.7519] | 0.6787 [0.6138, 0.7294] | 0.808 [0.747, 0.877] | yes |
| xlsr+ipa | 1.0 | 0.7775 | 0.7128 [0.6504, 0.7590] | 0.6757 [0.6023, 0.7345] | 0.807 [0.751, 0.865] | yes |
| xlsr+phoneticxeus | 10.0 | 0.7747 | 0.6740 [0.6048, 0.7267] | 0.6632 [0.5929, 0.7175] | 0.827 [0.761, 0.902] | yes |
| phoneticxeus | 1.0 | 0.7715 | 0.6542 [0.5763, 0.7141] | 0.6442 [0.5763, 0.6987] | 0.848 [0.779, 0.928] | yes |
| xlsr | 1.0 | 0.7684 | 0.6810 [0.6122, 0.7325] | 0.6495 [0.5717, 0.7093] | 0.830 [0.770, 0.896] | no |
| ipa | 1.0 | 0.6808 | 0.5768 [0.5033, 0.6389] | 0.5520 [0.4671, 0.6258] | 0.930 [0.854, 1.013] | yes |

**fluency**

| blocks | penalty | development Spearman | Pearson [95% CI] | Spearman [95% CI] | MAE [95% CI] | exposed |
|---|---|---|---|---|---|---|
| xeus+xlsr | 1.0 | 0.8865 | 0.7890 [0.7229, 0.8400] | 0.7884 [0.7393, 0.8273] | 0.620 [0.573, 0.678] | no |
| xeus+xlsr+ipa | 1.0 | 0.8848 | 0.7982 [0.7415, 0.8434] | 0.7942 [0.7454, 0.8324] | 0.617 [0.572, 0.672] | yes |
| xeus+xlsr+phoneticxeus+ipa | 10.0 | 0.8829 | 0.8021 [0.7431, 0.8490] | 0.8059 [0.7604, 0.8411] | 0.604 [0.554, 0.666] | yes |
| xeus+xlsr+phoneticxeus | 10.0 | 0.8825 | 0.7871 [0.7142, 0.8415] | 0.7981 [0.7523, 0.8342] | 0.616 [0.562, 0.684] | yes |
| xeus+phoneticxeus+ipa | 10.0 | 0.8800 | 0.7906 [0.7216, 0.8437] | 0.7976 [0.7509, 0.8345] | 0.613 [0.560, 0.678] | yes |
| xeus+phoneticxeus | 10.0 | 0.8796 | 0.7743 [0.6881, 0.8358] | 0.7878 [0.7425, 0.8256] | 0.627 [0.569, 0.699] | yes |
| xeus | 10.0 | 0.8791 | 0.7693 [0.6817, 0.8325] | 0.7821 [0.7346, 0.8204] | 0.637 [0.581, 0.707] | no |
| xeus+ipa | 1.0 | 0.8785 | 0.7861 [0.7197, 0.8371] | 0.7869 [0.7386, 0.8267] | 0.623 [0.575, 0.683] | yes |
| xlsr+phoneticxeus+ipa | 1.0 | 0.8757 | 0.8038 [0.7496, 0.8470] | 0.7954 [0.7487, 0.8336] | 0.613 [0.568, 0.670] | yes |
| xlsr | 1.0 | 0.8749 | 0.7829 [0.7287, 0.8258] | 0.7708 [0.7191, 0.8139] | 0.646 [0.599, 0.703] | no |
| xlsr+phoneticxeus | 1.0 | 0.8748 | 0.7899 [0.7219, 0.8409] | 0.7893 [0.7426, 0.8275] | 0.620 [0.571, 0.681] | yes |
| xlsr+ipa | 1.0 | 0.8680 | 0.7924 [0.7430, 0.8335] | 0.7821 [0.7318, 0.8232] | 0.633 [0.589, 0.686] | yes |
| phoneticxeus+ipa | 1.0 | 0.8624 | 0.7785 [0.7070, 0.8320] | 0.7766 [0.7284, 0.8161] | 0.635 [0.584, 0.700] | yes |
| phoneticxeus | 1.0 | 0.8609 | 0.7566 [0.6657, 0.8224] | 0.7634 [0.7140, 0.8058] | 0.650 [0.593, 0.723] | yes |
| ipa | 0.001 | 0.6972 | 0.6094 [0.5201, 0.6813] | 0.5824 [0.4939, 0.6568] | 0.846 [0.785, 0.912] | yes |

**prosody**

| blocks | penalty | development Spearman | Pearson [95% CI] | Spearman [95% CI] | MAE [95% CI] | exposed |
|---|---|---|---|---|---|---|
| xeus+xlsr | 1.0 | 0.8901 | 0.7899 [0.7324, 0.8358] | 0.7867 [0.7308, 0.8280] | 0.630 [0.581, 0.692] | no |
| xeus+xlsr+ipa | 1.0 | 0.8894 | 0.7947 [0.7413, 0.8381] | 0.7900 [0.7347, 0.8314] | 0.624 [0.576, 0.684] | yes |
| xeus+xlsr+phoneticxeus+ipa | 10.0 | 0.8850 | 0.8003 [0.7443, 0.8466] | 0.8036 [0.7525, 0.8407] | 0.614 [0.560, 0.681] | yes |
| xeus+xlsr+phoneticxeus | 10.0 | 0.8846 | 0.7871 [0.7197, 0.8392] | 0.7961 [0.7454, 0.8346] | 0.624 [0.567, 0.695] | yes |
| xeus+ipa | 1.0 | 0.8841 | 0.7806 [0.7192, 0.8297] | 0.7813 [0.7251, 0.8224] | 0.635 [0.584, 0.698] | yes |
| xeus | 1.0 | 0.8836 | 0.7740 [0.7044, 0.8268] | 0.7758 [0.7193, 0.8173] | 0.643 [0.591, 0.709] | no |
| xeus+phoneticxeus+ipa | 1.0 | 0.8814 | 0.7908 [0.7317, 0.8376] | 0.7886 [0.7335, 0.8279] | 0.626 [0.575, 0.688] | yes |
| xeus+phoneticxeus | 1.0 | 0.8806 | 0.7763 [0.7031, 0.8306] | 0.7813 [0.7255, 0.8212] | 0.635 [0.581, 0.703] | yes |
| xlsr+phoneticxeus+ipa | 1.0 | 0.8678 | 0.8036 [0.7540, 0.8459] | 0.7952 [0.7427, 0.8356] | 0.618 [0.570, 0.679] | yes |
| xlsr+phoneticxeus | 1.0 | 0.8670 | 0.7903 [0.7300, 0.8391] | 0.7889 [0.7337, 0.8294] | 0.629 [0.576, 0.694] | yes |
| xlsr | 1.0 | 0.8624 | 0.7807 [0.7303, 0.8221] | 0.7669 [0.7094, 0.8107] | 0.658 [0.608, 0.718] | no |
| xlsr+ipa | 1.0 | 0.8582 | 0.7892 [0.7404, 0.8308] | 0.7791 [0.7220, 0.8214] | 0.636 [0.588, 0.694] | yes |
| phoneticxeus+ipa | 1.0 | 0.8554 | 0.7722 [0.7003, 0.8281] | 0.7786 [0.7257, 0.8193] | 0.641 [0.584, 0.713] | yes |
| phoneticxeus | 1.0 | 0.8525 | 0.7524 [0.6624, 0.8170] | 0.7645 [0.7095, 0.8074] | 0.656 [0.594, 0.735] | yes |
| ipa | 1.0 | 0.6896 | 0.6091 [0.5162, 0.6824] | 0.6049 [0.5193, 0.6734] | 0.833 [0.764, 0.908] | yes |

## S4. The repository fusion network on the S1 block sets, five seeds

Two networks cover the five selection rules. The `scaler_mean`,
`scaler_scale` and `block_gain` of the released heads are bitwise
identical across the five rules for a given block set, so the
standardized design matrix the network reads depends on the block set
alone, and the five rules differ only in the ridge penalty, which a
network never uses. **The five rule rows of one block set therefore share
a single network and are not five independent results.** Each row below is
one block set, not one rule. `evaluate.py` asserts the identity and
raises if a future release breaks it.

Mean and standard deviation across the five seeds, as the protocol
fixes. The epoch of every network is chosen by the accuracy mean
absolute error on the development split, which is what the protocol and
speech/scripts/train_unified.py fix, including for the network whose
released constructs are fluency and prosody. The column "in S1
pairing" marks the one construct per block set whose released head
uses that block set. The other two rows of a network are reported but
are not part of the exposure pairing.

### T-all

| blocks | construct | Pearson mean ± sd | Spearman mean ± sd | MAE mean ± sd | best epochs | exposed | in S1 pairing |
|---|---|---|---|---|---|---|---|
| xeus+phoneticxeus+ipa | accuracy | 0.6887 ± 0.0074 | 0.6653 ± 0.0093 | 0.851 ± 0.011 | 27, 18, 12, 16, 17 | yes | yes |
| xeus+phoneticxeus+ipa | fluency | 0.7594 ± 0.0082 | 0.7509 ± 0.0057 | 0.684 ± 0.011 | 27, 18, 12, 16, 17 | yes | no |
| xeus+phoneticxeus+ipa | prosody | 0.7548 ± 0.0119 | 0.7536 ± 0.0114 | 0.679 ± 0.009 | 27, 18, 12, 16, 17 | yes | no |
| xeus+xlsr | accuracy | 0.6711 ± 0.0066 | 0.6401 ± 0.0149 | 0.880 ± 0.010 | 277, 243, 11, 276, 10 | no | no |
| xeus+xlsr | fluency | 0.7481 ± 0.0143 | 0.7478 ± 0.0134 | 0.700 ± 0.033 | 277, 243, 11, 276, 10 | no | yes |
| xeus+xlsr | prosody | 0.7458 ± 0.0090 | 0.7418 ± 0.0079 | 0.698 ± 0.012 | 277, 243, 11, 276, 10 | no | yes |

### T-cov

| blocks | construct | Pearson mean ± sd | Spearman mean ± sd | MAE mean ± sd | best epochs | exposed | in S1 pairing |
|---|---|---|---|---|---|---|---|
| xeus+phoneticxeus+ipa | accuracy | 0.6894 ± 0.0079 | 0.6650 ± 0.0096 | 0.851 ± 0.011 | 27, 18, 12, 16, 17 | yes | yes |
| xeus+phoneticxeus+ipa | fluency | 0.7600 ± 0.0084 | 0.7508 ± 0.0062 | 0.684 ± 0.012 | 27, 18, 12, 16, 17 | yes | no |
| xeus+phoneticxeus+ipa | prosody | 0.7559 ± 0.0121 | 0.7543 ± 0.0116 | 0.677 ± 0.010 | 27, 18, 12, 16, 17 | yes | no |
| xeus+xlsr | accuracy | 0.6721 ± 0.0067 | 0.6394 ± 0.0150 | 0.881 ± 0.010 | 277, 243, 11, 276, 10 | no | no |
| xeus+xlsr | fluency | 0.7491 ± 0.0142 | 0.7480 ± 0.0134 | 0.699 ± 0.033 | 277, 243, 11, 276, 10 | no | yes |
| xeus+xlsr | prosody | 0.7474 ± 0.0088 | 0.7429 ± 0.0082 | 0.696 ± 0.012 | 277, 243, 11, 276, 10 | no | yes |

## Upstream exposure, reported side by side

Both sides of the pairing are fixed in advance, so that neither side
gains the advantage of having been picked by the test partition.

The exposed side is the released application default, the `spearman`
rule, for S1 and S2. For S3 it is the exposed block combination chosen
by the same rule the clean side uses, the highest development Spearman
correlation with ties broken by the lower development mean absolute
error. For S4 it is the network whose block set is the one the released
head uses for that construct, reported as mean and standard deviation
over the five seeds.

The clean side is the S3 system whose blocks contain neither
`phoneticxeus` nor `ipa`, chosen by the rule module X6 uses: the highest
development Spearman correlation, ties broken by the lower development
mean absolute error. X6 averages that Spearman over three corpora with
equal weights and reports a development-size-weighted variant beside
it. S3 fits on one corpus and selects on one development set of 500
utterances, so the mean has a single term and the two weightings select
the same system. There is no second row to report.

For fluency and prosody the released heads, under the `spearman` rule,
use XEUS and XLS-R, which carry no exposure. Those two constructs
therefore have no exposed S1 or S2 row, and their S4 network of the S1
block set is not exposed either. Their pairing shows the clean
reference, the exposed S3 system chosen on development data, and the
descriptive rows.

Below the pre-specified pair, the system that is best by test Pearson
correlation is given for each side, marked descriptive, because that
figure does involve the test partition and carries the advantage of
having been chosen on it.

### T-all

| construct | role | blocks | rule | Pearson [95% CI] or mean ± sd | MAE [95% CI] or mean ± sd | exposed |
|---|---|---|---|---|---|---|
| accuracy | clean reference, X6 rule on dev | xeus | development-selected | 0.6835 [0.6175, 0.7370] | 0.810 [0.745, 0.883] | no |
| accuracy | S1, exposed, spearman rule as released | xeus+phoneticxeus+ipa | spearman | 0.7165 [0.6541, 0.7625] | 0.784 [0.723, 0.854] | yes |
| accuracy | S2, exposed, spearman rule as released | xeus+phoneticxeus+ipa | spearman | 0.7164 [0.6521, 0.7645] | 0.786 [0.726, 0.854] | yes |
| accuracy | S3, exposed, same dev rule | xeus+phoneticxeus+ipa | development-selected | 0.7165 [0.6541, 0.7625] | 0.784 [0.723, 0.854] | yes |
| accuracy | S4, exposed, S1 block set, 5 seeds | xeus+phoneticxeus+ipa | mean over seeds | 0.6887 ± 0.0074 | 0.851 ± 0.011 | yes |
| accuracy | descriptive, best clean by test Pearson | xeus+xlsr | S3 development-selected | 0.7039 [0.6407, 0.7511] | 0.794 [0.732, 0.863] | no |
| accuracy | descriptive, best exposed by test Pearson | xeus+xlsr+phoneticxeus+ipa | S3 development-selected | 0.7312 [0.6717, 0.7744] | 0.772 [0.714, 0.836] | yes |
| fluency | clean reference, X6 rule on dev | xeus+xlsr | development-selected | 0.7882 [0.7221, 0.8397] | 0.621 [0.573, 0.681] | no |
| fluency | S3, exposed, same dev rule | xeus+xlsr+ipa | development-selected | 0.7969 [0.7397, 0.8425] | 0.618 [0.572, 0.676] | yes |
| fluency | descriptive, best clean by test Pearson | xeus+xlsr | S2 banded_3ratio_spearman | 0.7948 [0.7222, 0.8482] | 0.606 [0.558, 0.667] | no |
| fluency | descriptive, best exposed by test Pearson | xlsr+phoneticxeus+ipa | S3 development-selected | 0.8019 [0.7466, 0.8462] | 0.615 [0.568, 0.675] | yes |
| prosody | clean reference, X6 rule on dev | xeus+xlsr | development-selected | 0.7887 [0.7317, 0.8348] | 0.632 [0.581, 0.695] | no |
| prosody | S3, exposed, same dev rule | xeus+xlsr+ipa | development-selected | 0.7933 [0.7409, 0.8378] | 0.626 [0.576, 0.688] | yes |
| prosody | descriptive, best clean by test Pearson | xeus+xlsr | S1 banded_3ratio_spearman | 0.7931 [0.7351, 0.8402] | 0.623 [0.571, 0.688] | no |
| prosody | descriptive, best exposed by test Pearson | xlsr+phoneticxeus+ipa | S3 development-selected | 0.8017 [0.7517, 0.8446] | 0.620 [0.571, 0.684] | yes |

### T-cov

| construct | role | blocks | rule | Pearson [95% CI] or mean ± sd | MAE [95% CI] or mean ± sd | exposed |
|---|---|---|---|---|---|---|
| accuracy | clean reference, X6 rule on dev | xeus | development-selected | 0.6841 [0.6170, 0.7369] | 0.810 [0.745, 0.883] | no |
| accuracy | S1, exposed, spearman rule as released | xeus+phoneticxeus+ipa | spearman | 0.7176 [0.6551, 0.7636] | 0.784 [0.723, 0.853] | yes |
| accuracy | S2, exposed, spearman rule as released | xeus+phoneticxeus+ipa | spearman | 0.7180 [0.6536, 0.7665] | 0.785 [0.726, 0.852] | yes |
| accuracy | S3, exposed, same dev rule | xeus+phoneticxeus+ipa | development-selected | 0.7176 [0.6551, 0.7636] | 0.784 [0.723, 0.853] | yes |
| accuracy | S4, exposed, S1 block set, 5 seeds | xeus+phoneticxeus+ipa | mean over seeds | 0.6894 ± 0.0079 | 0.851 ± 0.011 | yes |
| accuracy | descriptive, best clean by test Pearson | xeus+xlsr | S3 development-selected | 0.7045 [0.6409, 0.7513] | 0.795 [0.733, 0.863] | no |
| accuracy | descriptive, best exposed by test Pearson | xeus+xlsr+phoneticxeus+ipa | S3 development-selected | 0.7321 [0.6728, 0.7752] | 0.771 [0.713, 0.835] | yes |
| fluency | clean reference, X6 rule on dev | xeus+xlsr | development-selected | 0.7890 [0.7229, 0.8400] | 0.620 [0.573, 0.678] | no |
| fluency | S3, exposed, same dev rule | xeus+xlsr+ipa | development-selected | 0.7982 [0.7415, 0.8434] | 0.617 [0.572, 0.672] | yes |
| fluency | descriptive, best clean by test Pearson | xeus+xlsr | S2 banded_3ratio_spearman | 0.7960 [0.7243, 0.8485] | 0.605 [0.558, 0.663] | no |
| fluency | descriptive, best exposed by test Pearson | xlsr+phoneticxeus+ipa | S3 development-selected | 0.8038 [0.7496, 0.8470] | 0.613 [0.568, 0.670] | yes |
| prosody | clean reference, X6 rule on dev | xeus+xlsr | development-selected | 0.7899 [0.7324, 0.8358] | 0.630 [0.581, 0.692] | no |
| prosody | S3, exposed, same dev rule | xeus+xlsr+ipa | development-selected | 0.7947 [0.7413, 0.8381] | 0.624 [0.576, 0.684] | yes |
| prosody | descriptive, best clean by test Pearson | xeus+xlsr | S1 banded_3ratio_spearman | 0.7944 [0.7364, 0.8409] | 0.621 [0.571, 0.684] | no |
| prosody | descriptive, best exposed by test Pearson | xlsr+phoneticxeus+ipa | S3 development-selected | 0.8036 [0.7540, 0.8459] | 0.618 [0.570, 0.679] | yes |

## Published reference points

Recorded in PROTOCOL.md before this module produced a number, so they
could not inform any choice here. They are not comparisons computed by
this module: the splits, the targets and the metric conventions of those
papers were not re-derived.

- GOPT: fluency 0.753, prosodic 0.760
- HMamba: fluency 0.848, prosodic 0.843

Accuracy figures, and the numbers of Kim et al. 2022, are to be confirmed
against the primary sources at writing time.
