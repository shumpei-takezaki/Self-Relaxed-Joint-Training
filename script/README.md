## Proposed methods
- `co_teaching_ours`: [Co-teaching](https://proceedings.neurips.cc/paper_files/paper/2018/file/a19744e268754fb0148b017647355b7b-Paper.pdf) with our framework
- `jocor_ours`: [JoCor](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf) with our framework
- `codis_ours`: [CoDis](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_Combating_Noisy_Labels_with_Sample_Selection_by_Mining_High-Discrepancy_Examples_ICCV_2023_paper.pdf) with our framework
- *`co_teaching_abl`: Ablation study for Co-teaching with our framework

## Comparative methods
- `standard`: Training with Cross Entropy loss
- `sord`: [R.Diaz+, "Soft Labels for Ordinal Regression", CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf)
- `label_smooth`: [R.Mülle+, "When Does Label Smoothing Help?", NeurIPS2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf)
- `f_correction`: [G.Patrini+, "Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach", CVPR2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf)
- `reweight`: [T.Liu+, "Classification with Noisy Labels by Importance Reweighting", IEEE Trans. Pattern Anal. Mach. Intell.](https://arxiv.org/pdf/1411.7718)
- `mixup`: [H.Zhang+, "Mixup: Beyond Empirical Risk Minimization", ICLR2018](https://openreview.net/pdf?id=r1Ddp1-Rb)
- `cdr`: [X.Xia+, "Robust Early-learning: Hindering the Memorization of Noisy Labels", ICLR2021](https://openreview.net/pdf?id=Eql5b1_hTE4)
- `garg`: [B.Garg+, "Robust Deep Ordinal Regression under Label Noise", ACML2020](https://proceedings.mlr.press/v129/garg20a/garg20a.pdf)
- `co_teaching`: [B.Han+, "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels", NeurIPS2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/a19744e268754fb0148b017647355b7b-Paper.pdf)
- `jocor`: [H.Wei+, "Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization", CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf)
- `codis`: [X.Xia+, "Combating Noisy Labels with Sample Selection by Mining High-Discrepancy Examples", ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_Combating_Noisy_Labels_with_Sample_Selection_by_Mining_High-Discrepancy_Examples_ICCV_2023_paper.pdf)