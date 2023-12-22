## [ICML 2019] Shallow-Deep Networks (Website: [http://shallowdeep.network](http://shallowdeep.network))

This repository contains the code for reproducing the results in our paper:

- [Shallow-Deep Networks: Understanding and Mitigating Network Overthinking](https://arxiv.org/abs/1810.07052)
- [Yigitcan Kaya](http://www.cs.umd.edu/~yigitcan/), [Sanghyun Hong](https://sanghyun-hong.com), Tudor Dumitras
- University of Maryland, College Park

---

### Abstract

We characterize a prevalent weakness of deep neural networks (DNNs)-overthinking-which occurs when a DNN can reach correct predictions before its final layer. Overthinking is computationally wasteful, and it can also be destructive when, by the final layer, a correct prediction changes into a misclassification. Understanding overthinking requires studying how each prediction evolves during a DNN's forward pass, which conventionally is opaque. For prediction transparency, we propose the Shallow-Deep Network (SDN), a generic modification to off-the-shelf DNNs that introduces internal classifiers. We apply SDN to four modern architectures, trained on three image classification tasks, to characterize the overthinking problem. We show that SDNs can mitigate the wasteful effect of overthinking with confidence-based early exits, which reduce the average inference cost by more than 50% and preserve the accuracy. We also find that the destructive effect occurs for 50% of misclassifications on natural inputs and that it can be induced, adversarially, with a recent backdooring attack. To mitigate this effect, we propose a new confusion metric to quantify the internal disagreements that will likely lead to misclassifications.

---

### Pre-requisites

Requirements:
- Python 3.7
- PyTorch 1.0
- CUDA 8.0
- CUDNN 7.5
- Matplotlib Pyplot

Required data sets:
- CIFAR-10/100 (downloaded automatically)
- [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/)
- Download and place it under data/ and use data.py - create_val_folder() to generate proper directory structure

---

### Reproducing the results

Source code files and corresponding sections in the paper:

- Section 3.0: The Shallow-Deep Network --- SDNs/VGG_SDN.py, SDNs/ResNet_SDN.py, SDNs/MobileNet_SDN.py and SDNs/WideResNet_SDN.py implements the SDNs, train_networks.py trains CNNs and SDNs

- Section 4.0: Understanding the Overthinking Problem  --- overthinking.py quantifies the wasteful and destructive effects and generates the explanatory images

- Section 5.1: Confidence-based Early Exits --- early_exit_experiments.py searches for the early-exit threshold and returns the average inference cost for early exits

- Section 5.2: Network Confusion Analysis/Confusion Metric is an Error Indicator --- confusion_experiments.py computes the normalized confusion scores and generates the confusion histogram of the SDN on correct and wrong classifications.

- Section 5.2: Network Confusion Analysis/Visualizing Confusion Helps with Error Diagnosis --- gradcam_experiments.py finds the test samples that are classified correctly by the first internal classifier and wrongly by the final classifier and uses GradCam to visualize the prominent features that lead to this disagreement

---

### Cite Our Work

Please cite our work if you find our work is helpful.

```
@InProceedings{Kaya2019SDN,
  title={Shallow-Deep Networks: Understanding and Mitigating Network Overthinking},
  author={Kaya, Yigitcan and Hong, Sanghyun and Dumitras, Tudor},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  pages={3301--3310},
  year={2019},
  editor={Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume={97},
  series={Proceedings of Machine Learning Research},
  month={09--15 Jun},
  publisher={PMLR},
  pdf={http://proceedings.mlr.press/v97/kaya19a/kaya19a.pdf},
  url={https://proceedings.mlr.press/v97/kaya19a.html},
}
```

---

&nbsp;

Please contact [Yigitcan Kaya](mailto:cankaya@umiacs.umd.edu) for any questions and recommendations.




