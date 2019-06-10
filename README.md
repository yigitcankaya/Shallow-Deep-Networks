Source Code for ICML 2019 Paper "Shallow-Deep Networks: Understanding and Mitigating Network Overthinking"

Yigitcan Kaya, Sanghyun Hong, Tudor Dumitras

University of Maryland, College Park

Project Website: http://shallowdeep.network

Contact: Yigitcan Kaya - cankaya at umiacs.umd dot edu


Requirements:
- Python 3.7
- PyTorch 1.0
- CUDA 8.0
- CUDNN 7.5
- Matplotlib Pyplot

Required data sets:
- Download TinyImageNet from https://tiny-imagenet.herokuapp.com/, place it under data/ and use data.py - create_val_folder() to generate proper directory structure

- CIFAR-10 and CIFAR-100 will be downloaded automatically


Source code files and corresponding sections in the paper:

- Section 3.0: The Shallow-Deep Network --- SDNs/VGG_SDN.py, SDNs/ResNet_SDN.py, SDNs/MobileNet_SDN.py and SDNs/WideResNet_SDN.py implements the SDNs, train_networks.py trains CNNs and SDNs

- Section 4.0: Understanding the Overthinking Problem  --- overthinking.py quantifies the wasteful and destructive effects and generates the explanatory images

- Section 5.1: Confidence-based Early Exits --- early_exit_experiments.py searches for the early-exit threshold and returns the average inference cost for early exits

- Section 5.2: Network Confusion Analysis/Confusion Metric is an Error Indicator --- confusion_experiments.py computes the normalized confusion scores and generates the confusion histogram of the SDN on correct and wrong classifications.

- Section 5.2: Network Confusion Analysis/Visualizing Confusion Helps with Error Diagnosis --- gradcam_experiments.py finds the test samples that are classified correctly by the first internal classifier and wrongly by the final classifier and uses GradCam to visualize the prominent features that lead to this disagreement





