# Efficient-backdoor-attacks-for-deep-neural-networks-in-real-world-scenarios (PyTorch)

[Efficient Backdoor Attacks for Deep Neural Networks in Real-World Scenarios]()

Ziqiang Li, Hong Sun, Pengfei Xia, Heng Li, Beihao Xia, Yi Wu, and Bin Li

>Abstract: *Recent deep neural networks (DNNs) have came to rely on vast amounts of training data, providing an opportunity for malicious attackers to exploit and contaminate the data to carry out backdoor attacks. However, existing backdoor attack methods make unrealistic assumptions, assuming that all training data comes from a single source and that attackers have full access to the training data. In this paper, we introduce a more realistic attack scenario where victims collect data from multiple sources, and attackers cannot access the complete training data. We refer to this scenario as data-constrained backdoor attacks. In such cases, previous attack methods suffer from severe efficiency degradation due to the entanglement between benign and poisoning features during the backdoor injection process. To tackle this problem, we introduce three CLIP-based technologies from two distinct streams: Clean Feature Suppression and Poisoning Feature Augmentation The results demonstrate remarkable improvements, with some settings achieving over 100% improvement compared to existing attacks in data-constrained scenarios.*

## Number-constrained

```python
# poison_source = 'origin' is baseline, poison_source = 'clip' if CLIP-CFE.
python numbers_exp.py
```

## Class-constrained

```python
# poison_source = 'origin' is baseline, poison_source = 'clip' if CLIP-CFE.
python classes_xep.py
```

## Domain-constrained

```python
# poison_source = 'origin' is baseline, poison_source = 'clip' if CLIP-CFE.
python domains_exp.py
```

## Generate clip-cfe data

```python
# clip-cfe data for cifar-100
python clip_data_cfe.py
```

## Generate clip-uap and clip-cfa triggers

```python
# clip-uap and clip-cfa triggers for cifar-100
python clip_uap_cfa.py
```

## Generate poison idx

```python
# Generate poison idx
python poison_idx.py
```

## Citation

If you find this work useful for your research, please consider citing our paper:

@inproceedings{li2023efficient,
  title={Efficient Backdoor Attacks for Deep Neural Networks in Real-world Scenarios},
  author={Li, Ziqiang and Sun, Hong and Xia, Pengfei and Li, Heng and Xia, Beihao and Wu, Yi and Li, Bin},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
