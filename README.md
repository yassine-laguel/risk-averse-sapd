# risk-averse-sapd


This repository contains the dataset and code to reproduce the numerical illustrations from our paper "High Probability and Risk-averse guarantees for Stochastic Accelerated Primal Dual Methods", available [here](https://arxiv.org/abs/2304.00444).

The toy problems implemented in this repository incorporates code from the authors of the [Initial SAPD paper](https://arxiv.org/abs/2111.12743) who graciously shared their implementation.

## Requirements

Required packages may be installed via pip :
```bash
pip install -r requirements.txt
```

## Running Experiments

### Bilinear games

```
python bilinear.py
```

### Distributionally Robust Logistic Regression

```
python arcene.py
```

```
python drybean.py
```

## Citation
If you find this repository useful, or you use it in your research, please cite:
```
@article{laguel2023high,
  title={High probability and risk-averse guarantees for stochastic saddle point problems},
  author={Laguel, Yassine and Aybat, Necdet Serhat and G{\"u}rb{\"u}zbalaban, Mert},
  journal={arXiv preprint arXiv:2304.00444},
  year={2023}
}

@article{zhang2021robust,
  title={Robust accelerated primal-dual methods for computing saddle points},
  author={Zhang, Xuan and Aybat, Necdet Serhat and G{\"u}rb{\"u}zbalaban, Mert},
  journal={arXiv preprint arXiv:2111.12743},
  year={2021}
}
```
