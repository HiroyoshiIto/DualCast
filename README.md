# DualCast

## Requirements

- PyTorch (1.0.0 or later)
- python 3.8
- networkx
- numpy
- scipy

## How to run

You can try to run DualCast by following command in src directory:

`python train.py`

You can specify a dataset as follows:

`python train.py --dataset DBLP`

There are three datasets (`NIPS`, `DBLP`, and `Twitter`) in this repository.

## Paper
The latest version is available on [SIAM Publications Libarary](https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.6).
If you use our code for academic work, please cite
```
@inproceedings{ito2022dualcast,
  title={DualCast: Friendship-Preference Co-evolution Forecasting for Attributed Networks},
  author={Ito, Hiroyoshi and Faloutsos, Christos},
  booktitle={Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)},
  pages={46--54},
  month={Apr},
  year={2022},
  organization={SIAM},
  doi={https://doi.org/10.1137/1.9781611977172.6}
}
```
