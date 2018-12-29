
# Caser-PyTorch

A PyTorch implementation of Convolutional Sequence Embedding Recommendation Model (Caser) from the paper:

*Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18*

# Requirements
* Python 2 or 3
* [PyTorch v0.4+](https://github.com/pytorch/pytorch)
* Numpy
* SciPy

# Usage
1. Install required packages.
2. run <code>python train_caser.py</code>

# Configurations

#### Data

- Datasets are organized into 2 separate files: **_train.txt_** and **_test.txt_**

- Same to other data format for recommendation, each file contains a collection of triplets:

  > user item rating

  The only difference is the triplets are organized in *time order*.

- As the problem is Sequential Recommendation, the rating doesn't matter, so I convert them to all 1.

#### Model Args (in train_caser.py)

- <code>L</code>: length of sequence
- <code>T</code>: number of targets
- <code>d</code>: number of latent dimensions
- <code>nv</code>: number of vertical filters
- <code>nh</code>: number of horizontal filters
- <code>ac_conv</code>: activation function for convolution layer (i.e., phi_c in paper)
- <code>ac_fc</code>: activation function for fully-connected layer (i.e., phi_a in paper)
- <code>drop_rate</code>: drop ratio when performing dropout


# Citation

If you use this Caser in your paper, please cite the paper:

```
@inproceedings{tang2018caser,
  title={Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding},
  author={Tang, Jiaxi and Wang, Ke},
  booktitle={ACM International Conference on Web Search and Data Mining},
  year={2018}
}
```

# Comments

1. This PyTorch version may get better performance than what the paper reports. 

   > When d=50, L=5, T=3, and set other arguments to default, after 20 epochs, mAP may get to 0.17 on the test set.

# Acknowledgment

This project (utils.py, interactions.py, etc.) is heavily built on [Spotlight](https://github.com/maciejkula/spotlight). Thanks [Maciej Kula](https://github.com/maciejkula) for his great work.
