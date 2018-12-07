# The GATE model for Content-aware Recommendation
The implementation of the paper:

*Chen Ma, Peng Kang, Bin Wu, Qinglong Wang, and Xue Liu, "**Gated Attentive-Autoencoder for Content-Aware
Recommendation**", in the 12th ACM International Conference on Web Search and Data Mining (**WSDM 2019**)* 

Arxiv: TBA

**Please cite our paper if you use our code. Thanks!**

Author: Chen Ma (allenmc1230@gmail.com)


## Environments

- python 3.6
- PyTorch (version: 0.4.0)
- numpy (version: 1.15.0)
- scipy (version: 1.1.0)
- sklearn (version: 0.19.1)


## Dataset

In our experiments, the *citeulike-a* dataset is from http://www.wanghao.in/CDL.htm, the *movielens-20M* dataset is from https://grouplens.org/datasets/movielens/20m/, the *Amazon-CDs* and *Amazon-Books* datasets are from http://jmcauley.ucsd.edu/data/amazon/.


## Example to run the code

Data preprocessing:

The code for data preprocessing is put in the ```/preprocessing``` folder. ```Amazon_CDs.ipynb``` provides an example on how to transform the raw data into the ```.pickle``` files.

Train and evaluate the model (you are strongly recommended to run the program on a machine with GPU):

```
python run.py
```
