# The GATE model for Content-aware Recommendation
The implementation of the paper:

*Chen Ma, Peng Kang, Bin Wu, Qinglong Wang, and Xue Liu, "**Gated Attentive-Autoencoder for Content-Aware
Recommendation**", in the 12th ACM International Conference on Web Search and Data Mining (**WSDM 2019**)* 

Arxiv: https://arxiv.org/abs/1812.02869

**Please cite our paper if you use our code. Thanks!**

Author: Chen Ma (allenmc1230@gmail.com)

**Bibtex**
```
@inproceedings{DBLP:conf/wsdm/MaKWWL19,
  author    = {Chen Ma and
               Peng Kang and
               Bin Wu and
               Qinglong Wang and
               Xue Liu},
  title     = {Gated Attentive-Autoencoder for Content-Aware Recommendation},
  booktitle = {{WSDM}},
  pages     = {519--527},
  publisher = {{ACM}},
  year      = {2019}
}
```

## Environments

- python 3.6
- PyTorch (version: 0.4.0)
- numpy (version: 1.15.0)
- scipy (version: 1.1.0)
- sklearn (version: 0.19.1)


## Dataset

In our experiments, the *citeulike-a* dataset is from http://www.wanghao.in/CDL.htm, the *movielens-20M* dataset is from https://grouplens.org/datasets/movielens/20m/, the *Amazon-CDs* and *Amazon-Books* datasets are from http://jmcauley.ucsd.edu/data/amazon/. (If you need the data after preprocessing, please send me an email).

The ```XXX_user_records.pkl``` file is a list of lists that stores the inner item id of each user, e.g., ```user_records[0]=[item_id0, item_id1, item_id2,...]```.

The ```XXX_user_mapping.pkl``` file is a list that maps the user inner id to its original id, e.g., ```user_mapping[0]=A2SUAM1J3GNN3B```.

The ```XXX_item_mapping.pkl``` file is similar to ```XXX_user_mapping.pkl```.

The ```item_relation.pkl``` file is a list of lists that stores the neighbors of each item, e.g., ```item_relation[0]=[item_id0, item_id1, item_id2,...]```.

The ```review_word_sequence.pkl``` file is a list of lists that stores the word sequence in the description of each item , e.g., ```review_word_sequence[0]=[word_id0, word_id1, word_id2,...]```. The word id is the same as the line number (start from 0) in the ```vocabulary.txt``` file.

## Example to run the code

Data preprocessing:

The code for data preprocessing is put in the ```/preprocessing``` folder. ```Amazon_CDs.ipynb``` provides an example on how to transform the raw data into the ```.pickle``` files that used in our program.

Train and evaluate the model (you are strongly recommended to run the program on a machine with GPU):

```
python run.py
```
