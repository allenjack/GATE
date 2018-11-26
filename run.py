import datetime
import logging
import numpy as np
from argparse import ArgumentParser

import torch
from torch.autograd import Variable

from eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from model.model import GATE

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="double AE with word and neighbor-attention")
    parser.add_argument('-e', '--epoch', type=int, default=150, help='number of epochs for GAT')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='batch size for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('-att', '--num_attention', type=int, default=20, help='the number of dimension of attention')
    parser.add_argument('--inner_layers', nargs='+', type=int, default=[100, 50, 100], help='the number of latent factors')
    parser.add_argument('--rating_weight', type=int, default=20, help='the weight of the rating entry')
    parser.add_argument('--loss', type=str, default='mse', help='the weight of the word entry')
    parser.add_argument('-dr', '--dropout_rate', type=float, default=0.5, help='the dropout probability')
    parser.add_argument('-seed', type=int, default=0, help='random state to split the data')

    return parser.parse_args()


def get_mini_batch(batch_item_index, train_matrix, rating_weight_matrix, item_word_index, item_neighbor_index):
    return train_matrix[batch_item_index].toarray(), rating_weight_matrix[batch_item_index].toarray(), \
           item_word_index[batch_item_index], item_neighbor_index[batch_item_index]


def pad_data(matrix):
    """
    pad data to fit mini-batch training
    :param matrix:
    :return:
    """
    padded_index = matrix.shape[1]
    data, data_len = [], []
    for i in range(matrix.shape[0]):
        seq = matrix.getrow(i).indices
        data.append(seq)
        data_len.append(len(seq))

    longest_len = max(data_len)
    padded_data = np.ones((matrix.shape[0], longest_len)) * padded_index

    for i, seq in enumerate(data):
        padded_data[i, 0:len(seq)] = seq

    return padded_data


def pad_list(lists, cut_off_len=300):
    padded_index = max(max(lists))
    seq_len = []
    count = 0
    for data in lists:
        seq_len.append(len(data))
        if len(data) < cut_off_len:
            count += 1

    logger.debug('content coverage:{}'.format(count / len(seq_len)))

    seq_len = np.asarray(seq_len)
    seq_len[seq_len > cut_off_len] = cut_off_len
    longest_len = max(seq_len)
    padded_data = np.ones((len(lists), longest_len)) * padded_index

    for i, seq in enumerate(lists):
        padded_data[i, 0:seq_len[i]] = seq[:seq_len[i]]

    return padded_data, seq_len


def evaluate_model(train_matrix, test_set, item_word_seq, item_neighbor_index, GAT, batch_size):
    num_items, num_users = train_matrix.shape
    num_batches = int(num_items / batch_size) + 1
    item_indexes = np.arange(num_items)
    pred_matrix = None

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_items:
                end = num_items
            else:
                break

        batch_item_index = item_indexes[start:end]

        # get mini-batch data
        batch_x = train_matrix[batch_item_index].toarray()
        batch_word_seq = item_word_seq[batch_item_index]
        batch_neighbor_index = item_neighbor_index[batch_item_index]

        batch_item_index = Variable(torch.from_numpy(batch_item_index).type(T.LongTensor), requires_grad=False)
        batch_word_seq = Variable(torch.from_numpy(batch_word_seq).type(T.LongTensor), requires_grad=False)
        batch_neighbor_index = Variable(torch.from_numpy(batch_neighbor_index).type(T.LongTensor), requires_grad=False)
        batch_x = Variable(torch.from_numpy(batch_x.astype(np.float32)).type(T.FloatTensor), requires_grad=False)

        # Forward pass: Compute predicted y by passing x to the model
        rating_pred = GAT(batch_item_index, batch_x, batch_word_seq, batch_neighbor_index)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        if batchID == 0:
            pred_matrix = rating_pred.copy()
        else:
            pred_matrix = np.append(pred_matrix, rating_pred, axis=0)

    topk = 50
    pred_matrix[train_matrix.nonzero()] = 0
    pred_matrix = pred_matrix.transpose()
    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
    ind = np.argpartition(pred_matrix, -topk)
    ind = ind[:, -topk:]
    arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
    pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]

    precision, recall, MAP, ndcg = [], [], [], []
    for k in [5, 10, 15, 20, 30, 40, 50]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(ndcg_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg


def train_model(train_matrix, item_matrix, item_neighbor_matrix, word_seq, test_set, args):
    num_users, num_items = train_matrix.shape
    vocab_size = item_matrix.shape[1]
    train_matrix = train_matrix.transpose()

    rating_weight_matrix = train_matrix.copy()
    rating_weight_matrix[rating_weight_matrix > 0] = args.rating_weight
    train_matrix[train_matrix > 0] = 1.0

    # pad data for mini-batch
    # item_word_index = pad_data(item_matrix).astype(np.int32)
    item_neighbor_index = pad_data(item_neighbor_matrix).astype(np.int32)
    item_word_seq, word_seq_len = pad_list(word_seq)

    logger.debug(str(train_matrix.shape) + ',' + str(item_matrix.shape) + ',' + str(item_neighbor_matrix.shape))

    batch_size = args.batch_size

    dtype = T.FloatTensor

    # Construct our model by instantiating the class defined above
    GAT = GATE(num_users, num_items, vocab_size, args.inner_layers, args.dropout_rate, args.num_attention)

    if torch.cuda.is_available():
        GAT.cuda()

    if args.loss == 'mse':
        criterion = torch.nn.MSELoss(size_average=False, reduce=False)
    elif args.loss == 'bce':
        criterion = torch.nn.BCELoss(size_average=False, reduce=False)

    GAT_optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, GAT.parameters())),
                                     lr=args.learning_rate, weight_decay=args.weight_decay)

    item_indexes = np.arange(num_items)
    num_batches = int(num_items / batch_size) + 1
    GAT.train()
    for t in range(args.epoch):
        logger.debug("epoch:{}".format(t))
        np.random.shuffle(item_indexes)
        avg_cost = 0.
        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < num_items:
                    end = num_items
                else:
                    break

            batch_item_index = item_indexes[start:end]

            # get mini-batch data
            batch_x, batch_x_weight, batch_word_seq, batch_neighbor_index = \
                get_mini_batch(batch_item_index, train_matrix, rating_weight_matrix, item_word_seq, item_neighbor_index)
            batch_x_weight += 1

            batch_item_index = Variable(torch.from_numpy(batch_item_index).type(T.LongTensor), requires_grad=False)
            batch_word_seq = Variable(torch.from_numpy(batch_word_seq).type(T.LongTensor), requires_grad=False)
            batch_neighbor_index = Variable(torch.from_numpy(batch_neighbor_index).type(T.LongTensor),requires_grad=False)
            batch_x = Variable(torch.from_numpy(batch_x.astype(np.float32)).type(dtype), requires_grad=False)
            batch_x_weight = Variable(torch.from_numpy(batch_x_weight).type(dtype), requires_grad=False)

            # Forward pass: Compute predicted y by passing x to the model
            rating_pred = GAT(batch_item_index, batch_x, batch_word_seq, batch_neighbor_index)

            # Compute and print loss
            loss = (batch_x_weight * criterion(rating_pred, batch_x)).sum() / batch_size
            logger.debug('batch_id:{}, loss:{}'.format(batchID, loss.data))

            # Zero gradients, perform a backward pass, and update the weights.
            GAT_optimizer.zero_grad()
            loss.backward()
            GAT_optimizer.step()

            avg_cost += loss / num_items * batch_size
        logger.debug("Avg loss:{}".format(avg_cost))

        if t % 10 == 0 and t > 0:
            precision, recall, MAP, ndcg = evaluate_model(train_matrix, test_set, item_word_seq, item_neighbor_index, GAT, batch_size)
            logger.info('epoch:{}'.format(t))
            logger.info(', '.join(str(e) for e in precision))
            logger.info(', '.join(str(e) for e in recall))
            logger.info(', '.join(str(e) for e in MAP))
            logger.info(', '.join(str(e) for e in ndcg))

    GAT.eval()
    precision, recall, MAP, ndcg = evaluate_model(train_matrix, test_set, item_word_seq, item_neighbor_index, GAT, batch_size)
    logger.info('epoch:{}'.format(args.epoch))
    logger.info(', '.join(str(e) for e in precision))
    logger.info(', '.join(str(e) for e in recall))
    logger.info(', '.join(str(e) for e in MAP))
    logger.info(', '.join(str(e) for e in ndcg))
    logger.info('Parameters:')
    for arg, value in sorted(vars(args).items()):
        logger.info("%s: %r", arg, value)
    logger.info('\n')


def main():
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    args = parse_args()
    from dataset import Amazon
    train_matrix, train_set, test_set, item_content_matrix, item_relation_matrix, word_seq = Amazon.CDs().generate_dataset(args.seed)
    train_model(train_matrix, item_content_matrix, item_relation_matrix, word_seq, test_set, args)


if __name__ == '__main__':
    main()

