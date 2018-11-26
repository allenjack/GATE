import torch
from torch.autograd import Variable
import torch.nn.functional as F

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


class GATE(torch.nn.Module):
    def __init__(self, num_users, num_items, num_words, H, drop_rate=0.5, att_dim=20):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(GATE, self).__init__()
        self.H = H[1]
        self.drop_rate = drop_rate
        if torch.cuda.is_available():
            self.item_gated_embedding = torch.nn.Embedding(
                num_embeddings=num_items + 1,
                embedding_dim=self.H,
                padding_idx=num_items  # the last item index is the padded index
            ).cuda()
            self.item_gated_embedding.weight.requires_grad = False

            self.word_embedding = torch.nn.Embedding(
                num_embeddings=num_words + 1,
                embedding_dim=self.H,
                padding_idx=num_words  # the last word index is the padded index
            ).cuda()

            self.aggregation = torch.nn.Linear(att_dim, 1).cuda()
            self.att1 = torch.nn.Linear(self.H, self.H).cuda()
            self.att2 = torch.nn.Linear(self.H, att_dim).cuda()

            self.linear1 = torch.nn.Linear(num_users, H[0]).cuda()
            self.linear2 = torch.nn.Linear(H[0], H[1]).cuda()
            self.linear3 = torch.nn.Linear(H[1], H[2]).cuda()
            self.linear4 = torch.nn.Linear(H[2], num_users).cuda()

        else:
            self.item_gated_embedding = torch.nn.Embedding(
                num_embeddings=num_items + 1,
                embedding_dim=self.H,
                padding_idx=num_items  # the last item index is the padded index
            )
            self.item_gated_embedding.weight.requires_grad = False

            self.word_embedding = torch.nn.Embedding(
                num_embeddings=num_words + 1,
                embedding_dim=self.H,
                padding_idx=num_words  # the last word index is the padded index
            )

            self.aggregation = torch.nn.Linear(att_dim, 1)
            self.att1 = torch.nn.Linear(self.H, self.H)
            self.att2 = torch.nn.Linear(self.H, att_dim)

            self.linear1 = torch.nn.Linear(num_users, H[0])
            self.linear2 = torch.nn.Linear(H[0], H[1])
            self.linear3 = torch.nn.Linear(H[1], H[2])
            self.linear4 = torch.nn.Linear(H[2], num_users)

        self.neighbor_attention = Variable(torch.zeros(self.H, self.H).type(T.FloatTensor), requires_grad=True)
        self.neighbor_attention = torch.nn.init.xavier_uniform_(self.neighbor_attention)

        # G = sigmoid(gate_matrix1 \dot item_embedding + gate_matrix2 \dot item_context_embedding + bias)
        self.gate_matrix1 = Variable(torch.zeros(self.H, self.H).type(T.FloatTensor), requires_grad=True)
        self.gate_matrix2 = Variable(torch.zeros(self.H, self.H).type(T.FloatTensor), requires_grad=True)
        self.gate_matrix1 = torch.nn.init.xavier_uniform_(self.gate_matrix1)
        self.gate_matrix2 = torch.nn.init.xavier_uniform_(self.gate_matrix2)
        self.gate_bias = Variable(torch.zeros(1, self.H).type(T.FloatTensor), requires_grad=True)
        self.gate_bias = torch.nn.init.xavier_uniform_(self.gate_bias)

    def forward(self, batch_item_index, batch_x, batch_word_seq, batch_neighbor_index):
        z_1 = F.tanh(self.linear1(batch_x))
        # z_1 = F.dropout(z_1, self.drop_rate)

        z_rating = F.tanh(self.linear2(z_1))
        z_content = self.get_content_z(batch_word_seq)

        gate = F.sigmoid(z_rating.mm(self.gate_matrix1) + z_content.mm(self.gate_matrix2) + self.gate_bias)
        gated_embedding = gate * z_rating + (1 - gate) * z_content

        # save the embedding for direct lookup
        self.item_gated_embedding.weight[batch_item_index] = gated_embedding.data

        gated_neighbor_embedding = self.item_gated_embedding(batch_neighbor_index)

        # aug_gated_embedding: [256, 1, 50]
        aug_gated_embedding = torch.unsqueeze(gated_embedding, 1)
        score = torch.matmul(aug_gated_embedding, torch.unsqueeze(self.neighbor_attention, 0))
        # score: [256, 1, 480]
        score = torch.bmm(score, gated_neighbor_embedding.permute(0, 2, 1))

        # make the 0 in score, which will make a difference in softmax
        score = torch.where(score == 0, T.FloatTensor([float('-inf')]), score)
        score = F.softmax(score, dim=2)
        # if the vectors all are '-inf', softmax will generate 'nan', so replace with 0
        score = torch.where(score != score, T.FloatTensor([0]), score)
        gated_neighbor_embedding = torch.bmm(score, gated_neighbor_embedding)
        gated_neighbor_embedding = torch.squeeze(gated_neighbor_embedding, 1)

        # gated_embedding = F.dropout(gated_embedding, self.drop_rate)
        # gated_neighbor_embedding = F.dropout(gated_neighbor_embedding, self.drop_rate)

        z_3 = F.tanh(self.linear3(gated_embedding))
        # z_3 = F.dropout(z_3, self.drop_rate)
        z_3_neighbor = F.tanh(self.linear3(gated_neighbor_embedding))
        # z_3_neighbor = F.dropout(z_3_neighbor, self.drop_rate)

        y_pred = F.sigmoid(self.linear4(z_3) + z_3_neighbor.mm(self.linear4.weight.t()))

        return y_pred

    def get_content_z(self, batch_word_seq):
        # [batch_size, num_word, hidden_dim], e.g., [256, 300, 100]
        batch_word_embedding = self.word_embedding(batch_word_seq)
        score = F.tanh(self.att1(batch_word_embedding))
        score = F.tanh(self.att2(score))
        # score dimension: [256, 300, 20]
        score = F.softmax(score, dim=1)
        # permute to make the matrix as [256, 50, 176]
        matrix_z = torch.bmm(batch_word_embedding.permute(0, 2, 1), score)
        linear_z = self.aggregation(matrix_z)
        linear_z = torch.squeeze(linear_z, 2)
        z = F.tanh(linear_z)

        return z
