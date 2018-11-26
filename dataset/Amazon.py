from dataset.Dataset import DataSet


# Amazon review dataset
class Electronics(DataSet):
    def __init__(self):
        self.dir_path = './dataset/data/amazon/Electronics/'
        self.user_record_file = 'Electronics_user_records.pkl'
        self.user_mapping_file = 'Electronics_user_mapping.pkl'
        self.item_mapping_file = 'Electronics_item_mapping.pkl'
        self.item_content_file = 'word_counts.txt'
        self.item_relation_file = 'item_relation.pkl'

        # data structures used in the model
        self.num_users = 37204
        self.num_items = 13881
        self.vocab_size = 10104

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, seed=0):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)

        self.num_users = len(user_mapping)
        self.num_items = len(item_mapping)

        inner_data_records, user_inverse_mapping, item_inverse_mapping = \
            self.convert_to_inner_index(user_records, user_mapping, item_mapping)

        train_set, test_set = self.split_data_randomly(inner_data_records, seed)

        train_matrix = self.generate_rating_matrix(train_set, self.num_users, self.num_items)
        # train_matrix = self.fill_zero_col(train_matrix)
        item_content_matrix = self.load_item_content(self.dir_path + self.item_content_file, self.vocab_size)
        item_relation_matrix = self.load_pickle(self.dir_path + self.item_relation_file)

        return train_matrix, train_set, test_set, item_content_matrix, item_relation_matrix


class Books(DataSet):
    def __init__(self):
        self.dir_path = './dataset/data/amazon/Books/'
        self.user_record_file = 'Books_user_records.pkl'
        self.user_mapping_file = 'Books_user_mapping.pkl'
        self.item_mapping_file = 'Books_item_mapping.pkl'
        self.item_content_file = 'word_counts.txt'
        self.item_relation_file = 'item_relation.pkl'
        self.item_word_seq_file = 'review_word_sequence.pkl'

        # data structures used in the model
        self.num_users = 65476
        self.num_items = 41264
        self.vocab_size = 27584

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, seed=0):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)
        word_seq = self.load_pickle(self.dir_path + self.item_word_seq_file)

        self.num_users = len(user_mapping)
        self.num_items = len(item_mapping)

        inner_data_records, user_inverse_mapping, item_inverse_mapping = \
            self.convert_to_inner_index(user_records, user_mapping, item_mapping)

        train_set, test_set = self.split_data_randomly(inner_data_records, seed)

        train_matrix = self.generate_rating_matrix(train_set, self.num_users, self.num_items)
        item_content_matrix = self.load_item_content(self.dir_path + self.item_content_file, self.vocab_size)
        item_relation_matrix = self.load_pickle(self.dir_path + self.item_relation_file)

        return train_matrix, train_set, test_set, item_content_matrix, item_relation_matrix, word_seq


class CDs(DataSet):
    def __init__(self):
        self.dir_path = './dataset/data/amazon/CDs/'
        self.user_record_file = 'CDs_user_records.pkl'
        self.user_mapping_file = 'CDs_user_mapping.pkl'
        self.item_mapping_file = 'CDs_item_mapping.pkl'
        self.item_content_file = 'word_counts.txt'
        self.item_relation_file = 'item_relation.pkl'
        self.item_word_seq_file = 'review_word_sequence.pkl'

        # data structures used in the model
        self.num_users = 24934
        self.num_items = 24634
        self.vocab_size = 24341

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, seed=0):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)
        word_seq = self.load_pickle(self.dir_path + self.item_word_seq_file)

        self.num_users = len(user_mapping)
        self.num_items = len(item_mapping)

        inner_data_records, user_inverse_mapping, item_inverse_mapping = \
            self.convert_to_inner_index(user_records, user_mapping, item_mapping)

        train_set, test_set = self.split_data_randomly(inner_data_records, seed)

        train_matrix = self.generate_rating_matrix(train_set, self.num_users, self.num_items)
        item_content_matrix = self.load_item_content(self.dir_path + self.item_content_file, self.vocab_size)
        item_relation_matrix = self.load_pickle(self.dir_path + self.item_relation_file)

        return train_matrix, train_set, test_set, item_content_matrix, item_relation_matrix, word_seq


if __name__ == '__main__':
    data_set = CDs()
    train_matrix, train_set, test_set, item_content_matrix, item_relation_matrix, word_seq = data_set.generate_dataset()
    print(word_seq[-1])
    max_len = 0
    for word_list in word_seq:
        max_len = max(len(word_list), max_len)
    print(max_len)
    for i in range(item_content_matrix.shape[0]):
        if item_content_matrix.getrow(i).getnnz() == 0:
            print(i)