import numpy as np

data_split = (0.90, 0.02, 0.08)

class DataLoader(object):
    def __init__(self, input_file_name):
        self.raw_text = open(input_file_name).read().split()
        all_words = set(self.raw_text)
        self.vocabulary_size = len(all_words)
        self.dataset_size = len(self.raw_text)

        self.training_set_size = int(data_split[0] * self.dataset_size)
        self.validation_set_size = int(data_split[1] * self.dataset_size)
        self.test_set_size = int(data_split[2] * self.dataset_size)

        self.word_to_pos = dict(zip(all_words, range(self.vocabulary_size)))
        self.pos_to_word = dict(zip(range(self.vocabulary_size), all_words))

        tokens = [ self.word_to_pos[word] for word in self.raw_text]
        self.training_set = tokens[:self.training_set_size]
        self.validation_set = tokens[self.training_set_size:self.training_set_size + self.validation_set_size]
        print(len(self.validation_set))
        self.test_set = tokens[-self.test_set_size:]

    def new_training_batch(self, batch_size, seq_length):
        pos = np.random.random_integers(0,
            self.dataset_size - seq_length - 1, (batch_size))
        data = np.empty((batch_size, seq_length))
        labels = np.empty((batch_size, seq_length))
        for i in range(batch_size):
            for j in range(seq_length):
                word = self.raw_text[pos[i] + j]
                data[i, j] = self.word_to_pos[word]
                target_word = self.raw_text[pos[i] + j + 1]
                labels[i, j] = self.word_to_pos[target_word]
        return data, labels

    def _create_batch_from(self, data_set, seq_length):
        data_set_size = len(data_set)
        # Ensure we have one more item for next-word label
        elems_to_take = (data_set_size - 1) - ((data_set_size - 1) % seq_length)
        data = np.array(data_set[:elems_to_take])
        labels = np.array(data_set[1:elems_to_take + 1])
        return np.reshape(data, (-1, seq_length)), np.reshape(labels, (-1, seq_length))

    def validation_batch(self, seq_length):
        return self._create_batch_from(self.validation_set, seq_length)

    def testing_batch(self, seq_length):
        return self._create_batch_from(self.test_set, seq_length)
