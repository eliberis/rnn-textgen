import tensorflow as tf
import numpy as np
import math

class EmbedRnn(object):
    def __init__(self, seq_length, internal_state_size,
                 vocabulary_size, embedding_size):
        self.input_data = tf.placeholder(tf.int32, shape=(None, seq_length), name="input_data")
        self.target_labels = tf.placeholder(tf.int32, shape=(None, seq_length), name="target_labels")

        with tf.name_scope("embedding"):
            emb = tf.Variable(
                initial_value=tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
                name="embedding_matrix")
            embedded_input = tf.nn.embedding_lookup(emb, self.input_data)

        with tf.name_scope("rnn"):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=internal_state_size,
                                           state_is_tuple=True)

            outputs, last_states = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=embedded_input,
                dtype=tf.float32)

            flat_outputs = tf.reshape(outputs, [-1, internal_state_size],
                                      name="flat_outputs")
            flat_labels = tf.reshape(self.target_labels, [-1, 1],
                                     name="flat_labels")

        with tf.name_scope("softmax"):
            W = tf.Variable(
                initial_value=
                    tf.truncated_normal(
                        [vocabulary_size, internal_state_size],
                        stddev=1.0 / math.sqrt(internal_state_size)),
                name="W")
            b = tf.Variable(initial_value=tf.zeros([vocabulary_size]),
                            name="b")
            examples_num = tf.shape(flat_outputs)[0]
            flat_biases = tf.tile(b, tf.expand_dims(examples_num, 0))
            bs = tf.reshape(flat_biases, [-1, examples_num])
            self.all_probabilities = tf.nn.softmax(
                tf.matmul(flat_outputs, W, transpose_b=True) + tf.transpose(bs),
                name="all_probabilities")

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(W, b, flat_outputs,
                    flat_labels, 64, vocabulary_size))

        with tf.name_scope("eval"):
            self.all_predictions = tf.cast(tf.argmax(self.all_probabilities, 1,
                                             name="predictions"), tf.int32)
            correct_predictions = tf.equal(self.all_predictions, flat_labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                                           name="accuracy")
