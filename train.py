import tensorflow as tf
import numpy as np
import math
from embed_rnn import EmbedRnn
from data_loader import DataLoader

tf.flags.DEFINE_integer("training_batch_size", 64, "Number of sequences in a batch")
tf.flags.DEFINE_integer("seq_length", 4, "Number of tokens (words) in a sequence")
tf.flags.DEFINE_integer("internal_state", 128, "Width of an RNN hidden state")
tf.flags.DEFINE_integer("embedding_size", 128, "Word embedding vector width")
tf.flags.DEFINE_string("input_file_name", 'input_text.txt', "Input file name")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("Loading data...")

data = DataLoader(FLAGS.input_file_name)
vocabulary_size = data.vocabulary_size
dataset_size = data.dataset_size
batch_size = FLAGS.training_batch_size
seq_length = FLAGS.seq_length
internal_state_size = FLAGS.internal_state
embedding_size = FLAGS.embedding_size

print("Vocabulary size: {}".format(vocabulary_size))
print("Dataset size: {}".format(dataset_size))

graph = tf.Graph()
with graph.as_default():
    cnn = EmbedRnn(seq_length,
                   internal_state_size,
                   vocabulary_size,
                   embedding_size)
    opt = tf.train.AdamOptimizer(0.01).minimize(cnn.loss)

num_steps = 5000
with tf.Session(graph=graph) as session:
    validation_data, validation_labels = data.validation_batch(seq_length)
    testing_data, testing_labels = data.testing_batch(seq_length)
    def training_step():
        batch_data, batch_labels = data.new_training_batch(batch_size, seq_length)
        feed_dict = {
            cnn.input_data: batch_data,
            cnn.target_labels: batch_labels
        }
        _, l = session.run([opt, cnn.loss], feed_dict=feed_dict)
        return l

    def validation_step():
        feed_dict = {
            cnn.input_data: validation_data,
            cnn.target_labels: validation_labels
        }
        return session.run([cnn.accuracy], feed_dict=feed_dict)

    def testing_step():
        feed_dict = {
            cnn.input_data: testing_data,
            cnn.target_labels: testing_labels
        }
        return session.run([cnn.accuracy], feed_dict=feed_dict)

    tf.initialize_all_variables().run()
    avg_loss = 0
    for i in range(num_steps):
        avg_loss += training_step()
        if (i + 1) % 100 == 0:
            print("Loss after {0} steps: {1}".format(i + 1, avg_loss / 100))
            print("Validation set accuracy: {}".format(validation_step()))
            avg_loss = 0

    def sampler(seed_words, num_tokens):
        text = seed_words
        seeds = [ data.word_to_pos[seed_word] for seed_word in seed_words ]
        for i in range(num_tokens):
            inputs = np.reshape(np.array(seeds), (1, len(seeds)))
            feed_dict = { cnn.input_data: inputs }
            word_probs = session.run([cnn.all_probabilities], feed_dict=feed_dict)
            next_word_probs = word_probs[-1][-1]
            next_word = np.random.choice(vocabulary_size, p=next_word_probs)
            seeds.pop(0)
            seeds.append(next_word)
            text.append(data.pos_to_word[next_word])
        return text

    print(" ".join(sampler("thou accents of woman".split(), 400)))
