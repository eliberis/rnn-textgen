from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import np_utils
import numpy as np

INPUT_FILE="excuses_clean.txt"

def batched_input(seq_len):
    with open(INPUT_FILE) as f:
        excuses = f.readlines()
    all_excuses = " ".join(excuses)
    words = set(all_excuses.split())

    idx2word = [w for w in words]
    word2idx = {word : idx for idx, word in enumerate(words)}

    X = []
    y = []
    for excuse in excuses:
        excuse = [word2idx[word] for word in excuse.split()]
        X.extend([excuse[(i - seq_len):i] for i in range(seq_len, len(excuse))])
        y.extend([excuse[i] for i in range(seq_len, len(excuse))])

    X = np.array(X, dtype='int32')
    y = np_utils.to_categorical(y, nb_classes=len(words))

    return X, y, idx2word, word2idx

def get_model(seq_len, num_classes):
    model = Sequential()
    model.add(Embedding(num_classes, 64, input_length=seq_len))
    # model.add(LSTM(256, dropout_U=0.1, return_sequences=True))
    model.add(LSTM(128, dropout_W=0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

def do_training():
    seq_len = 5
    X, y, idx2word, word2idx = batched_input(seq_len)
    num_classes = len(idx2word)

    model = get_model(seq_len, num_classes)
    model.fit(X, y, batch_size=64, nb_epoch=50, validation_split=0.1)
    # model.save_weights("wordmodel-1000.h5")

    seeds = ["i am sorry i can", "dear matthew i need to"]
    for seed in seeds:
        seedidx = [word2idx[w] for w in seed.split()]
        print("Excuse 1", sample_text(model, idx2word, seedidx.copy(), 50))
        print("Excuse 2", sample_text(model, idx2word, seedidx.copy(), 50))
        print("Excuse 3", sample_text(model, idx2word, seedidx.copy(), 50))
        print("Excuse 4", sample_text(model, idx2word, seedidx.copy(), 50))

def sample_text(model, idx2word, seed, text_len):
    gen_text = seed.copy()
    while len(gen_text) < text_len:
        seed_vec = np.expand_dims(np.array(seed, dtype='int32'), axis=0)
        probs = model.predict(seed_vec, verbose=0)[0]

        best_idx = sample(probs, t=5.0)
        gen_text.append(best_idx)

        del seed[0]
        seed.append(best_idx)

    return " ".join([idx2word[i] for i in gen_text])

# Sample a probability distribution, with a given temperature parameter.
# Blatantly copied from (thanks, Petar!):
# https://github.com/PetarV-/a-trip-down-lstm-lane/blob/master/char_lstm.py
def sample(p, t=1.0):
    p = np.asarray(p).astype('float64')
    p = np.log(p) / t
    ex_p = np.exp(p)
    p = ex_p / np.sum(ex_p)
    p[p < 0] = 0
    p = p / np.sum(p)
    p = np.random.multinomial(1, p)
    return np.argmax(p)

do_training()
