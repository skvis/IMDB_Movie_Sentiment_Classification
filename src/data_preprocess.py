import config
import dataset
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib


def data_process():
    imdb, _ = dataset.load_dataset()

    train_data, test_data = imdb['train'], imdb['test']

    train_sentences, train_labels = [], []
    test_sentences, test_labels = [], []

    for sent, lab in train_data:
        train_sentences.append(str(sent.numpy()))
        train_labels.append(lab.numpy())

    for sent, lab in test_data:
        test_sentences.append(str(sent.numpy()))
        test_labels.append(lab.numpy())

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_sentences, train_labels, test_sentences, test_labels


def tokenizer_sequences():
    train_sentences, train_labels, test_sentences, test_labels = data_process()

    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE,
                          oov_token=config.OOV_TOKEN)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences,
                                 maxlen=config.MAX_LENGTH,
                                 padding=config.PAD_TYPE,
                                 truncating=config.TRUNC_TYPE)

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_sequences,
                                maxlen=config.MAX_LENGTH,
                                padding=config.PAD_TYPE,
                                truncating=config.TRUNC_TYPE)

    joblib.dump(word_index, f"{config.MODEL_PATH}word_ind.pkl")
    return train_padded, train_labels, test_padded, test_labels


def decode_review(text):
    reverse_word_index = joblib.load(f"{config.MODEL_PATH}rev_word_ind.pkl")
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


if __name__ == '__main__':
    tr_p, _, _, _ = tokenizer_sequences()

    word_index = joblib.load(f"{config.MODEL_PATH}word_ind.pkl")
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    joblib.dump(reverse_word_index, f"{config.MODEL_PATH}rev_word_ind.pkl")
    print(decode_review(tr_p[2]))
