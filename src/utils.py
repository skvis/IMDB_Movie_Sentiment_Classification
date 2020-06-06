import config
import io
import tensorflow as tf
import joblib


def save_weights(weights, reverse_word_index):

    out_v = io.open(f'{config.MODEL_PATH}/vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open(f'{config.MODEL_PATH}/meta.tsv', 'w', encoding='utf-8')
    for word_num in range(1, config.VOCAB_SIZE):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + '\n')
        out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
    out_v.close()
    out_m.close()


if __name__ == '__main__':
    load_model = tf.keras.models.load_model(f"{config.MODEL_PATH}my_model.h5")
    reverse_word_index = joblib.load(f"{config.MODEL_PATH}rev_word_ind.pkl")
    e = load_model.layers[0]
    weights = e.get_weights()[0]
    save_weights(weights, reverse_word_index)
