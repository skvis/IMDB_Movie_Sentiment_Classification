import config
import tensorflow as tf


def rnn():
    model = tf.keras.Sequential([
                tf.keras.layers.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM, input_length=config.MAX_LENGTH),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation='sigmoid')])

    return model
