from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import model
import config
import data_preprocess
import pickle
import joblib


def train_fn():

    rnn_model = model.rnn()
    # print(model.summary())

    rnn_model.compile(optimizer=Adam(lr=config.LEARNING_RATE),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    train_padded, train_labels, test_padded, test_labels = data_preprocess.tokenizer_sequences()

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=2)]

    history = rnn_model.fit(train_padded, train_labels,
                            validation_data=(test_padded, test_labels),
                            epochs=config.NUM_EPOCHS,
                            verbose=2,
                            callbacks=callbacks)

    rnn_model.save(f"{config.MODEL_PATH}my_model.h5")
    np.save(f'{config.MODEL_PATH}my_history.npy', history.history)
    joblib.dump(test_padded, f"{config.MODEL_PATH}test_padded.pkl")
    joblib.dump(test_labels, f"{config.MODEL_PATH}test_labels.pkl")


if __name__ == '__main__':
    train_fn()
