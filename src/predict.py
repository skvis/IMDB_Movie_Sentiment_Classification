import config
import joblib
import tensorflow as tf


def evaluate(load_model, test_padded, test_labels):
    print(load_model.evaluate(test_padded, test_labels))


if __name__ == '__main__':
    test_padded = joblib.load(f"{config.MODEL_PATH}test_padded.pkl")
    test_labels = joblib.load(f"{config.MODEL_PATH}test_labels.pkl")
    load_model = tf.keras.models.load_model(f"{config.MODEL_PATH}my_model.h5")
    evaluate(load_model, test_padded, test_labels)
