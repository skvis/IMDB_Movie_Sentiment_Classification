import tensorflow_datasets as tfds
import config


def load_dataset():
    imdb, info = tfds.load('imdb_reviews',
                           data_dir=config.DATA_PATH,
                           with_info=True,
                           as_supervised=True)
    return imdb, info


if __name__ == '__main__':
    load_dataset()
