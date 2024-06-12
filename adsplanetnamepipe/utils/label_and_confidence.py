import os
import traceback
import time

import pandas as pd
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

try:
    import cPickle as pickle
except ImportError:
    import pickle

from adsplanetnamepipe.utils.common import EntityArgs


class LabelAndConfidence(object):

    units = [16, 8, 8, 1]
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    epoch = 100
    batch_size = 1

    training_file = '/label_and_confidence_files/training.csv'
    model_file = os.path.dirname(__file__) + '/label_and_confidence_files/model'

    max_build_retry = 5
    accuracy_threshold = 0.96

    def __init__(self, args: EntityArgs, train_mode: bool=False):
        """

        :param args:
        :param train_mode:
        """
        try:
            if not train_mode:
                self.load()
            else:
                self.train_and_save()
        except:
            self.model = None

        self.args = args
        self.confidence_format = '%.{}f'.format(config['PLANETARYNAMES_PIPELINE_FORMAT_SIGNIFICANT_DIGITS'])

    def forward(self, knowledge_graph_score, paper_relevance_score, local_llm_score):
        """

        :param knowledge_graph_score:
        :param paper_relevance_score:
        :param local_llm_score:
        :return: confidence and label
        """
        try:
            if not self.model:
                return '', -1

            scores = [knowledge_graph_score, paper_relevance_score, local_llm_score]
            prediction_score = self.model.predict([scores], verbose=0)[0][0].item()
            confidence = float(self.confidence_format % prediction_score)
            label = next((item['label'] for item in self.args.name_entity_labels if item['value'] == 1), None) if confidence >= 0.5 else \
                    next((item['label'] for item in self.args.name_entity_labels if item['value'] == 0), None)
            return label, confidence
        except Exception as e:
            logger.error(f"Exception in forward method of LabelAndConfidence: {str(e)}")
            return '', -1

    def train(self):
        """

        :return:
        """
        try:
            df = pd.read_csv(os.path.dirname(__file__) + self.training_file)
            properties = list(df.columns.values)
            properties.remove('label')
            X = df[properties]
            y = df['label']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            len_test_valid = len(y_test)
            x_val = X_train[:len_test_valid]
            partial_x_train = X_train[len_test_valid:]
            y_val = y_train[:len_test_valid]
            partial_y_train = y_train[len_test_valid:]

            width = X_train.shape[1]
            self.model = Sequential([
                layers.Flatten(input_shape=(width,)),
                layers.Dense(self.units[0], activation="relu"),
                layers.Dense(self.units[1], activation="relu"),
                layers.Dense(self.units[2], activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ])

            self.model.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=['accuracy'])

            # create an EarlyStopping callback
            early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5, mode='max', verbose=1)
            self.model.fit(partial_x_train,
                        partial_y_train,
                        epochs=self.epoch,
                        batch_size=self.batch_size,
                        validation_data=(x_val, y_val),
                        verbose=1,
                        callbacks=[early_stopping])

            self.test_loss, self.test_accuracy = self.model.evaluate(X_test, y_test)
            logger.debug("test accuracy = %.2f"%self.test_accuracy)
            return self.test_accuracy
        except Exception as e:
            logger.error(f"Exception in training method: {str(e)}")

    def train_and_save(self):
        """

        :return:
        """
        # to accept the training and save, the accuracy has to be above the threshold
        # if run out of number of tries to go above the threshold, then it is a failure
        for _ in range(self.max_build_retry):
            if self.train() > self.accuracy_threshold:
                break
        if self.model:
            self.save()

    def save(self):
        """

        :return:
        """
        try:
            keras.models.save_model(model=self.model, filepath=self.model_file)
        except Exception as e:
            logger.error(f"An error occurred while saving the model: {str(e)}")

    def load(self):
        """

        :return:
        """
        try:
            self.model = keras.models.load_model(self.model_file)
        except Exception as e:
            self.model = None
            logger.error(f"An error occurred while loading the model: {str(e)}")


def create_keras_model():  # pragma: no cover
    """
    create a keras model and save it to a pickle file
    this is not part of the production, it is run only when
    a new model need to be trained and saved

    :return:
    """
    try:
        start_time = time.time()
        keras_model = LabelAndConfidence(train_mode=True)
        if not keras_model.model:
            raise
        keras_model.save()
        logger.debug("keras models trained and saved in %s ms" % ((time.time() - start_time) * 1000))
        return keras_model
    except Exception as e:
        logger.error('Exception: %s' % (str(e)))
        logger.error(traceback.format_exc())
        return None
