import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from tensorflow.keras.layers import Layer

from adsplanetnamepipe.utils.label_and_confidence import LabelAndConfidence
from adsplanetnamepipe.utils.common import EntityArgs


class TestKerasModel(unittest.TestCase):

    def setUp(self):
        """ Set up the config class and create an instance of LabelAndConfidence """

        self.args = EntityArgs(
            target="Mars",
            feature_type="Crater",
            feature_type_plural="Craters",
            feature_name="Rayleigh",
            context_ambiguous_feature_names=["asteroid", "main belt asteroid", "Moon", "Mars"],
            multi_token_containing_feature_names=["Rayleigh A", "Rayleigh B", "Rayleigh C", "Rayleigh D"],
            name_entity_labels=[{'label': 'planetary', 'value': 1}, {'label': 'non planetary', 'value': 0}],
            timestamp='2000-01-01',
            all_targets = ["Mars", "Mercury", "Moon", "Venus"]
        )
        self.label_and_confidence = LabelAndConfidence(self.args)

    def test_init(self):
        """ test the __init__ method """

        # Test case 1: train_mode=False and model file exists, so load
        with patch('tensorflow.keras.models.load_model') as mock_load_model:
            label_and_confidence = LabelAndConfidence(self.args, train_mode=False)
            mock_load_model.assert_called_once_with(label_and_confidence.model_file)
            self.assertIsNotNone(label_and_confidence.model)

        # Test case 2: train_mode=True, so call train and save
        with patch.object(LabelAndConfidence, 'train_and_save') as mock_train_and_save:
            LabelAndConfidence(self.args, train_mode=True)
            mock_train_and_save.assert_called_once()

    def test_init_fail(self):
        """ test the __init__ method when failing """

        # Test case 1: train_mode=False and an exception occurs during loading
        with patch('tensorflow.keras.models.load_model', side_effect=Exception('Loading failed')), \
             patch('adsplanetnamepipe.utils.label_and_confidence.logger.error') as mock_error_logger:
            label_and_confidence = LabelAndConfidence(self.args, train_mode=False)
            self.assertIsNone(label_and_confidence.model)
            mock_error_logger.assert_called_with("An error occurred while loading the model: Loading failed")

        # Test case 2: train_mode=True and an exception occurs during saving
        with patch.object(LabelAndConfidence, 'train', side_effect=[0.97]), \
             patch.object(LabelAndConfidence, 'model', create=True), \
             patch('tensorflow.keras.models.save_model', side_effect=Exception('Saving failed')), \
             patch('adsplanetnamepipe.utils.label_and_confidence.logger.error') as mock_error_logger:
            LabelAndConfidence(self.args, train_mode=True)
            mock_error_logger.assert_called_with("An error occurred while saving the model: Saving failed")

        # Test case 3: train_mode=True and an exception occurs during training
        with patch('adsplanetnamepipe.utils.label_and_confidence.Sequential', side_effect=Exception('Training failed')), \
             patch('adsplanetnamepipe.utils.label_and_confidence.logger.error') as mock_error_logger:
            label_and_confidence = LabelAndConfidence(self.args, train_mode=True)
            self.assertIsNone(label_and_confidence.model)
            mock_error_logger.assert_called_with("Exception in training method: Training failed")

    def test_forward(self):
        """ test forward method """

        self.assertEqual(('planetary', 0.9), self.label_and_confidence.forward(0.7, 0.8, 0.5))
        self.assertEqual(('planetary', 0.67), self.label_and_confidence.forward(0.2, 0.8, 0.5))
        self.assertEqual(('planetary', 0.58), self.label_and_confidence.forward(0.7, 0.2, 0.5))
        self.assertEqual(('non planetary', 0.34), self.label_and_confidence.forward(0.7, 0.8, 0.2))

    @patch('adsplanetnamepipe.utils.label_and_confidence.logger.error')
    def test_forward_fail(self, mock_error_logger):
        """ test forward method when failing """

        # Test case 1: getting exception
        with patch('tensorflow.keras.Model.predict', side_effect=Exception('Prediction error')):
            results = self.label_and_confidence.forward(0.5, 0.6, 0.7)
            self.assertEqual(results, ('', -1))
            mock_error_logger.assert_called_with('Exception in forward method of LabelAndConfidence: Prediction error')

        # Text case 2: model was not loaded
        self.label_and_confidence.model = None
        results = self.label_and_confidence.forward(0.7, 0.8, 0.5)
        self.assertEqual(results, ('', -1))

    def test_train(self):
        """ test train method """

        # create a MagicMock object to mimic the behavior of a Keras Layer
        mock_layer = MagicMock(spec=Layer)

        # patch('tensorflow.keras.Sequential') as mock_sequential, \
        # Mock the necessary dependencies
        with patch('pandas.read_csv') as mock_read_csv, \
                patch('adsplanetnamepipe.utils.label_and_confidence.train_test_split') as mock_train_test_split, \
                patch('adsplanetnamepipe.utils.label_and_confidence.Sequential') as mock_sequential, \
                patch('adsplanetnamepipe.utils.label_and_confidence.layers.Flatten', return_value=mock_layer) as mock_flatten, \
                patch('adsplanetnamepipe.utils.label_and_confidence.layers.Dense') as mock_dense, \
                patch('adsplanetnamepipe.utils.label_and_confidence.EarlyStopping') as mock_early_stopping:
            # set the return value of the mock_layer to another MagicMock
            # this ensures that when you use it, it behaves as expected
            mock_layer.return_value = MagicMock()

            mock_data = {
                'knowledge graph score': [0.2, 1.0, 1.0, 1.0, 1.0, 1.0],
                'paper relevence score': [0.8, 0.6, 0.6, 0.6, 0.8, 0.8],
                'local llm score': [0.9, 0.5, 0.0, 0.0, 0.8, 1],
                'label': [1, 1, 0, 0, 1, 1]
            }
            mock_read_csv.return_value = pd.DataFrame(mock_data)

            feature_columns = ['knowledge graph score', 'paper relevence score', 'local llm score']
            mock_train_test_split.return_value = (
                pd.DataFrame(mock_data)[feature_columns][:4],  # X_train
                pd.DataFrame(mock_data)[feature_columns][4:],  # X_test
                pd.Series(mock_data['label'][:4]),  # y_train
                pd.Series(mock_data['label'][4:])  # y_test
            )

            # mock the compile, fit, and evaluate methods
            mock_model = MagicMock()
            mock_sequential.return_value = mock_model
            mock_model.compile.return_value = None
            mock_model.fit.return_value = None
            mock_model.evaluate.return_value = (0.2, 0.8)  # Mock test loss and accuracy

            # mock the EarlyStopping callback
            mock_early_stopping.return_value = MagicMock()

            # instantiate to call the train method
            self.label_and_confidence.train()

            training_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')) + self.label_and_confidence.training_file

            # assert that the necessary methods were called
            mock_read_csv.assert_called_once_with(training_file)
            mock_train_test_split.assert_called_once()
            mock_sequential.assert_called_once()
            mock_flatten.assert_called_once()
            mock_early_stopping.assert_called_once()
            self.assertEqual(mock_dense.call_count, 4)

            # assert that compile, fit, and evaluate methods were called
            mock_model.compile.assert_called_once()
            mock_model.fit.assert_called_once()
            mock_model.evaluate.assert_called_once()

            self.assertIsInstance(self.label_and_confidence.test_accuracy, float)
            self.assertEqual(self.label_and_confidence.test_accuracy, 0.8)

    @patch('adsplanetnamepipe.utils.label_and_confidence.logger.error')
    def test_train_failuare(self, mock_error_logger):
        """ test train method when there is an exception """

        with patch('pandas.read_csv', side_effect=FileNotFoundError('File not found')):
            result = self.label_and_confidence.train()
            self.assertIsNone(result)
            mock_error_logger.assert_called_with("Exception in training method: File not found")

        with patch('pandas.read_csv', return_value=MagicMock()), \
             patch('adsplanetnamepipe.utils.label_and_confidence.train_test_split', side_effect=ValueError('Value error')):
            result = self.label_and_confidence.train()
            self.assertIsNone(result)
            mock_error_logger.assert_called_with("Exception in training method: list.remove(x): x not in list")

        with patch('pandas.read_csv', return_value=MagicMock(columns=['column1', 'column2'])), \
             patch('adsplanetnamepipe.utils.label_and_confidence.logger.error') as mock_error_logger:
            result = self.label_and_confidence.train()
            self.assertIsNone(result)
            mock_error_logger.assert_called_with("Exception in training method: 'list' object has no attribute 'values'")

        with patch('pandas.read_csv', side_effect=Exception('Unexpected error')), \
             patch('adsplanetnamepipe.utils.label_and_confidence.logger.error') as mock_error_logger:
            result = self.label_and_confidence.train()
            self.assertIsNone(result)
            mock_error_logger.assert_called_with("Exception in training method: Unexpected error")


if __name__ == '__main__':
    unittest.main()
