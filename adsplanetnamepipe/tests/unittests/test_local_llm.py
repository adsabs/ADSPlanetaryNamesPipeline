import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)


import unittest
from unittest.mock import MagicMock, patch

from adsplanetnamepipe.utils.local_llm import LocalLLM
from adsplanetnamepipe.utils.common import EntityArgs

from adsplanetnamepipe.tests.unittests.stubdata import solrdata
from adsplanetnamepipe.tests.unittests.stubdata import excerpts


class TestLocalLLM(unittest.TestCase):

    """
    Tests the local llm module
    """

    def setUp(self):
        """ Set up the config class and create an instance of LocalLLM """

        self.args = EntityArgs(
            target="Mars",
            feature_type = "Crater",
            feature_type_plural = "Craters",
            feature_name = "Rayleigh",
            context_ambiguous_feature_names = ["asteroid", "main belt asteroid", "Moon", "Mars"],
            multi_token_containing_feature_names = ["Rayleigh A", "Rayleigh B", "Rayleigh C", "Rayleigh D"],
            name_entity_labels = [{'label': 'planetary', 'value': 1}, {'label': 'non planetary', 'value': 0}],
            timestamp='2000-01-01',
            all_targets = ["Mars", "Mercury", "Moon", "Venus"]
        )
        self.local_llm = LocalLLM(self.args)

    @patch('requests.post')
    def test_forward(self, mock_post):
        """ test forward method """

        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': '0.8'}
        mock_post.return_value = mock_response

        result = self.local_llm.forward(solrdata.doc_1['title'], solrdata.doc_1['abstract'], excerpts.doc_1_excerpts[9]['excerpt'])
        self.assertEqual(result, 0.8)

    @patch('requests.post')
    def test_forward_error(self, mock_post):
        """ test when the response is not numeric as expected """

        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'Invalid'}
        mock_post.return_value = mock_response

        result = self.local_llm.forward(solrdata.doc_1['title'], solrdata.doc_1['abstract'], excerpts.doc_1_excerpts[9]['excerpt'])

        self.assertEqual(result, 0)

    @patch('requests.post')
    def test_forward_with_error_response(self, mock_post):
        # test when the response from the API is 500

        mock_response = unittest.mock.Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = self.local_llm.forward(solrdata.doc_1['title'], solrdata.doc_1['abstract'], excerpts.doc_1_excerpts[9]['excerpt'])

        self.assertEqual(result, 0)

    def test_forward_with_no_abstract(self):
        """ test when no abstract is provided """

        result = self.local_llm.forward(solrdata.doc_1['title'], None, excerpts.doc_1_excerpts[9]['excerpt'])

        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
