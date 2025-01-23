import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)


import unittest

from unittest.mock import MagicMock, patch

from adsplanetnamepipe.utils.paper_relevance import PaperRelevance
from adsplanetnamepipe.utils.common import EntityArgs

from adsplanetnamepipe.tests.unittests.stubdata import solrdata


class TestPaperRelevance(unittest.TestCase):

    """
    Tests the paper relevance module
    """

    def setUp(self):
        """ Set up the config class and create an instance of PaperRelevance """

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
        self.paper_relevance = PaperRelevance(self.args)

    def test_forward(self):
        """ test forward method """

        text = f"{' '.join(solrdata.doc_1['title'])} {solrdata.doc_1['abstract']} {solrdata.doc_1['body']}"
        bibstem = 'ApJ'
        databases = ['astronomy']
        astronomy_main_journals = ['ApJ', 'MNRAS']
        len_existing_wikidata = 5

        score = self.paper_relevance.forward(text, bibstem, databases, astronomy_main_journals, len_existing_wikidata)
        self.assertEqual(score, 1.0)


if __name__ == '__main__':
    unittest.main()
