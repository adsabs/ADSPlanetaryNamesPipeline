import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)


import unittest
from unittest.mock import MagicMock, patch

from typing import List, Tuple

from adsplanetnamepipe.identify import IdentifyPlanetaryEntities
from adsplanetnamepipe.models import NamedEntity, NamedEntityHistory
from adsplanetnamepipe.utils.common import EntityArgs

from adsplanetnamepipe.tests.unittests.stubdata import solrdata
from adsplanetnamepipe.tests.unittests.stubdata import excerpts


class TestCollectKnowldegeBase(unittest.TestCase):

    """
    Tests the collect module
    """

    def setUp(self):
        """ Set up the config class and create an instance of IdentifyPlanetaryEntities """

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
        self.identify_planetary_entities = IdentifyPlanetaryEntities(self.args, [[]], [[]])

    def test_get_knowledge_graph_score(self):
        """ test get_knowledge_graph_score method """

        self.identify_planetary_entities.knowledge_graph_positive.forward = MagicMock(return_value=0.7)
        self.identify_planetary_entities.knowledge_graph_negative.forward = MagicMock(return_value=0.3)
        score = self.identify_planetary_entities.get_knowledge_graph_score([])
        self.assertEqual(score, 0.7)

    def test_get_paper_relevance_score(self):
        """ test get_paper_relevance_score method """

        self.identify_planetary_entities.paper_relevance.forward = MagicMock(return_value=0.8)
        score = self.identify_planetary_entities.get_paper_relevance_score(solrdata.doc_1)
        self.assertEqual(score, 0.8)

    def test_get_local_llm_score(self):
        """ test get_local_llm_score """

        self.identify_planetary_entities.local_llm.forward = MagicMock(return_value=0.7)
        score = self.identify_planetary_entities.get_local_llm_score(solrdata.doc_1, excerpts.doc_1_excerpts[0]['excerpt'])
        self.assertEqual(score, 0.7)

    def test_identify(self):
        """ test identify method """

        keywords_forward = ['ripple', 'mars', 'discovery', 'eolian', 'meridiani planum', 'crater', 'formed', 'evidence',
                            'past', 'dune', 'bed']
        special_keywords_forward = ['ejecta', 'floors']

        self.identify_planetary_entities.search_retrieval.identify_terms_query = MagicMock(return_value=[solrdata.doc_1])
        self.identify_planetary_entities.match_excerpt.forward = MagicMock(return_value=(True, [excerpts.doc_1_excerpts[0]['excerpt']]))
        self.identify_planetary_entities.extract_keywords.forward = MagicMock(return_value=keywords_forward)
        self.identify_planetary_entities.extract_keywords.forward_special = MagicMock(return_value=special_keywords_forward)
        self.identify_planetary_entities.knowledge_graph_positive.forward = MagicMock(return_value=0.7)
        self.identify_planetary_entities.knowledge_graph_negative.forward = MagicMock(return_value=0.3)

        self.identify_planetary_entities.get_knowledge_graph_score = MagicMock(return_value=0.7)
        self.identify_planetary_entities.get_paper_relevance_score = MagicMock(return_value=0.8)
        self.identify_planetary_entities.get_local_llm_score = MagicMock(return_value=0.7)

        result = self.identify_planetary_entities.identify()

        # Assert the expected behavior
        self.assertEqual(len(result), 1)
        history_entry, identified_doc = result[0]
        self.assertIsInstance(history_entry, NamedEntityHistory)
        self.assertEqual(history_entry.feature_name_entity, 'Rayleigh')
        self.assertEqual(history_entry.feature_type_entity, 'Crater')
        self.assertEqual(history_entry.target_entity, 'Mars')
        self.assertEqual(len(identified_doc), 1)
        self.assertIsInstance(identified_doc[0], NamedEntity)
        self.assertEqual(identified_doc[0].keywords_item_id, 1)
        self.assertEqual(identified_doc[0].excerpt, excerpts.doc_1_excerpts[0]['excerpt'])
        self.assertEqual(identified_doc[0].keywords, keywords_forward)
        self.assertEqual(identified_doc[0].special_keywords, special_keywords_forward)


    def test_identify_no_special_keywords(self):
        """ test identify and verify that currently nasa concept is not operational """

        keywords_forward = ['ripple', 'mars', 'discovery', 'eolian', 'meridiani planum', 'crater', 'formed', 'evidence',
                            'past', 'dune', 'bed']

        self.identify_planetary_entities.search_retrieval.identify_terms_query = MagicMock(return_value=[solrdata.doc_1])
        self.identify_planetary_entities.match_excerpt.forward = MagicMock(return_value=(True, [excerpts.doc_1_excerpts[0]['excerpt']]))
        self.identify_planetary_entities.extract_keywords.forward = MagicMock(return_value=keywords_forward)
        self.identify_planetary_entities.knowledge_graph_positive.forward = MagicMock(return_value=0.7)
        self.identify_planetary_entities.knowledge_graph_negative.forward = MagicMock(return_value=0.3)

        self.identify_planetary_entities.get_knowledge_graph_score = MagicMock(return_value=0.7)
        self.identify_planetary_entities.get_paper_relevance_score = MagicMock(return_value=0.8)
        self.identify_planetary_entities.get_local_llm_score = MagicMock(return_value=0.7)

        result = self.identify_planetary_entities.identify()

        # Assert the expected behavior
        self.assertEqual(len(result), 1)
        history_entry, identified_doc = result[0]
        self.assertIsInstance(history_entry, NamedEntityHistory)
        self.assertEqual(history_entry.feature_name_entity, 'Rayleigh')
        self.assertEqual(history_entry.feature_type_entity, 'Crater')
        self.assertEqual(history_entry.target_entity, 'Mars')
        self.assertEqual(len(identified_doc), 1)
        self.assertIsInstance(identified_doc[0], NamedEntity)
        self.assertEqual(identified_doc[0].keywords_item_id, 1)
        self.assertEqual(identified_doc[0].excerpt, excerpts.doc_1_excerpts[0]['excerpt'])
        self.assertEqual(identified_doc[0].keywords, keywords_forward)
        self.assertEqual(identified_doc[0].special_keywords, [])

if __name__ == '__main__':
    unittest.main()
