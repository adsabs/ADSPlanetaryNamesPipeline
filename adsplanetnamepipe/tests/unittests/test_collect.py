import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)


import unittest
from unittest.mock import MagicMock, patch

from typing import List, Tuple

from adsplanetnamepipe.collect import CollectKnowldegeBase
from adsplanetnamepipe.models import KnowledgeBase, KnowledgeBaseHistory
from adsplanetnamepipe.utils.common import EntityArgs

from adsplanetnamepipe.tests.unittests.stubdata import solrdata
from adsplanetnamepipe.tests.unittests.stubdata import excerpts


class TestCollectKnowldegeBase(unittest.TestCase):

    """
    Tests the collect module
    """

    def setUp(self):
        """ Set up the config class and create an instance of CollectKnowldegeBase """

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
        self.collect_knowldegebase = CollectKnowldegeBase(self.args)

    def test_get_paper_relevance_score(self):
        """ test get_paper_relevance_score method """

        self.collect_knowldegebase.paper_relevance.forward = MagicMock(return_value=0.8)
        score = self.collect_knowldegebase.get_paper_relevance_score(solrdata.doc_1)
        self.assertEqual(score, 0.8)

    def test_get_local_llm_score(self):
        """ test get_local_llm_score """

        self.collect_knowldegebase.local_llm.forward = MagicMock(return_value=0.7)
        score = self.collect_knowldegebase.get_local_llm_score(solrdata.doc_1, excerpts.doc_1_excerpts[0]['excerpt'])
        self.assertEqual(score, 0.7)

    def test_collect_KB_positive(self):
        """ test collect_KB_positive method """

        keywords_forward_doc = ['crater', 'rayleigh', 'ripple', 'figure', 'exposed', 'rockingham', 'drake', 'north',
                                'diameter', 'diligence', 'interior', 'banding', 'small', 'image', 'layer', 'opportunity',
                                'superposed', 'rim', 'bedrock', 'sandy']
        keywords_forward = ['ripple', 'mars', 'discovery', 'eolian', 'meridiani planum', 'crater', 'formed', 'evidence',
                            'past', 'dune', 'bed']
        keywords_forward_special = ['topography', 'craters', 'radii']

        self.collect_knowldegebase.search_retrieval.collect_usgs_terms_query = MagicMock(return_value=[solrdata.doc_1])
        self.collect_knowldegebase.match_excerpt.forward = MagicMock(return_value=(True, [excerpts.doc_1_excerpts[0]['excerpt']]))
        self.collect_knowldegebase.extract_keywords.forward_doc = MagicMock(return_value=keywords_forward_doc)
        self.collect_knowldegebase.extract_keywords.forward = MagicMock(return_value=keywords_forward)
        self.collect_knowldegebase.extract_keywords.forward_special = MagicMock(return_value=keywords_forward_special)

        self.collect_knowldegebase.get_paper_relevance_score = MagicMock(return_value=0.8)
        self.collect_knowldegebase.get_local_llm_score = MagicMock(return_value=0.7)

        result = self.collect_knowldegebase.collect_KB_positive()

        # Assert the expected behavior
        self.assertEqual(len(result), 1)
        history_entry, collected_doc = result[0]
        self.assertIsInstance(history_entry, KnowledgeBaseHistory)
        self.assertEqual(history_entry.feature_name_entity, 'Rayleigh')
        self.assertEqual(history_entry.feature_type_entity, 'Crater')
        self.assertEqual(history_entry.target_entity, 'Mars')
        self.assertEqual(history_entry.named_entity_label, 'planetary')
        self.assertEqual(len(collected_doc), 2)
        self.assertIsInstance(collected_doc[0], KnowledgeBase)
        self.assertIsInstance(collected_doc[1], KnowledgeBase)
        self.assertEqual(collected_doc[0].keywords_item_id, 0)
        self.assertEqual(collected_doc[1].keywords_item_id, 1)
        self.assertEqual(collected_doc[0].excerpt, None)
        self.assertEqual(collected_doc[1].excerpt, excerpts.doc_1_excerpts[0]['excerpt'])
        self.assertEqual(collected_doc[0].keywords, keywords_forward_doc)
        self.assertEqual(collected_doc[1].keywords, keywords_forward)
        self.assertEqual(collected_doc[0].special_keywords, [])
        self.assertEqual(collected_doc[1].special_keywords, keywords_forward_special)

    def test_collect_KB_negative(self):
        """ test collect_KB_positive method """

        keywords_forward_doc = ['order', 'scattering', 'frost', 'cboe', 'ice', 'regolith', 'second', 'particle',
                                'small', 'wavelength', 'polarization', 'dust', 'higher', 'surface', 'smaller',
                                'thick', 'decrease', 'ten', 'increase', 'crystal']

        self.collect_knowldegebase.search_retrieval.collect_non_usgs_terms_query = MagicMock(return_value=[solrdata.doc_2])
        self.collect_knowldegebase.extract_keywords.forward_doc = MagicMock(return_value=keywords_forward_doc)

        result = self.collect_knowldegebase.collect_KB_negative()

        # Assert the expected behavior
        self.assertEqual(len(result), 1)
        history_entry, collected_doc = result[0]
        self.assertIsInstance(history_entry, KnowledgeBaseHistory)
        self.assertEqual(history_entry.feature_name_entity, 'Rayleigh')
        self.assertEqual(history_entry.feature_type_entity, 'Crater')
        self.assertEqual(history_entry.target_entity, 'Mars')
        self.assertEqual(history_entry.named_entity_label, 'non planetary')
        self.assertEqual(len(collected_doc), 1)
        self.assertIsInstance(collected_doc[0], KnowledgeBase)
        self.assertEqual(collected_doc[0].keywords_item_id, 0)
        self.assertEqual(collected_doc[0].excerpt, None)
        self.assertEqual(collected_doc[0].keywords, keywords_forward_doc)

    @patch('adsplanetnamepipe.collect.CollectKnowldegeBase.collect_KB_positive')
    @patch('adsplanetnamepipe.collect.CollectKnowldegeBase.collect_KB_negative')
    def test_collect(self, mock_collect_KB_negative, mock_collect_KB_positive):
        """ test the collect method """

        positive_result: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]] = [
            (KnowledgeBaseHistory(None, None, None, None, None), [KnowledgeBase(None, None, None, None, None, None, None),
                                                                  KnowledgeBase(None, None, None, None, None, None, None)])]
        negative_result: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]] = [
            (KnowledgeBaseHistory(None, None, None, None, None), [KnowledgeBase(None, None, None, None, None, None, None)])]

        mock_collect_KB_positive.return_value = positive_result
        mock_collect_KB_negative.return_value = negative_result

        result = self.collect_knowldegebase.collect()
        self.assertEqual(len(result), 2)

        self.assertEqual(result[0], positive_result[0])
        self.assertEqual(result[1], negative_result[0])


if __name__ == '__main__':
    unittest.main()
