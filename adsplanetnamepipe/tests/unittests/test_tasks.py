import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)


import unittest
from unittest.mock import MagicMock, patch

from config import PLANETARYNAMES_PIPELINE_ACTION
from adsplanetnamepipe.tasks import task_process_planetary_nomenclature, FailedRequest
from adsplanetnamepipe.utils.common import EntityArgs


class TestPlanetaryNomenclature(unittest.TestCase):

    def setUp(self):
        """ Set up the config class and create an instance of AstroBERTNER """

        self.args = EntityArgs(
            target="Mars",
            feature_type = "Crater",
            feature_type_plural = "Craters",
            feature_name = "Rayleigh",
            context_ambiguous_feature_names = ["asteroid", "main belt asteroid", "Moon", "Mars"],
            multi_token_containing_feature_names = ["Rayleigh A", "Rayleigh B", "Rayleigh C", "Rayleigh D"],
            name_entity_labels = [{'label': 'planetary', 'value': 1}, {'label': 'non planetary', 'value': 0}],
            all_targets = ["Mars", "Mercury", "Moon", "Venus"]
        )

    @patch('adsplanetnamepipe.tasks.app')
    @patch('adsplanetnamepipe.tasks.CollectKnowldegeBase')
    def test_task_process_planetary_nomenclature_collect(self, mock_collect_knowledgebase, mock_app):
        """ calling tasks queue when in collecting stage """

        mock_record = MagicMock()
        mock_collect_instance = MagicMock()
        mock_collect_instance.collect.return_value = [mock_record]
        mock_collect_knowledgebase.return_value = mock_collect_instance

        mock_app.insert_knowledge_base_records.return_value = True

        the_task = {'action_type': PLANETARYNAMES_PIPELINE_ACTION.collect, 'args': self.args}

        result = task_process_planetary_nomenclature(the_task)

        mock_collect_knowledgebase.assert_called_once_with(the_task['args'])
        mock_app.insert_knowledge_base_records.assert_called_once_with([mock_record])
        self.assertTrue(result)

    @patch('adsplanetnamepipe.tasks.app')
    @patch('adsplanetnamepipe.tasks.IdentifyPlanetaryEntities')
    def test_task_process_planetary_nomenclature_identify(self, mock_identify_planetary_entities, mock_app):
        """ calling task queue when in identifying stage """

        mock_named_entity_record = MagicMock()
        mock_identify_instance = MagicMock()
        mock_identify_instance.identify.return_value = [mock_named_entity_record]
        mock_identify_planetary_entities.return_value = mock_identify_instance

        mock_app.get_knowledge_base_keywords.side_effect = [['positive_keyword'], ['negative_keyword']]
        mock_app.insert_named_entity_records.return_value = True

        the_task = {'action_type': PLANETARYNAMES_PIPELINE_ACTION.identify, 'args': self.args}

        result = task_process_planetary_nomenclature(the_task)

        mock_identify_planetary_entities.assert_called_once_with(the_task['args'], ['positive_keyword'],
                                                                 ['negative_keyword'])
        mock_app.get_knowledge_base_keywords.assert_any_call(the_task['args'].feature_name,
                                                             the_task['args'].feature_type,
                                                             the_task['args'].target,
                                                             the_task['args'].name_entity_labels[0]['label'])
        mock_app.get_knowledge_base_keywords.assert_any_call(the_task['args'].feature_name,
                                                             the_task['args'].feature_type,
                                                             the_task['args'].target,
                                                             the_task['args'].name_entity_labels[1]['label'])
        mock_app.insert_named_entity_records.assert_called_once_with([mock_named_entity_record])
        self.assertTrue(result)

    def test_task_process_planetary_nomenclature_invalid_task(self):
        """ calling tasks queue when in collecting stage and fails """

        the_task = {}

        result = task_process_planetary_nomenclature(the_task)

        self.assertFalse(result)

    @patch('adsplanetnamepipe.tasks.CollectKnowldegeBase')
    def test_task_process_planetary_nomenclature_collect_failed(self, mock_collect_knowledgebase):
        """  """
        mock_collect_instance = MagicMock()
        mock_collect_instance.collect.return_value = []
        mock_collect_knowledgebase.return_value = mock_collect_instance

        the_task = {'action_type': PLANETARYNAMES_PIPELINE_ACTION.collect, 'args': self.args}

        result = task_process_planetary_nomenclature(the_task)

        self.assertFalse(result)

    @patch('adsplanetnamepipe.tasks.app.get_knowledge_base_keywords')
    @patch('adsplanetnamepipe.tasks.IdentifyPlanetaryEntities')
    def test_task_process_planetary_nomenclature_identify_failed(self, mock_identify_planetary_entities, mock_get_keywords):
        """ calling task queue when in identifying stage and fails """

        mock_get_keywords.return_value = False
        mock_identify_instance = mock_identify_planetary_entities.return_value
        mock_identify_instance.identify.return_value = None

        the_task = {'action_type': PLANETARYNAMES_PIPELINE_ACTION.identify, 'args': self.args}

        result = task_process_planetary_nomenclature(the_task)

        self.assertEqual(mock_get_keywords.call_count, 2)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()