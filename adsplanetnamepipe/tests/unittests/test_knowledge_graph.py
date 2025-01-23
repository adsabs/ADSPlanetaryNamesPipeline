import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

import unittest
from unittest.mock import MagicMock

import networkx as nx

from adsplanetnamepipe.utils.knowledge_graph import KnowledgeGraph
from adsplanetnamepipe.utils.common import EntityArgs

from adsplanetnamepipe.tests.unittests.stubdata import keywords


class TestKnowledgeGraph(unittest.TestCase):

    def setUp(self):
        """ Set up the config class and create two instances of KnowledgeGraph: positive and negative """

        self.args = EntityArgs(
            target="Moon",
            feature_type = "Crater",
            feature_type_plural = "Craters",
            feature_name = "Antoniadi",
            context_ambiguous_feature_names = ["Moon", "Mars"],
            multi_token_containing_feature_names = ["Antoniadi Dorsum"],
            name_entity_labels = [{'label': 'planetary', 'value': 1}, {'label': 'non planetary', 'value': 0}],
            timestamp='2000-01-01',
            all_targets = ["Mars", "Mercury", "Moon", "Venus"]
        )

    def test_forward(self):
        """  """
        knowledge_graph_positive = KnowledgeGraph(self.args, [row[2] for row in keywords.collect_positive],
                                                             [row[3] for row in keywords.collect_positive])
        knowledge_graph_negative = KnowledgeGraph(self.args, [row[2] for row in keywords.collect_negative],
                                                             [row[3] for row in keywords.collect_negative])

        expected_result_positive = [
            3.79, 2.4, 2.2, 5.5, 4.0, 5.37, 4.6, 5.2, 5.5, 2.5, 2.1, 1.0, 3.9, 5.05, 4.4, 1.9, 3.06, 4.7, 4.4, 2.79,
            3.2, 0.71, 3.6, 2.78, 0.4, 3.67, 4.3, 4.25, 3.7, 4.4, 4.1, 2.6, 3.2, 3.6, 3.6, 3.9, 1.1, 1.87, 4.0, 1.2,
            3.59, 3.5, 3.3, 3.5, 3.9, 1.7, 1.9, 3.79, 5.25, 1.2, 3.2, 1.6, 2.2, 1.8, 2.9, 3.4, 2.76, 3.5, 3.5, 4.56        ]
        expected_result_negative = [
            0.16, 0.0, 0.2, 0.0, 0.2, 0.53, 0.0, 0.3, 0.3, 0.4, 0.6, 0.1, 0.1, 0.74, 0.0, 0.1, 0.44, 0.4, 0.5, 0.16,
            0.0, 0.18, 0.1, 0.39, 0.1, 0.17, 0.9, 0.25, 0.6, 0.8, 0.2, 0.0, 0.1, 0.6, 0.4, 0.2, 0.1, 0.8, 0.0, 0.5,
            1.24, 0.0, 0.2, 0.4, 0.7, 0.4, 0.1, 0.05, 0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.6, 0.1, 0.24, 0.3, 0.3, 0.44
        ]

        identify_keywords = [row[2] for row in keywords.identify]
        result_positive = [knowledge_graph_positive.forward(identify_keyword) for identify_keyword in identify_keywords]
        result_negative = [knowledge_graph_negative.forward(identify_keyword) for identify_keyword in identify_keywords]
        self.assertEqual(result_positive, expected_result_positive)
        self.assertEqual(result_negative, expected_result_negative)

    def test_forward_no_negative_graph(self):
        """  """

        knowledge_graph_positive = KnowledgeGraph(self.args, [row[2] for row in keywords.collect_positive],
                                                             [row[3] for row in keywords.collect_positive])
        knowledge_graph_negative = KnowledgeGraph(self.args, [], [])

        expected_result_positive = [
            3.79, 2.4, 2.2, 5.5, 4.0, 5.37, 4.6, 5.2, 5.5, 2.5, 2.1, 1.0, 3.9, 5.05, 4.4, 1.9, 3.06, 4.7, 4.4, 2.79,
            3.2, 0.71, 3.6, 2.78, 0.4, 3.67, 4.3, 4.25, 3.7, 4.4, 4.1, 2.6, 3.2, 3.6, 3.6, 3.9, 1.1, 1.87, 4.0, 1.2,
            3.59, 3.5, 3.3, 3.5, 3.9, 1.7, 1.9, 3.79, 5.25, 1.2, 3.2, 1.6, 2.2, 1.8, 2.9, 3.4, 2.76, 3.5, 3.5, 4.56
        ]
        expected_result_negative = [
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1
        ]

        identify_keywords = [row[2] for row in keywords.identify]
        result_positive = [knowledge_graph_positive.forward(identify_keyword) for identify_keyword in identify_keywords]
        result_negative = [knowledge_graph_negative.forward(identify_keyword) for identify_keyword in identify_keywords]
        self.assertEqual(result_positive, expected_result_positive)
        self.assertEqual(result_negative, expected_result_negative)

    def test_query_path_exception(self):
        """  """

        mock_graph = MagicMock(spec=nx.Graph)
        mock_graph.has_node.return_value = True

        # Create a mock args object
        mock_args = MagicMock()
        mock_args.feature_name = self.args.feature_name

        knowledge_graph_positive = KnowledgeGraph(self.args, [row[2] for row in keywords.collect_positive], [])
        knowledge_graph_positive.graph = mock_graph

        identify_keywords = [row[2] for row in keywords.identify]
        with unittest.mock.patch('networkx.shortest_path', side_effect=nx.NetworkXNoPath()):
            result = knowledge_graph_positive.query_path(identify_keywords[0][0])

            self.assertEqual(result, 0)
            mock_graph.has_node.assert_called_once_with(identify_keywords[0][0])


if __name__ == '__main__':
    unittest.main()
