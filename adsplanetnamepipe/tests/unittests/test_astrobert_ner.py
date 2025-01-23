import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)


import unittest

from unittest.mock import MagicMock, patch

from adsplanetnamepipe.utils.astrobert_ner import AstroBERTNER
from adsplanetnamepipe.utils.common import EntityArgs

from adsplanetnamepipe.tests.unittests.stubdata import excerpts


class TestAstroBERTNER(unittest.TestCase):

    """
    Tests the astrobert NER module
    """

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
            timestamp='2000-01-01',
            all_targets = ["Mars", "Mercury", "Moon", "Venus"]
        )
        self.astrobert_ner = AstroBERTNER(self.args)

    def test_forward(self):
        """  """

        # when entity is recognized as CelestialObject
        self.astrobert_ner.astrobert_ner = MagicMock(return_value=[
            {'entity_group': 'CelestialObject', 'score': 0.42149615, 'word': 'Rayleigh', 'start': 350, 'end': 358},
            {'entity_group': 'CelestialObject', 'score': 0.39897928, 'word': 'Drake', 'start': 449, 'end': 454},
            {'entity_group': 'Mission', 'score': 0.41518763, 'word': 'Opportunity', 'start': 501, 'end': 512}
        ])
        self.astrobert_ner.is_citation_or_reference = MagicMock(return_value=False)
        result = self.astrobert_ner.forward(excerpts.doc_1_excerpts[3]['excerpt'],
                                            excerpts.doc_1_excerpts[3]['entity_span_within_excerpt'])
        self.assertTrue(result)

        # when entity is not recognized
        self.astrobert_ner.astrobert_ner = MagicMock(return_value=[
            {'entity_group': 'CelestialObject', 'score': 0.39897928, 'word': 'Drake', 'start': 449, 'end': 454},
            {'entity_group': 'Mission', 'score': 0.41518763, 'word': 'Opportunity', 'start': 501, 'end': 512}
        ])
        result = self.astrobert_ner.forward(excerpts.doc_1_excerpts[3]['excerpt'],
                                            excerpts.doc_1_excerpts[3]['entity_span_within_excerpt'])
        self.assertTrue(result)

        # when entity is recognized as other then CelestialObject
        self.astrobert_ner.astrobert_ner = MagicMock(return_value=[
            {'entity_group': 'Model', 'score': 0.42149615, 'word': 'Rayleigh', 'start': 350, 'end': 358},
            {'entity_group': 'CelestialObject', 'score': 0.39897928, 'word': 'Drake', 'start': 449, 'end': 454},
            {'entity_group': 'Mission', 'score': 0.41518763, 'word': 'Opportunity', 'start': 501, 'end': 512}
        ])
        result = self.astrobert_ner.forward(excerpts.doc_1_excerpts[3]['excerpt'],
                                            excerpts.doc_1_excerpts[3]['entity_span_within_excerpt'])
        self.assertFalse(result)

        # when entity is citation or reference
        self.astrobert_ner.astrobert_ner = MagicMock(return_value=[
            {'entity_group': 'CelestialObject', 'score': 0.39897928, 'word': 'Drake', 'start': 449, 'end': 454},
            {'entity_group': 'Mission', 'score': 0.41518763, 'word': 'Opportunity', 'start': 501, 'end': 512}
        ])
        self.astrobert_ner.is_citation_or_reference = MagicMock(return_value=True)
        result = self.astrobert_ner.forward(excerpts.doc_1_excerpts[3]['excerpt'],
                                            excerpts.doc_1_excerpts[3]['entity_span_within_excerpt'])
        self.assertFalse(result)

    @patch('adsplanetnamepipe.utils.astrobert_ner.logger')
    def test_forward_exception(self, mock_logger):
        """ test astrobert_ner method when raises a RuntimeError """

        self.astrobert_ner.astrobert_ner = MagicMock(side_effect=RuntimeError)
        result = self.astrobert_ner.forward(excerpts.doc_1_excerpts[3]['excerpt'],
                                            excerpts.doc_1_excerpts[3]['entity_span_within_excerpt'])
        self.assertFalse(result)
        mock_logger.error.assert_called_once_with('AstroBERT NER throw RuntimeError.')

    def test_is_citation_or_reference(self):
        """ test is_citation_or_reference method """

        # not citation or reference
        result = self.astrobert_ner.is_citation_or_reference(excerpts.doc_1_excerpts[8]['excerpt'], excerpts.doc_1_excerpts[8]['entity_span_within_excerpt'])
        self.assertFalse(result)

        # citation strings
        citation_excerpts = [
            "This method of plotting profiles of pyroclastic deposits was first implemented by Weinek (1973) and is most often used by terrestrial volcanologists to calculate deposit volumes using, for example, methods described by Pyle (1989).",
            "The HiRISE dataset is included in Whipple et al. (2007).",
            "The HRSC dataset is included in Walter and van Gasselt (2014).",
            "See table 23 in Weber and Bischoff (1988).",
            "According to Wallach 2021, the results were conclusive.",
            "The study by Walter 1985a provided valuable insights.",
            "The findings were consistent with previous research (Davis and Wilson, 2018).",
            "As demonstrated by Patel, Werner, and Lee (2019), the method was effective.",
            "The theory was first proposed by Weyl 2010.",
            "The experiment was replicated in a recent study (White et al., 2020).",
            "Wright et al. 2017 expanded on the previous findings.",
        ]

        entity_spans = [(82, 88), (34, 41), (32, 38), (16, 21), (13, 20), (13, 19),
                        (63, 69), (26, 32), (33, 37), (49, 54), (0, 6)]
        for excerpt, entity_span in zip(citation_excerpts, entity_spans):
            result = self.astrobert_ner.is_citation_or_reference(excerpt, entity_span)
            self.assertTrue(result)

        # reference strings
        reference_excerpts = ["Manrique, H. M., Friston, K. J., & Walker, M. J. (2024). 'Snakes and ladders' in paleoanthropology: From cognitive surprise to skillfulness a million years ago. Physics of Life Reviews, 49, 40-70. https://doi.org/10.1016/j.plrev.2024.01.004",
                              "Walker, S. 'In Search of the New.' Sky and Telescope, vol. 147, no. 6, 2024, p. 62."
        ]
        entity_spans = [(35, 41), (0, 6)]
        self.astrobert_ner.args = EntityArgs(
            target="Moon",
            feature_type="Crater",
            feature_type_plural="Craters",
            feature_name="Walker",
            context_ambiguous_feature_names=[],
            multi_token_containing_feature_names=[],
            name_entity_labels=[{'label': 'planetary', 'value': 1}, {'label': 'non planetary', 'value': 0}],
            timestamp='2000-01-01',
            all_targets = []
        )
        for excerpt, entity_span in zip(reference_excerpts, entity_spans):
            result = self.astrobert_ner.is_citation_or_reference(excerpt, entity_span)
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
