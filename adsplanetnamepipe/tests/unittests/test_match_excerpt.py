import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

import regex

import unittest
from unittest.mock import MagicMock, patch

from adsplanetnamepipe.utils.match_excerpt import MatchExcerpt, RegExResult, RegExPattern
from adsplanetnamepipe.utils.common import EntityArgs

from adsplanetnamepipe.tests.unittests.stubdata import solrdata
from adsplanetnamepipe.tests.unittests.stubdata import excerpts


class TestMatchExcerpt(unittest.TestCase):

    """
    Tests the match excerpt module
    """

    def setUp(self):
        """ Set up the config class and create an instance of MatchExcerpt """

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
        self.match_excerpt = MatchExcerpt(self.args)

    def test_get_fulltext(self):
        """ Test the get_fulltext method """

        # when it is success
        fulltext = self.match_excerpt.get_fulltext(solrdata.doc_1)
        self.assertEqual(fulltext, f"{' '.join(solrdata.doc_1['title'])} {solrdata.doc_1['abstract']} {solrdata.doc_1['body']}")

        # when it fails
        self.match_excerpt.is_language_english = MagicMock(return_value=False)
        fulltext = self.match_excerpt.get_fulltext(solrdata.doc_1)
        self.assertEqual(fulltext, '')

    @patch('adsplanetnamepipe.utils.match_excerpt.detect')
    def test_is_language_english(self, mock_detect):
        """ Test the is_language_english method """

        mock_detect.return_value = 'en'
        result = self.match_excerpt.is_language_english('The Milky Way is a barred spiral galaxy that contains our solar system.', 'bibcode123')
        self.assertTrue(result)

        mock_detect.return_value = 'gr'
        result = self.match_excerpt.is_language_english('Die Milchstraße ist eine Balkenspiralgalaxie, die unser Sonnensystem enthält.', 'bibcode456')
        self.assertFalse(result)

    @patch('adsplanetnamepipe.utils.match_excerpt.detect')
    @patch('adsplanetnamepipe.utils.match_excerpt.logger')
    def test_is_language_english_exception(self, mock_logger, mock_detect):
        """ Test the is_language_english method when throws exception """

        mock_detect.side_effect = Exception("Language detection error")
        result = self.match_excerpt.is_language_english('Some text that is going to cause exception.', 'bibcode123')

        self.assertFalse(result)
        mock_logger.error.assert_called_with(f"Unable to detect the language for `bibcode123`. Concluding the fulltext is not in English and ignoring this record.")

    def test_determine_celestial_body_relevance(self):
        """ Test the determine_celestial_body_relevance method """

        # Test case 1: contains multiple instances of target, Mars
        text = solrdata.doc_1['abstract']
        result = self.match_excerpt.determine_celestial_body_relevance(text)
        self.assertTrue(result)

        # Test case 2: the excerpt does not contain the target, Mars
        text = solrdata.doc_1['body']
        result = self.match_excerpt.determine_celestial_body_relevance(text)
        self.assertFalse(result)

        # Test case 3: when the ambiguous entities appear in the record the same number of times
        text = 'If in the text there are the same number of instances of the ambigous entities (ie, Moon and Mars) ' \
               'then the function returns true, saying proceed to the next step to determine disambiguity. Hence, ' \
               'if Moon and Mars appear the same number of times, with the the other two terms not appearing, or appearing ' \
               'less than Moon and Mars, then this function says I cannot determine if the record is about Moon or Mars. ' \
               'Actually the other two terms can also apper the same number of times as Moon and Mars, the behavior would be ' \
               'the same. So in this text there are 5 instances of Moon and Mars, and with this one it is 6, so the function ' \
               'shall return True, proceed until later.'
        result = self.match_excerpt.determine_celestial_body_relevance(text)
        self.assertTrue(result)

        # Test case 4: the entity is not ambiguous
        self.args.context_ambiguous_feature_names = []
        result = self.match_excerpt.determine_celestial_body_relevance('')
        self.assertTrue(result)

    def test_select_excerpts(self):
        """ Test the select_excerpts method """

        # there are 10 instances of entity Rayleigh in the following excerpt
        text = solrdata.doc_1['body']
        excerpts = self.match_excerpt.select_excerpts(text)
        for excerpt in excerpts:
            self.assertEqual(len(excerpts), 10)

    def test_is_context_non_planetary(self):
        """ Test the is_context_non_planetary method """

        # entity is usgs, asking it if it is not, if it contains target entity -> False
        result = self.match_excerpt.is_context_non_planetary(excerpts.doc_1_excerpts[1]['excerpt'])
        self.assertFalse(result)

        # entity is usgs, asking it if it is not, if it contains feature type entity either singular or plural -> False
        result = self.match_excerpt.is_context_non_planetary(excerpts.doc_1_excerpts[2]['excerpt'])
        self.assertFalse(result)

        # entity is not usgs, asking it if it is not -> True
        text = "mixture of Rubidium (with 73% of 85Rb and 27% of 87Rb) at room temperature. Inside the cell, the probe " \
               "(coupling) beam has a waist diameter of 0.3 (0.4) mm with an 11.5 (14.5) cm Rayleigh length, which is longer " \
               "than the 7.5 cm vapor cell length. The probe input power is 0.4 µW, and the coupling one is 54 mW. A horn " \
               "antenna (MVG QR18000) is placed 57.5 cm away from"
        result = self.match_excerpt.is_context_non_planetary(text)
        self.assertTrue(result)

    @patch('adsplanetnamepipe.utils.match_excerpt.SpacyWrapper')
    @patch('adsplanetnamepipe.utils.match_excerpt.YakeWrapper')
    def test_validate_feature_name(self, mock_yake, mock_spacy):
        """ Test the validate_feature_name method """

        # when both spacy and yake pass the validation, the result is True
        mock_spacy.validate_feature_name.return_value = True
        mock_yake.validate_feature_name.return_value = True
        result = self.match_excerpt.validate_feature_name(excerpts.doc_1_excerpts[7]['excerpt'],
                                                          excerpts.doc_1_excerpts[7]['entity_span_within_excerpt'],
                                                          True)
        self.assertTrue(result)

        # when yake does not pass the excerpt, the result is False
        mock_spacy.validate_feature_name.return_value = True
        mock_yake.validate_feature_name.return_value = False
        result = self.match_excerpt.validate_feature_name(excerpts.doc_2_excerpts[0]['excerpt'],
                                                          excerpts.doc_2_excerpts[0]['entity_span_within_excerpt'],
                                                          True)
        self.assertFalse(result)

        # when spacy does not pass the excerpt, the result is False
        mock_spacy.validate_feature_name.return_value = False
        mock_yake.validate_feature_name.return_value = True
        result = self.match_excerpt.validate_feature_name(excerpts.doc_3_excerpts[2]['excerpt'],
                                                          excerpts.doc_3_excerpts[2]['entity_span_within_excerpt'],
                                                          True)
        self.assertFalse(result)

    @patch('adsplanetnamepipe.utils.match_excerpt.ADSabsNER')
    @patch('adsplanetnamepipe.utils.match_excerpt.logger')
    def test_forward_1(self, mock_logger, mock_astrobert_ner):
        """ Test the forward method -- case 1: usgs_term=True, relevant document """

        self.match_excerpt.get_fulltext = MagicMock(return_value=f"{' '.join(solrdata.doc_1['title'])} {solrdata.doc_1['abstract']} {solrdata.doc_1['body']}")
        self.match_excerpt.determine_celestial_body_relevance = MagicMock(return_value=True)
        self.match_excerpt.select_excerpts = MagicMock(return_value=[MagicMock(excerpt=excerpts.doc_1_excerpts[0]['excerpt']),
                                                                     MagicMock(excerpt=excerpts.doc_1_excerpts[1]['excerpt'])])
        self.match_excerpt.validate_feature_name = MagicMock(return_value=True)
        mock_astrobert_ner.forward = MagicMock(return_value=True)
        result = self.match_excerpt.forward(solrdata.doc_1, mock_astrobert_ner, usgs_term=True)
        self.assertTrue(result[0])
        self.assertEqual(result[1], [excerpts.doc_1_excerpts[0]['excerpt'], excerpts.doc_1_excerpts[1]['excerpt']])
        mock_logger.info.assert_called_with("For record `2010JGRE..115.0F08G` there are 2 relevant excerpts extracted in the step Match Excerpts.")

    @patch('adsplanetnamepipe.utils.match_excerpt.ADSabsNER')
    @patch('adsplanetnamepipe.utils.match_excerpt.logger')
    def test_forward_2(self, mock_logger, mock_astrobert_ner):
        """ Test the forward method -- case 2: usgs_term=True, irrelevant document """

        self.match_excerpt.get_fulltext = MagicMock(return_value=f"{' '.join(solrdata.doc_2['title'])} {solrdata.doc_2['abstract']} {solrdata.doc_2['body']}")
        self.match_excerpt.determine_celestial_body_relevance = MagicMock(return_value=False)
        self.match_excerpt.select_excerpts = MagicMock(return_value=[])
        result = self.match_excerpt.forward(solrdata.doc_2, mock_astrobert_ner, usgs_term=True)
        self.assertFalse(result[0])
        self.assertEqual(result[1], [])
        mock_logger.info.assert_called_with("Record `2023Icar..39615503S` is determined not to be relevant for target Mars. Record filtered out.")

    @patch('adsplanetnamepipe.utils.match_excerpt.ADSabsNER')
    @patch('adsplanetnamepipe.utils.match_excerpt.logger')
    def test_forward_3(self, mock_logger, mock_astrobert_ner):
        """ Test the forward method -- case 3: usgs_term=False, non-planetary context """

        self.match_excerpt.is_context_non_planetary = MagicMock(return_value=True)
        result = self.match_excerpt.forward(solrdata.doc_3, mock_astrobert_ner, usgs_term=False)
        self.assertTrue(result[0])
        self.assertEqual(result[1], [])
        mock_logger.info.assert_called_with("For record `2023GeoRL..5001666K` determined it is not planetary record and hence keywords will be extacted from the fulltext in the next step.")

    @patch('adsplanetnamepipe.utils.match_excerpt.ADSabsNER')
    @patch('adsplanetnamepipe.utils.match_excerpt.logger')
    def test_forward_4(self, mock_logger, mock_astrobert_ner):
        """ Test the forward method -- case 4: usgs_term=False, planetary context """

        self.match_excerpt.is_context_non_planetary = MagicMock(return_value=False)
        result = self.match_excerpt.forward(solrdata.doc_1, mock_astrobert_ner, usgs_term=False)
        self.assertFalse(result[0])
        self.assertEqual(result[1], [])
        mock_logger.info.assert_called_with("Record `2010JGRE..115.0F08G` is determined to be usgs relevant, and hence cannot be processed for non usgs phase.")

    @patch('adsplanetnamepipe.utils.match_excerpt.ADSabsNER')
    @patch('adsplanetnamepipe.utils.match_excerpt.logger')
    def test_forward_5(self, mock_logger, mock_astrobert_ner):
        """ Test the forward method -- case 5: when excerpt is determined by adsabs_ner not to be relevant """

        self.match_excerpt.get_fulltext = MagicMock(return_value=f"{' '.join(solrdata.doc_1['title'])} {solrdata.doc_1['abstract']} {solrdata.doc_1['body']}")
        self.match_excerpt.determine_celestial_body_relevance = MagicMock(return_value=True)
        self.match_excerpt.select_excerpts = MagicMock(return_value=[MagicMock(excerpt=excerpts.doc_1_excerpts[0]['excerpt'])])
        self.match_excerpt.validate_feature_name = MagicMock(return_value=True)
        mock_astrobert_ner.forward = MagicMock(return_value=False)
        result = self.match_excerpt.forward(solrdata.doc_1, mock_astrobert_ner, usgs_term=True)
        self.assertTrue(result[0])
        self.assertEqual(result[1], [])
        mock_logger.info.assert_any_call("An excerpt from the record `2010JGRE..115.0F08G` is determined not relevant by AstroBERT NER. Record filtered out.")
        mock_logger.info.assert_any_call("For record `2010JGRE..115.0F08G` there are 0 relevant excerpts extracted in the step Match Excerpts.")

    @patch('adsplanetnamepipe.utils.match_excerpt.ADSabsNER')
    @patch('adsplanetnamepipe.utils.match_excerpt.logger')
    def test_forward_6(self, mock_logger, mock_astrobert_ner):
        """ Test the forward method -- case 6: when excerpt is determined by token_phrase analysis not to be relevant """

        self.match_excerpt.get_fulltext = MagicMock(return_value=f"{' '.join(solrdata.doc_1['title'])} {solrdata.doc_1['abstract']} {solrdata.doc_1['body']}")
        self.match_excerpt.determine_celestial_body_relevance = MagicMock(return_value=True)
        self.match_excerpt.select_excerpts = MagicMock(return_value=[MagicMock(excerpt=excerpts.doc_1_excerpts[0]['excerpt'])])
        self.match_excerpt.validate_feature_name = MagicMock(return_value=False)
        mock_astrobert_ner.forward = MagicMock(return_value=True)
        result = self.match_excerpt.forward(solrdata.doc_1, mock_astrobert_ner, usgs_term=True)
        self.assertTrue(result[0])
        self.assertEqual(result[1], [])
        mock_logger.info.assert_any_call("An excerpt from the record `2010JGRE..115.0F08G` is determined not relevant by token/pharse analysis. Record filtered out.")
        mock_logger.info.assert_any_call("For record `2010JGRE..115.0F08G` there are 0 relevant excerpts extracted in the step Match Excerpts.")

    @patch('adsplanetnamepipe.utils.match_excerpt.ADSabsNER')
    @patch('adsplanetnamepipe.utils.match_excerpt.logger')
    def test_forward_7(self, mock_logger, mock_astrobert_ner):
        """ Test the forward method -- case 7: when the record is determined not to be English """

        self.match_excerpt.get_fulltext = MagicMock(return_value=None)
        result = self.match_excerpt.forward(solrdata.doc_3, mock_astrobert_ner, usgs_term=False)
        self.assertFalse(result[0])
        self.assertEqual(result[1], [])
        mock_logger.info.assert_called_with(f"Record `2023GeoRL..5001666K` is determined not to be in English. It is filtered out.")

    def test_synonyms_class_get_method(self):
        """ test Synonyms class' get method """

        # term with synonyms
        expected_output = "Mars|mars|martian|marsquakes|martians|martain|marslike|circummartian|marsshine"
        result = self.match_excerpt.synonyms.get('Mars')
        self.assertEqual(result, expected_output)

        # term without synonyms
        result = self.match_excerpt.synonyms.get('Vesta')
        self.assertEqual(result, 'Vesta')

    def test_unicode_class_replace_control_chars_method(self):
        """ test Unicode class' replace_control_chars method give string with controls chars """

        text = "Hello\x00World\x1fTest\x7fExample\x9f"
        expected_output = "Hello World Test Example"
        result = self.match_excerpt.unicode.replace_control_chars(text)
        self.assertEqual(result, expected_output)

    def test_regexresult_class_false_include(self):
        """ test RegExResult class when include is set to False """

        # match entity `Rayleigh` with at least 3 tokens on each side
        rgx = regex.compile(RegExPattern % ('Rayleigh', 3, 'Rayleigh', 3))

        # Test case 1: having enough tokens at both ends
        text = "one two three four five Rayleigh, this is for when the entity is at the beginning, " \
               "and now when the entity is at the end, Rayleigh five four three two one."
        results = [RegExResult(m, 'Mars|Crater') for m in rgx.finditer(text)]
        self.assertEqual(len(results), 2)

        # Test case 2: not having enough tokens at least in one end
        text = "Rayleigh, this is for when the entity is at the beginning, " \
               "and now when the entity is at the end, Rayleigh."
        results = [RegExResult(m, 'Mars|Crater') for m in rgx.finditer(text)]
        self.assertEqual(len(results), 0)

        # Test case 3: ignore if there is hyphen before or after the target token
        text = "one two three four five Rayleigh-this is for when the entity is at the beginning, " \
               "and now when the entity is at the end-Rayleigh five four three two one."
        results = [RegExResult(m, 'Mars|Crater') for m in rgx.finditer(text)]
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].include)
        self.assertFalse(results[1].include)

        # Test case 4: ignore if before or after token is capitalized, except exception tokens before the entity
        text = "one two three four five Rayleigh This is for when the entity is at the beginning, " \
               "and now when the entity is at the end Rayleigh Five four three two one. " \
               "one two three four five The Rayleigh this is for when the entity is at the beginning, " \
               "and now when the entity is at the end For Rayleigh five four three two one. " \
               "one two three four five Mars Rayleigh this is for when the entity is at the beginning, " \
               "and now when the entity is at the end Crater Rayleigh five four three two one. " \
               "one two three four five Rayleigh Mars this is for when the entity is at the beginning, " \
               "and now when the entity is at the end Rayleigh Crater five four three two one. "
        results = [RegExResult(m, capitalized_entities='Mars|Crater') for m in rgx.finditer(text)]
        self.assertEqual(len(results), 8)
        self.assertFalse(results[0].include)    # `This` capital before Rayleigh -> ignore
        self.assertFalse(results[1].include)    # `Five` capital after Rayleigh -> ignore
        self.assertTrue(results[2].include)     # `The` before is exception -> include
        self.assertTrue(results[3].include)     # `For` before is exception -> include
        self.assertTrue(results[4].include)     # `Mars` (passed in) before is exception -> include
        self.assertTrue(results[5].include)     # `Crater` (passed in) before is exception -> include
        self.assertTrue(results[6].include)     # `Mars` (passed in) after is exception -> include
        self.assertTrue(results[7].include)     # `Crater` (passed in) after is exception -> include

        # Test case 5: ignore if after starts with apostrophe or digit
        text = "one two three four five Rayleigh's this is for when the entity is at the beginning, " \
               "and now when the entity is at the end Rayleigh 11 five four three two one. "
        results = [RegExResult(m, capitalized_entities='Mars|Crater') for m in rgx.finditer(text)]
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].include)
        self.assertFalse(results[1].include)

        # Test case 6: raises an exception
        match = MagicMock()
        match.expandf.return_value = 'something'
        match.starts.side_effect = IndexError()
        match.ends.side_effect = IndexError()
        result = RegExResult(match, 'entity1|entity2')
        self.assertFalse(result.include)


if __name__ == '__main__':
    unittest.main()
