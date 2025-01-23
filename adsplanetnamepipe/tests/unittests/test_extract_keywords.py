import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)


import unittest
from unittest.mock import MagicMock, patch

from adsplanetnamepipe.utils.extract_keywords import ExtractKeywords
from adsplanetnamepipe.utils.common import EntityArgs, Synonyms

from adsplanetnamepipe.tests.unittests.stubdata import solrdata
from adsplanetnamepipe.tests.unittests.stubdata import excerpts
from adsplanetnamepipe.tests.unittests.stubdata import nasadata


class TestExtractKeywords(unittest.TestCase):

    """
    Tests the extract keywords module
    """

    def setUp(self):
        """ Set up the config class and create an instance of ExtractKeywords """

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
        self.extract_keywords = ExtractKeywords(self.args)

    def test_forward(self):
        """ test forward method -- extracting keywords from excerpt """

        # test when there are common keywords between spacy and yake, and also there are wikidata keywords
        self.extract_keywords.spacy.extract_top_keywords = MagicMock(return_value=['eolian', 'Mars', 'Rayleigh', 'Discovery'])
        self.extract_keywords.yake.extract_top_keywords = MagicMock(return_value=['crater', 'Mars', 'past', 'ripple', 'bed', 'formed', 'evidence', 'dune'])
        self.extract_keywords.wiki.extract_top_keywords = MagicMock(return_value=['Rayleigh', 'crater', 'Meridiani Planum'])
        expected_keywords = ['ripple', 'mars', 'discovery', 'eolian', 'meridiani planum', 'crater', 'formed', 'evidence', 'past', 'dune', 'bed']
        result = self.extract_keywords.forward(excerpts.doc_1_excerpts[0]['excerpt'])
        self.assertEqual(sorted(result), sorted(expected_keywords))

        # test when there are no yake and wikidata
        self.extract_keywords.spacy.extract_top_keywords = MagicMock(return_value=['eolian', 'Rayleigh', 'Discovery', 'crater', 'Mars', 'past'])
        self.extract_keywords.yake.extract_top_keywords = MagicMock(return_value=[])
        self.extract_keywords.wiki.extract_top_keywords = MagicMock(return_value=[])
        expected_keywords = ['crater', 'discovery', 'eolian', 'mars', 'past']
        result = self.extract_keywords.forward(excerpts.doc_1_excerpts[0]['excerpt'], num_keywords=5)
        self.assertEqual(sorted(result), sorted(expected_keywords))

        # test when there are no spacy and wikidata
        self.extract_keywords.spacy.extract_top_keywords = MagicMock(return_value=[])
        self.extract_keywords.yake.extract_top_keywords = MagicMock(return_value=['eolian', 'Rayleigh', 'Discovery', 'crater', 'Mars', 'past'])
        self.extract_keywords.wiki.extract_top_keywords = MagicMock(return_value=[])
        expected_keywords = ['crater', 'discovery', 'eolian', 'mars', 'past']
        result = self.extract_keywords.forward(excerpts.doc_1_excerpts[0]['excerpt'], num_keywords=5)
        self.assertEqual(sorted(result), sorted(expected_keywords))

    @patch('adsplanetnamepipe.utils.extract_keywords.logger')
    def test_forward_returning_empty_list(self, mock_logger):
        """ test forward method -- when any of the keyword extractors fail on verify """

        self.extract_keywords.spacy.extract_top_keywords = MagicMock(return_value=['eolian', 'Mars', 'Rayleigh scattering', 'Discovery'])
        result = self.extract_keywords.forward(excerpts.doc_3_excerpts[3]['excerpt'])
        self.assertEqual(result, [])
        mock_logger.info.assert_called_with('SpaCy identified a phrase that included feature name. Excerpt filtered out.')

        self.extract_keywords.spacy.extract_top_keywords = MagicMock(return_value=['eolian', 'Mars', 'Rayleigh', 'Discovery'])
        self.extract_keywords.yake.extract_top_keywords = MagicMock(return_value=['crater', 'Mars', 'past', 'Rayleigh ripple', 'bed', 'formed', 'evidence', 'dune'])
        result = self.extract_keywords.forward(excerpts.doc_3_excerpts[3]['excerpt'])
        self.assertEqual(result, [])
        mock_logger.info.assert_called_with('Yake identified a phrase that included feature name. Excerpt filtered out.')

        self.extract_keywords.spacy.extract_top_keywords = MagicMock(return_value=['eolian', 'Mars', 'Rayleigh', 'Discovery'])
        self.extract_keywords.yake.extract_top_keywords = MagicMock(return_value=['crater', 'Mars', 'past', 'ripple', 'bed', 'formed', 'evidence', 'dune'])
        self.extract_keywords.wiki.extract_top_keywords = MagicMock(return_value=['Rayleigh distribution', 'crater', 'Meridiani Planum'])
        result = self.extract_keywords.forward(excerpts.doc_3_excerpts[3]['excerpt'])
        self.assertEqual(result, [])
        mock_logger.info.assert_called_with('Wikidata keyword has a phrase that includes feature name. Excerpt filtered out.')

        self.extract_keywords.spacy.extract_top_keywords = MagicMock(return_value=['eolian'])
        self.extract_keywords.yake.extract_top_keywords = MagicMock(return_value=['crater', 'Mars'])
        self.extract_keywords.wiki.extract_top_keywords = MagicMock(return_value=[])
        result = self.extract_keywords.forward(excerpts.doc_3_excerpts[3]['excerpt'], num_keywords=5)
        self.assertEqual(result, [])
        mock_logger.info.assert_called_with('Not enough keywords were extracted. Excerpt filtered out.')

    def test_forward_doc(self):
        """ test forward_doc method -- extracting keywords from the fulltext """

        tfidf_top_keywords = [
             'crater', 'rayleigh', 'ripple', 'figure', 'exposed', 'rockingham', 'drake', 'north', 'diameter',
             'diligence', 'interior', 'banding', 'small', 'image', 'layer', 'opportunity', 'superposed', 'rim',
             'bedrock', 'sandy', 'hirise', 'sol', 'smaller'
        ]
        self.extract_keywords.tfidf.extract_top_keywords = MagicMock(return_value=tfidf_top_keywords)
        vocabulary = Synonyms().add_synonyms([self.args.target,
                                              self.args.feature_type, self.args.feature_type_plural,
                                              self.args.feature_name])
        expected_keywords = [
            'crater', 'rayleigh', 'ripple', 'figure', 'exposed', 'rockingham', 'drake', 'north', 'diameter',
            'diligence', 'interior', 'banding', 'small', 'image', 'layer', 'opportunity', 'superposed', 'rim',
            'bedrock', 'sandy'
        ]
        result = self.extract_keywords.forward_doc(solrdata.doc_1, vocabulary, usgs_term=True)
        self.assertEqual(sorted(result), sorted(expected_keywords))

        # now tell it this is non usgs term, eventhough the entity in excerpt is actually usgs entity
        result = self.extract_keywords.forward_doc(solrdata.doc_1, vocabulary, usgs_term=False)
        self.assertEqual(result, [])

        # now tell it this is non usgs term, and it is correct
        self.extract_keywords.tfidf.extract_top_keywords = MagicMock(return_value=['ripple', 'figure', 'exposed', 'rockingham', 'drake'])
        result = self.extract_keywords.forward_doc(solrdata.doc_3, vocabulary, usgs_term=False)
        self.assertEqual(sorted(result), sorted(['ripple', 'figure', 'exposed', 'rockingham', 'drake']))

    def test_spacy_extract_top_keywords(self):
        """ test SpacyWrapper's extract_top_keywords """

        expected_keywords = ['Drake', 'Rockingham', 'Navcam', 'Pancam', 'Rayleigh']
        result = self.extract_keywords.spacy.extract_top_keywords(excerpts.doc_1_excerpts[3]['excerpt'])
        self.assertEqual(sorted(result), sorted(expected_keywords))

    def test_yake_extract_top_keywords(self):
        """ test YakeWrapper's extract_top_keywords """

        expected_keywords = ['east', 'face', 'layer', 'west', 'Drake', 'dip', 'Rayleigh', 'Rockingham',
                             'made', 'crater', 'showing', 'banding', 'ripple']
        result = self.extract_keywords.yake.extract_top_keywords(excerpts.doc_1_excerpts[3]['excerpt'])
        self.assertEqual(sorted(result), sorted(expected_keywords))

    def test_wiki_extract_top_keywords(self):
        """ test WikiWrapper's extract_top_keywords """

        expected_keywords = ['Opportunity', 'Rayleigh', 'crater']
        result = self.extract_keywords.wiki.extract_top_keywords(excerpts.doc_1_excerpts[3]['excerpt'])
        self.assertEqual(sorted(result), sorted(expected_keywords))

    def test_wiki_extract_top_keywords_single_match(self):
        """ test when there is only one match with wiki """

        # create a mock instance of regex
        mock_regex = MagicMock()
        mock_regex.findall.return_value = ["single_match"]

        # replace the real regex instance with the mock
        self.extract_keywords.wiki.re_wiki_vocab = mock_regex

        result = self.extract_keywords.wiki.extract_top_keywords(excerpts.doc_1_excerpts[1]['excerpt'])

        self.assertEqual(result, ["single_match"])
        mock_regex.findall.assert_called_once_with(excerpts.doc_1_excerpts[1]['excerpt'])

    def test_tfidf_extract_top_keywords(self):
        """ test TfidfWrapper's extract_top_keywords """

        expected_keywords = ['crater', 'rayleigh', 'ripple', 'figure', 'exposed', 'rockingham', 'drake', 'north',
                             'diameter', 'diligence', 'interior', 'banding', 'small', 'image', 'layer', 'opportunity',
                             'superposed', 'rim', 'bedrock', 'sandy', 'hirise', 'sol', 'smaller']
        result = self.extract_keywords.tfidf.extract_top_keywords(solrdata.doc_1)
        self.assertEqual(sorted(result), sorted(expected_keywords))

    def test_validate_feature_name_spacy(self):
        """ test SpacyWrapper's validate_feature_name """

        args = EntityArgs(
            target="Moon",
            feature_type="Crater",
            feature_type_plural="Craters",
            feature_name="Green",
            context_ambiguous_feature_names=[],
            multi_token_containing_feature_names=[],
            name_entity_labels=[{'label': 'planetary', 'value': 1}, {'label': 'non planetary', 'value': 0}],
            timestamp='2000-01-01',
            all_targets = []
        )
        text = "study are formed within basalt flows (boundaries drawn in white, Ramsey et al., ) whose sources are " \
               "off-map to the southwest. Focused caves in this work include Skull Cave (Figure top) and Valentine " \
               "Cave (Figure bottom). 4 Maps of GPR and LiDAR surveys of two lava tubes at LBNM: (top) Skull Cave; " \
               "(bottom) Valentine Cave. Selected GPR survey lines are mapped over each cave. Green polygons " \
               "represent terrestrial LiDAR scan (TLS) coverage of the tubes' interiors. At Skull, small holes in the " \
               "coverage polygon are TLS coverage gaps; at Valentine, larger holes are pillars within the tube " \
               "(see Figure ). Grid coordinates given in UTM zone 10 N. 5 LiDAR data of Skull Cave (LBNM) that shows a " \
               "side view of the surface and two different mapped levels. Ice forms"
        span = (376, 381)
        result = self.extract_keywords.spacy.validate_feature_name(text, args, span, True)
        self.assertFalse(result)

        # send text staring with `Green` entity
        result = self.extract_keywords.spacy.validate_feature_name(text[span[0]:], args, (0, len('Green')), True)
        self.assertFalse(result)

        # send text ending with `Green` entity
        result = self.extract_keywords.spacy.validate_feature_name(text[:span[1]], args, (span[1]-len('Green'), span[1]), True)
        self.assertFalse(result)

    def test_validate_feature_name_yake(self):
        """ test YakeWrapper's validate_feature_name """

        args = EntityArgs(
            target="Moon",
            feature_type="Crater",
            feature_type_plural="Craters",
            feature_name="Green",
            context_ambiguous_feature_names=[],
            multi_token_containing_feature_names=[],
            name_entity_labels=[{'label': 'planetary', 'value': 1}, {'label': 'non planetary', 'value': 0}],
            timestamp='2000-01-01',
            all_targets = []
        )
        text = "study are formed within basalt flows (boundaries drawn in white, Ramsey et al., ) whose sources are " \
               "off-map to the southwest. Focused caves in this work include Skull Cave (Figure top) and Valentine " \
               "Cave (Figure bottom). 4 Maps of GPR and LiDAR surveys of two lava tubes at LBNM: (top) Skull Cave; " \
               "(bottom) Valentine Cave. Selected GPR survey lines are mapped over each cave. Green polygons " \
               "represent terrestrial LiDAR scan (TLS) coverage of the tubes' interiors. At Skull, small holes in the " \
               "coverage polygon are TLS coverage gaps; at Valentine, larger holes are pillars within the tube " \
               "(see Figure ). Grid coordinates given in UTM zone 10 N. 5 LiDAR data of Skull Cave (LBNM) that shows a " \
               "side view of the surface and two different mapped levels. Ice forms"
        span = (376, 381)

        # send text staring with `Green` entity
        result = self.extract_keywords.yake.validate_feature_name(text[span[0]:], args, (0, len('Green')), True)
        self.assertFalse(result)

        # send text ending with `Green` entity
        result = self.extract_keywords.yake.validate_feature_name(text[:span[1]], args, (span[1] - len('Green'), span[1]), True)
        self.assertFalse(result)

    @patch('requests.post')
    def test_forward_special(self, mock_post):
        """ test forward_special method -- extracting sti-keywords """

        # mock the successful response from the API
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = nasadata.doc_1
        mock_post.return_value = mock_response

        expected_keywords = ['craters', 'slopes', 'surface properties', 'rayleigh scattering']
        result = self.extract_keywords.forward_special(excerpts.doc_1_excerpts[3]['excerpt'])
        self.assertEqual(sorted(result), sorted(expected_keywords))

    @patch('requests.post')
    def test_forward_fail(self, mock_post):
        """ test forward_special method -- when fails """

        # mock the failed response from the API
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = self.extract_keywords.forward_special(excerpts.doc_1_excerpts[3]['excerpt'])
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
