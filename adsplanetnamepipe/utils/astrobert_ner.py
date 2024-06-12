import os
import regex

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import TokenClassificationPipeline

from adsplanetnamepipe.utils.common import EntityArgs


model_path = 'adsabs/astroBERT'
# model_path = os.path.dirname(__file__) + '/astrobert_ner_files'
model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=model_path, revision='NER-DEAL')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, add_special_tokens=True, do_lower_case=False, model_max_length=130)
astrobert_ner = TokenClassificationPipeline(model=model, tokenizer=tokenizer, task='astroBERT NER_DEAL', aggregation_strategy='average', ignore_labels=['O'])


class AstroBERTNER():

    astrobert_ner = astrobert_ner

    author_pattern = r"((?:[A-Z][A-Za-z'`-]+)?(?:,?\s+(?:(?:van|von|de|der)\s+)?[A-Z][A-Za-z'`-]+)*(?:,?\s+(?:Jr\.|Sr\.|I{1,3}V?|IV|V|VI{1,3}))?\s*)"
    extras_pattern = r"(?:,? (?:(?:and|&)?\s*%s|(?:et\s*al\s*\.?\s*)),?)*\s*" % author_pattern
    year_pattern = "(?:[12][09]\d\d[a-z]{0,1}|\([12][09]\d\d[a-z]{0,1}\))"
    re_citition = regex.compile(r"[\(\[]*(?:(%s%s%s|%s%s))[\)\]]*" % (author_pattern, extras_pattern, year_pattern,
                                                                      author_pattern, year_pattern))

    reference_pattern = r"(%s[,\s]+[A-Z]\.?\s+|[A-Z]+\.?\s%s)"

    def __init__(self, args: EntityArgs):
        """

        :param args:
        """
        self.args = args

    def forward(self, text, feature_name_span):
        """
        these are astrobert NER tags
            Archive,
            CelestialObject, CelestialObjectRegion, CelestialRegion,
            Citation, Collaboration, ComputingFacility, Database, Dataset, EntityOfFutureInterest, Event,
            Fellowship, Formula, Grant, Identifier, Instrument, Location, Mission, Model,
            ObservationalTechniques, Observatory, Organization, Person, Proposal, Software, Survey,
            Tag, Telescope, TextGarbage, URL, Wavelength
        we want the ones tagged: CelestialObject, CelestialObjectRegion, CelestialRegion
        or not tagged at all

        :param text:
        :param feature_name_span:
        :return:
        """
        try:
            results = self.astrobert_ner(text)
        except RuntimeError:
            logger.error('AstroBERT NER throw RuntimeError.')
            return False

        for i, result in enumerate(results):
            if self.args.feature_name in result['word'] and result['start'] <= feature_name_span[0] and result['end'] >= feature_name_span[1]:
                # if token has been recognized as anything but these, it is not usgs term
                if result['entity_group'] not in ['CelestialObject', 'CelestialObjectRegion', 'CelestialRegion']:
                    return False
        if self.is_citation_or_reference(text, feature_name_span):
            return False
        # if it was not recognized or it was recognized as Celestial then, consider it for further processing
        return True

    def is_citation_or_reference(self, text, feature_name_span):
        """
        most citations should be identified astrobert ner model
        lets caught the ones that it missed
        to match citations of the following formats
            ...markedly with Uranus season ( Alexander, 1965 ) .
            ... crystal field transitions ( Adams and McCord, 1971 ).

        :param text:
        :param feature_name_span:
        :return:
        """
        for match in self.re_citition.finditer(text):
            if feature_name_span[0] >= match.start() and feature_name_span[1] <= match.end():
                return True
        match = regex.search(self.reference_pattern % (self.args.feature_name, self.args.feature_name), text)
        if match:
            if feature_name_span[0] >= match.start() and feature_name_span[1] <= match.end():
                return True
        return False

