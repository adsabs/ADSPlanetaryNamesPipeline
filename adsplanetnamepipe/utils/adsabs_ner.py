import os
import regex
from typing import Tuple

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import TokenClassificationPipeline

from adsplanetnamepipe.utils.common import EntityArgs


# configuration and initialization of the AstroBERT Named Entity Recognition (NER) model
# includes setting the model path, loading the pre-trained model and tokenizer,
# and creating a TokenClassificationPipeline for NER tasks specific to astronomical text
# note: attempted to load the model locally, but due to GitHub file size limitations,
# the large binary file couldn't be included
# I think for production, this should be loaded from a local path
model_path = 'adsabs/astroBERT'
# model_path = os.path.dirname(__file__) + '/astrobert_ner_files'
model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=model_path, revision='NER-DEAL')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, add_special_tokens=True, do_lower_case=False, model_max_length=130)
adsabs_ner = TokenClassificationPipeline(model=model, tokenizer=tokenizer, task='astroBERT NER_DEAL', aggregation_strategy='average', ignore_labels=['O'])


class ADSabsNER():

    """
    a class that implements named entity recognition using the AstroBERT model

    this class uses a pre-trained AstroBERT model to identify and classify named entities
    in astronomical texts, with a focus on celestial objects and regions. It also includes
    methods to detect citations and references that might be misclassified as entities.
    """

    # class-level reference to the global AstroBERT named entity recognition object
    adsabs_ner = adsabs_ner

    # regular expression pattern to match author names in various formats
    author_pattern = r"((?:[A-Z][A-Za-z'`-]+)?(?:,?\s+(?:(?:van|von|de|der)\s+)?[A-Z][A-Za-z'`-]+)*(?:,?\s+(?:Jr\.|Sr\.|I{1,3}V?|IV|V|VI{1,3}))?\s*)"
    # pattern to match additional authors (e.g., "and", "&", "et al.")
    extras_pattern = r"(?:,? (?:(?:and|&)?\s*%s|(?:et\s*al\s*\.?\s*)),?)*\s*" % author_pattern
    # pattern to match publication years in various formats
    year_pattern = "(?:[12][09]\d\d[a-z]{0,1}|\([12][09]\d\d[a-z]{0,1}\))"
    # compiled regular expression to match citations in text
    re_citition = regex.compile(r"[\(\[]*(?:(%s%s%s|%s%s))[\)\]]*" % (author_pattern, extras_pattern, year_pattern,
                                                                      author_pattern, year_pattern))
    # pattern to match references in text
    reference_pattern = r"(%s[,\s]+[A-Z]\.?\s+|[A-Z]+\.?\s%s)"

    def __init__(self, args: EntityArgs):
        """
        initialize the ADSabsNER class

        :param args: configuration arguments for entity recognition
        """
        self.args = args

    def forward(self, text: str, feature_name_span: Tuple[int, int]) -> bool:
        """
        perform named entity recognition on the given text

        these are astrobert NER tags
            Archive,
            CelestialObject, CelestialObjectRegion, CelestialRegion,
            Citation, Collaboration, ComputingFacility, Database, Dataset, EntityOfFutureInterest, Event,
            Fellowship, Formula, Grant, Identifier, Instrument, Location, Mission, Model,
            ObservationalTechniques, Observatory, Organization, Person, Proposal, Software, Survey,
            Tag, Telescope, TextGarbage, URL, Wavelength
        we want the entities tagged as CelestialObject, CelestialObjectRegion, CelestialRegion
        or entities not tagged at all

        :param text: the text to analyze
        :param feature_name_span: tuple containing the start and end indices of the feature name in the text
        :return: boolean indicating whether the feature name is a valid celestial object or region
        """
        try:
            results = self.adsabs_ner(text)
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

    def is_citation_or_reference(self, text: str, feature_name_span: Tuple[int, int]) -> bool:
        """
        check if the feature name is part of a citation or reference

        most citations should be identified astrobert ner model
        lets caught the ones that it missed
        to match citations of the following formats
            ...markedly with Uranus season ( Alexander, 1965 ) .
            ... crystal field transitions ( Adams and McCord, 1971 ).

        :param text: the text to analyze
        :param feature_name_span: tuple containing the start and end indices of the feature name in the text
        :return: boolean indicating whether the feature name is part of a citation or reference
        """
        for match in self.re_citition.finditer(text):
            if feature_name_span[0] >= match.start() and feature_name_span[1] <= match.end():
                return True
        match = regex.search(self.reference_pattern % (self.args.feature_name, self.args.feature_name), text)
        if match:
            if feature_name_span[0] >= match.start() and feature_name_span[1] <= match.end():
                return True
        return False
