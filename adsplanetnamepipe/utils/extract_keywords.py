import os
import regex
import math
from collections import OrderedDict
import requests
from requests.exceptions import RequestException
from typing import List, Dict, Tuple, Set

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

import spacy
spacy_model = spacy.load("en_core_web_lg")

import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from adsplanetnamepipe.utils.common import EntityArgs


class SpacyWrapper():

    """
    a wrapper class for Spacy NLP operations

    this class provides methods for extracting phrases, validating feature names,
    and extracting keywords using Spacy NLP model
    """

    # class-level reference to the global spacy en_core_web_lg model
    model = spacy_model

    def extract_phrases(self, annotated_text, args: EntityArgs) -> List[str]:
        """
        extract noun phrases from annotated text that contain the feature name

        :param annotated_text: Spacy-annotated text
        :param args: EntityArgs object containing feature information
        :return: list of extracted noun phrases
        """

        def all_nouns(phrase: str) -> str:
            """
            extract all nouns and proper nouns from a given phrase

            :param phrase: input phrase to analyze
            :return: string containing all tokens that are either NOUN or PROPN, separated by spaces
            """
            all_noun_tokens = ''
            annotated = self.model(phrase)
            if len(annotated) > 1:
                for token in annotated:
                    if token.pos_ in ['NOUN', 'PROPN']:
                        all_noun_tokens += ' %s'%token.text
            return all_noun_tokens.strip()

        # spacy phrase extraction
        phrases = [chunk.text for chunk in annotated_text.noun_chunks]

        feature_name = args.feature_name.lower()
        noun_phrases = []
        for phrase in phrases:
            if feature_name in phrase.lower() and feature_name != phrase.strip().lower():
                noun_phrase_tokens = all_nouns(phrase)
                if noun_phrase_tokens:
                    noun_phrases.append(noun_phrase_tokens)
        return noun_phrases

    def validate_feature_name_adjective(self, annotated_text, args: EntityArgs) -> bool:
        """
        checks if the feature name has appeared as an adjective in the text,
        indicating that we cannot consider it as a valid USGS term
        for example, there are Black/Green/White entities that are moon craters

        :param annotated_text: Spacy-annotated text
        :param args: EntityArgs object containing feature information
        :return: True if feature name is not used as an adjective, False otherwise
        """
        feature_name_tag = [t.pos_ for t in annotated_text if t.text == args.feature_name]
        if len(feature_name_tag) > 0:
            if feature_name_tag[0] == 'ADJ':
                return False
        return True

    def validate_feature_name_phrase(self, annotated_text, args: EntityArgs) -> bool:
        """
        check if the feature name appears as part of a valid phrase in the text
        suggesting that we cannot consider it as a valid USGS term

        :param annotated_text: Spacy-annotated text
        :param args: EntityArgs object containing feature information
        :return: True if feature name is part of a valid phrase, False otherwise
        """
        # get phrases that contains feature name
        phrases = self.extract_phrases(annotated_text, args)
        if not phrases:
            return True

        identifiers = set(args.feature_type.lower().split(', ') + [args.target.lower()])
        # check the phrases that contain feature name is it with the identifiers, or another token
        # if it is another token, then feature name has another context
        # ie Kaiser Crater is OK, but Russell Kaiser is not
        part_of = [phrase for phrase in phrases if phrase.count(' ') > 0 and len(set(phrase.lower().split()).intersection(identifiers)) == 0]
        return len(part_of) == 0

    def validate_feature_name(self, text: str, args: EntityArgs, feature_name_span: Tuple[int, int], usgs_term: bool) -> bool:
        """
        validate the feature name in the given text context

        :param text: input text
        :param args: EntityArgs object containing feature information
        :param feature_name_span: tuple of start and end indices of feature name in text
        :param usgs_term: boolean indicating if it's a USGS term
        :return: True if feature name is a usgs term, False otherwise
        """
        annotated_text = self.model(text)

        if usgs_term:
            if not self.validate_feature_name_adjective(annotated_text, args):
                return False

            # need to have a few tokens before and after the feature_name
            # if does not exist, quit, since we need them to decide if the feature_name
            # has been identified as part of a phrase which makes it not usgs term
            before_feature = ' '.join(text[:feature_name_span[0]].strip().split(' ')[-4:])
            after_feature = ' '.join(text[feature_name_span[1]:].strip().split(' ')[:4])
            if not (before_feature and after_feature):
                return False

            if not self.validate_feature_name_phrase(annotated_text, args):
                return False

        # if either the term in the context of usgs is valid, or
        # it is considered to be in the context of non usgs, then should be valid
        return True

    def extract_top_keywords(self, text: str) -> List[str]:
        """
        extract top keywords from the text using Spacy NER

        :param text: input text
        :return: list of extracted keywords
        """
        annotated_text = self.model(text)

        entities = []
        for ent in annotated_text.ents:
            if ent.text.isalpha() and len(ent.text) >= 3:
                entities.append(ent.text)
        return list(set(entities))


class YakeWrapper():

    """
    a wrapper class for YAKE keyword extraction

    this class provides methods for extracting phrases, validating feature names,
    and extracting keywords using YAKE

    :param model: the YAKE keyword extractor model
    :param lemmatizer: WordNet lemmatizer for processing extracted keywords
    """

    # YAKE keyword extractor configured for English, bi-grams, with deduplication and limited to top 20 results
    model = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.5, dedupFunc='seqm', windowsSize=1, top=20, stopwords=None, features=None)
    # WordNet lemmatizer for reducing words to their base forms
    lemmatizer = WordNetLemmatizer()

    def extract_phrases(self, text: str, args: EntityArgs) -> List[str]:
        """
        extract phrases from text that contain the feature name

        :param text: input text
        :param args: EntityArgs object containing feature information
        :return: list of extracted phrases
        """
        # yake phrase extraction
        phrases = [token for token, _ in self.model.extract_keywords(text)]

        feature_name = args.feature_name.lower()
        phrases = [phrase for phrase in phrases if
                   feature_name in phrase.lower() and feature_name != phrase.strip().lower()]

        return phrases

    def validate_feature_name(self, text: str, args: EntityArgs, feature_name_span: Tuple[int, int], usgs_term: bool) -> bool:
        """
        validate the feature name in the given text context

        :param text: input text
        :param args: EntityArgs object containing feature information
        :param feature_name_span: tuple of start and end indices of feature name in text
        :param usgs_term: boolean indicating if it's a USGS term
        :return: True if feature name is a usgs term, False otherwise
        """
        if usgs_term:
            # need to have a few tokens before and after the feature_name
            # if does not exist, quit, since we need them to decide if the feature_name
            # has been identified as part of a phrase which makes it not usgs term
            before_feature = ' '.join(text[:feature_name_span[0]].strip().split(' ')[-4:])
            after_feature = ' '.join(text[feature_name_span[1]:].strip().split(' ')[:4])
            if not (before_feature and after_feature):
                return False

            if not self.validate_feature_name_phrase(text, args):
                return False

        # if either the term in the context of usgs is valid, or
        # it is considered to be in the context of non usgs, then should be valid
        return True

    def validate_feature_name_phrase(self, text: str, args: EntityArgs) -> bool:
        """
        check if the feature name appears as part of a valid phrase in the text
        suggesting that we cannot consider it as a valid USGS term

        :param text: input text
        :param args: EntityArgs object containing feature information
        :return: True if feature name is a valid usgs term (ie, not part of a phrase), False otherwise
        """
        # get phrases that contains feature name
        phrases = self.extract_phrases(text, args)
        if not phrases:
            return True

        identifiers = set(args.feature_type.lower().split(', ') + [args.target.lower()])
        # check the phrases that contain feature name is it with the identifiers, or another token
        # if it is another token, then feature name has another context
        # ie Kaiser Crater is OK, but Russell Kaiser is not
        part_of = [phrase for phrase in phrases if phrase.count(' ') > 0 and len(set(phrase.lower().split()).intersection(identifiers)) == 0]
        return len(part_of) == 0

    def extract_top_keywords(self, text: str) -> List[str]:
        """
        extract top keywords from the text using YAKE

        :param text: input text
        :return: list of extracted keywords
        """
        tokens = [self.lemmatizer.lemmatize(token) for token, _ in self.model.extract_keywords(text) if token.isalpha() and len(token) >= 3]
        return list(set(tokens))


class TfidfWrapper():

    """
    a wrapper class for TF-IDF based keyword extraction

    this class provides methods for extracting top keywords using TF-IDF
    """

    # TF-IDF vectorizer for converting text to a matrix of TF-IDF features
    vectorizer = TfidfVectorizer()
    # WordNet lemmatizer for reducing words to their base forms
    lemmatizer = WordNetLemmatizer()

    # set of English stop words from NLTK, adding in custom stop words
    stop_words = set(stopwords.words('english'))
    custom_stops = ['this', 'that', 'these', 'those',
                    'of', 'in', 'to', 'for', 'with', 'on', 'at',
                    'a', 'an', 'the',
                    'and', 'but', 'yet', 'so', 'for', 'nor', 'or',
                    'is', 'was', 'were', 'has', 'have', 'had',
                    'very', 'also', 'just', 'being', 'over', 'own', 'yours', 'such']
    stop_words.update(custom_stops)

    def extract_top_keywords(self, doc: Dict) -> List[str]:
        """
        extract top keywords from the document using TF-IDF

        :param doc: input document as a dictionary
        :return: list of extracted keywords
        """
        segments = self.get_segments(doc)
        tfidf_vectors = self.vectorizer.fit_transform(segments)
        tfidf_features = self.vectorizer.get_feature_names_out()
        top_tfidf = sorted(zip(tfidf_features, tfidf_vectors.sum(axis=0).tolist()[0]), key=lambda x: x[1], reverse=True)[:40]
        top_tfidf = [(p, s) for p, s in top_tfidf if p not in self.stop_words]

        # lemmatize and remove small entities, also make sure the entities are all alpha characters
        top_tfidf = [(self.lemmatizer.lemmatize(p), s) for p, s in top_tfidf if p.isalpha() and len(p) >= 3]

        # get the unique entities, keep the order
        unique_entities = OrderedDict.fromkeys(p for p, _ in top_tfidf)
        return list(unique_entities.keys())

    def get_segments(self, doc: Dict) -> List[str]:
        """
        split the document into segments for TF-IDF processing

        :param doc: input document as a dictionary
        :return: list of text segments
        """
        title = ' '.join(doc.get('title', ''))
        abstract = doc.get('abstract', '')

        segments = []
        if title:
            segments.append(title)
        if abstract:
            segments.append(abstract)

        body = doc.get('body', '')
        # split body into words
        tokens = body.split(' ')
        # now combine to get 20 segments
        length = len(tokens)
        chunks = math.ceil(length / 19)
        body_segments = []
        for i in list(range(length))[0::chunks]:
            body_segments.append(' '.join(tokens[i:i + chunks]))
        if body_segments:
            segments += body_segments

        return segments


class WikiWrapper():

    """
    a wrapper class for Wikipedia-based keyword extraction

    this class provides methods for extracting keywords based on a predefined Wikipedia vocabulary
    """

    def __init__(self):
        """
        initialize the WikiWrapper class by loading the Wikipedia vocabulary
        """
        file_path = os.path.dirname(__file__) + '/data_files/wiki_vocab.dat'
        with open(file_path, 'r') as file:
            elements = file.read().splitlines()

        self.wiki_vocab = '|'.join(elements)
        self.re_wiki_vocab = regex.compile(r'\b(%s)\b' % self.wiki_vocab)

    def extract_top_keywords(self, text: str) -> List[str]:
        """
        extract top keywords from the text based on Wikipedia vocabulary

        :param text: input text
        :return: list of extracted keywords
        """
        matches = []
        for match in self.re_wiki_vocab.findall(text):
            if isinstance(match, tuple):
                match = [item for item in match if item]
            else:
                match = [match]
            matches += match
        return list(set(matches))


class NASAWrapper():

    """
    a wrapper class for NASA concept extraction

    this class provides methods for extracting keywords using NASA's concept extraction API
    """

    def forward(self, excerpt: str) -> List[str]:
        """
        extract keywords from the excerpt using NASA's concept extraction API

        :param excerpt: input text excerpt
        :return: list of extracted keywords
        """
        try:
            url = config['PLANETARYNAMES_PIPELINE_NASA_CONCEPT_URL']
            payload = {
                "text": excerpt,
                "probability_threshold": 0.5,
                "topic_threshold": 1,
                "request_id": "example_request_id"
            }
            response = requests.post(url, json=payload)

            if response.status_code == 200:
                result = response.json()['payload']
                sti_keywords = result.get('sti_keywords',[[]])[0]

                # extract the 'unstemmed' from 'sti_keywords'
                sti_keywords = [kw['unstemmed'].lower() for kw in sti_keywords]
                return sti_keywords
            else:
                logger.error(f"From Nasa Concept status code {response.status_code}")
        except RequestException as e:
            # logger.info('Not hosted by ADS.')
            pass
        return []


class ExtractKeywords():

    """
    top level class to combine various keyword extraction methods

    this class integrates SpacyWrapper, YakeWrapper, WikiWrapper, TfidfWrapper, and NASAWrapper
    to provide comprehensive keyword extraction functionality
    """

    def __init__(self, args: EntityArgs):
        """
        initialize the ExtractKeywords class

        :param args: EntityArgs object containing feature information
        """
        self.args = args
        self.spacy = SpacyWrapper()
        self.yake = YakeWrapper()
        self.wiki = WikiWrapper()
        self.tfidf = TfidfWrapper()
        self.nasa = NASAWrapper()

    def forward(self, excerpt: str, num_keywords: int = 16) -> List[str]:
        """
        extract keywords from the excerpt using multiple methods

        three ways to identify top keywords from the excerpt and then merge
        1- entities identified by Spacy
        2- top keywords identified by yack
        3- match with astronomy object vocabs from wiki
        if any of the keywords is a phrase containing the feature_name, quit

        :param excerpt: input text excerpt
        :param num_keywords: number of keywords to extract
        :return: list of extracted keywords
        """
        feature_types = set([type.lower() for type in [self.args.feature_type, self.args.feature_type_plural]])
        feature_name = [self.args.feature_name.lower()]

        spacy_keywords = list(set([token.lower() for token in self.spacy.extract_top_keywords(excerpt) if token != self.args.feature_name]))
        if not self.verify(spacy_keywords, feature_name, feature_types):
            logger.info('SpaCy identified a phrase that included feature name. Excerpt filtered out.')
            return []
        yake_keywords = list(set([token.lower() for token in self.yake.extract_top_keywords(excerpt) if token != self.args.feature_name]))
        if not self.verify(yake_keywords, feature_name, feature_types):
            logger.info('Yake identified a phrase that included feature name. Excerpt filtered out.')
            return []
        wikidata_keywords = list(set([token.lower() for token in self.wiki.extract_top_keywords(excerpt) if token != self.args.feature_name]))
        if not self.verify(wikidata_keywords, feature_name, feature_types):
            logger.info('Wikidata keyword has a phrase that includes feature name. Excerpt filtered out.')
            return []

        # find tokens shared between spacy and yake
        spacy_yake_shared = []
        if spacy_keywords and yake_keywords:
            for keyword in yake_keywords.copy():
                for entity in spacy_keywords:
                    if keyword in entity:
                        spacy_yake_shared.append(keyword)
                        spacy_keywords.remove(entity)
                        yake_keywords.remove(keyword)
                        break

        # add remaining tokens from wikidata if available
        num_keywords_from_wiki = num_keywords - len(spacy_yake_shared)
        keywords = list(set(spacy_yake_shared + wikidata_keywords[:num_keywords_from_wiki]))

        # add more keywords to be returned if not have enough keywords yet
        if len(keywords) < num_keywords:
            keywords += wikidata_keywords[num_keywords_from_wiki:]
            keywords = list(set(keywords))

        # if still dont have enough keywords
        # first combine the two lists of spacy and yake, alternate between them, and then add them to the returned token list
        if len(keywords) < num_keywords:
            if spacy_keywords and yake_keywords:
                count = min(len(spacy_keywords), len(yake_keywords))
                combined = [''] * (count * 2)
                combined[::2] = spacy_keywords[:count]
                combined[1::2] = yake_keywords[:count]
                combined.extend(spacy_keywords[count:])
                combined.extend(yake_keywords[count:])
                keywords += combined
            elif spacy_keywords:
                keywords += spacy_keywords
            elif yake_keywords:
                keywords += yake_keywords
        keywords = list(set(keywords))[:num_keywords]

        # need to at least return 2/3 of keywords asked for, otherwise return nothing
        if len(keywords) >= num_keywords * 0.66:
            return keywords

        logger.info('Not enough keywords were extracted. Excerpt filtered out.')
        return []

    def forward_doc(self, doc: Dict, vocabulary: List[str], usgs_term: bool, num_keywords: int = 20) -> List[str]:
        """
        extract keywords from a document using TF-IDF

        :param doc: input document as a dictionary
        :param vocabulary: list of vocabulary terms
        :param usgs_term: boolean indicating if it's a USGS term
        :param num_keywords: number of keywords to extract
        :return: list of extracted keywords
        """
        re_vocabulary = regex.compile(r'(?i)\b(?:%s)\b' % '|'.join(vocabulary))
        tfidf_keywords = self.tfidf.extract_top_keywords(doc)
        count_matches = len(list(set([token for token in tfidf_keywords if re_vocabulary.search(token)])))
        # when usgs_term, planetary related, need at least one matched token to among the vocabulary
        # to get included for processing
        if usgs_term and count_matches >= 1:
            return tfidf_keywords[:num_keywords]
        # for non usgs_term, non planetary related, non of the matched tokens should be among the vocabulary
        # to get included for processing
        if not usgs_term and count_matches == 0:
            return tfidf_keywords[:num_keywords]
        return []

    def forward_special(self, excerpt: str) -> List[str]:
        """
        extract special keywords using NASA's concept extraction

        :param excerpt: input text excerpt
        :return: list of extracted special keywords
        """
        return self.nasa.forward(excerpt)

    def verify(self, keywords: List[str], feature_name: List[str], feature_types: Set[str]) -> bool:
        """
        verify if extracted keywords are valid in the context of feature name and types
        if both feature name and feature types are part of a keyword phrase or
        neither have appeared in the identified keyword phrases all good -> proceed,
        return True
        otherwise if feature name is part of a phrase with another token, that means
        the feature name has a context other than planetary that we are trying to id,
        in that case we have to case stop, return False

        :param keywords: list of extracted keywords
        :param feature_name: list containing the feature name
        :param feature_types: set of feature types
        :return: True if keywords are valid, False otherwise
        """
        for keyword in keywords:
            keyword_tokens = set([k.lower() for k in keyword.split()])
            # we need to compare multi token keywords
            if len(keyword_tokens) == 1:
                continue
            # is keyword either the feature name or feature types
            matched_feature_name = len(keyword_tokens.intersection(feature_name))
            matched_feature_types = len(keyword_tokens.intersection(feature_types))
            if matched_feature_name != matched_feature_types and matched_feature_name >= 1:
                return False
        return True