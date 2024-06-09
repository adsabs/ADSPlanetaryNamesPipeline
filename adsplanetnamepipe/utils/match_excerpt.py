import regex

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

from adsplanetnamepipe.utils.common import EntityArgs, Synonyms, Unicode
from adsplanetnamepipe.utils.extract_keywords import SpacyWrapper, YakeWrapper
from adsplanetnamepipe.utils.astrobert_ner import AstroBERTNER

from langdetect import detect

# need at least 1 token before and one token after the entity
RegExPattern = r"(?<entity>\b%s\b)(?<= (?<before>(?:(?<wdb>\w+)\W+){1,%d})\b%s\b(?=(?<after>(?:\W+(?<wda>\w+)){1,%d})))"

    
class RegExResult(object):

    def __init__(self, match, capitalized_entities):
        try:
            re_before = regex.compile(r'^((?!The|For|%s)[a-z]*[A-Z0-9\-\‐\–\/\[\(]+)$' % capitalized_entities)
            re_after = regex.compile(r"^((?!%s)[A-Z0-9\-\‐\–'’=\/\]\)]+.*)$" % capitalized_entities)

            # ignore if there is hyphen before or after the target token
            # ignore if before or after token is capitalized, except exception tokens before the entity
            # ignore if after starts with apostrophe or digit
            before = re_before.search(match.expandf('{before}').strip().split()[-1].strip())
            after = re_after.search(match.expandf('{after}').strip().split()[0].strip())

            # both have be none to proceed
            if not before and not after:
                # entity span within the text
                self.entity_span = match.span()
                # excerpt span within the text
                self.excerpt_span = (match.starts('before')[0], match.ends('after')[0])
                self.excerpt = match.expandf('{before}{entity}{after}')
                self.entity_span_within_excerpt = (self.entity_span[0] - self.excerpt_span[0], self.entity_span[1] - self.excerpt_span[0])
                self.include = True
            else:
                self.include = False
        except IndexError:
            # there is either no text before or after the identified target token
            self.include = False


class MatchExcerpt(object):

    synonyms_pattern = r'\b(%s)\b'

    def __init__(self, args: EntityArgs):
        """

        :param args:
        """

        self.synonyms = Synonyms()
        self.unicode = Unicode()
        self.spacy = SpacyWrapper()
        self.yake = YakeWrapper()
        self.args = args
        self.wnd = 64
        self.feature_types_and_target = '|'.join([item.capitalize() for item in ("%s, %s"%(self.args.feature_type, self.args.target)).split(', ')])

    def forward(self, doc, astrobert_ner: AstroBERTNER=None, usgs_term=True):
        """

        :param doc:
        :param astrobert_ner:
        :param usgs_term:
        :return:
        """
        # get the fulltext
        # only if it could not determine the language returns empty
        fulltext = self.get_fulltext(doc)
        if not fulltext:
            logger.info(f"Record `{doc['bibcode']}` is determined not to be in English. It is filtered out.")
            return False, []

        # if either in the followeing two phases:
        # identifying the entity, or collecting positive data
        # see if the feature name is ambiguous and determine if the doc is relevant to the feature name
        if usgs_term:
            relevant = self.determine_celestial_body_relevance(fulltext)
            if not relevant:
                logger.info(f"Record `{doc['bibcode']}` is determined not to be relevant for target {self.args.target}. Record filtered out.")
                return False, []

            relevant_excerpts = []
            excerpts = self.select_excerpts(fulltext)
            logger.info(f"For the record `{doc['bibcode']}` fetched {len(excerpts)} excerpts for further processing.")

            # for each excerpt if it is valid, in each step, keep it, otherwise filter it out
            for excerpt in excerpts:
                if self.validate_feature_name(excerpt.excerpt, excerpt.entity_span_within_excerpt, usgs_term):
                    if astrobert_ner.forward(excerpt.excerpt, excerpt.entity_span_within_excerpt):
                        relevant_excerpts.append(excerpt.excerpt)
                    else:
                        logger.info(f"An excerpt from the record `{doc['bibcode']}` is determined not relevant by AstroBERT NER. Record filtered out.")
                else:
                    logger.info(f"An excerpt from the record `{doc['bibcode']}` is determined not relevant by token/pharse analysis. Record filtered out.")

            logger.info(f"For record `{doc['bibcode']}` there are {len(relevant_excerpts)} relevant excerpts extracted in the step Match Excerpts.")
            return True, relevant_excerpts
        else:
            # if there are any mention of target or feature_name type in the fulltext
            # eliminate the record from the negative side
            if not self.is_context_non_planetary(fulltext):
                logger.info(f"Record `{doc['bibcode']}` is determined to be usgs relevant, and hence cannot be processed for non usgs phase.")
                return False, []
            # decided not collect excerpts for non usgs side, keywords are extracted from the entire document
            # but return true to proceed
            logger.info(f"For record `{doc['bibcode']}` determined it is not planetary record and hence keywords will be extacted from the fulltext in the next step.")
            return True, []

    def get_fulltext(self, doc):
        """

        :param doc:
        :return:
        """
        text = ' '.join(doc.get('title', '')) + ' ' + doc.get('abstract', '') + ' ' + doc.get('body', '')
        text_limit = str(text[:256].encode('utf-8').decode('ascii', 'ignore'))
        if not self.is_language_english(text_limit, doc['bibcode']):
            return ''
        return text

    def is_language_english(self, text, bibcode):
        """

        :param text:
        :param bibcode:
        :return:
        """
        try:
            language = detect(text)
            if language == 'en':
                return True
        except:
            logger.error(f"Unable to detect the language for `{bibcode}`. Concluding the fulltext is not in English and ignoring this record.")
        return False

    def determine_celestial_body_relevance(self, text):
        """
        return True if with high probability the text is about the celestial body specified 
        or if the probabilities are too close to determine
        return False otherwise to filtered out as irrelevant.

        :param text:
        :return:
        """
        # no ambigous context, we are done
        if not self.args.context_ambiguous_feature_names:
            return True

        ambiguous_context_count = []
        for term in self.args.context_ambiguous_feature_names:
            # regex is slow, so only use it if any of the terms exists in text
            if term in text:
                ambiguous_context_count.append(len(regex.findall(self.synonyms_pattern % self.synonyms.get(term), text, flags=regex.IGNORECASE)))
            else:
                # term not in the text, count is 0
                ambiguous_context_count.append(0)

        the_sum = sum(ambiguous_context_count)
        if the_sum > 0:
            # normalize
            ambiguous_context_count = [float(term) / the_sum for term in ambiguous_context_count]
            # sort both the terms and values ascending
            ambiguous_context_count, associated_terms = (list(t) for t in zip(
                *sorted(zip(ambiguous_context_count, self.args.context_ambiguous_feature_names), reverse=True)))
            # if more than one maximum value
            if ambiguous_context_count.count(ambiguous_context_count[0]) > 1:
                # let it go through, was not able to disambiguate here
                return True
            # if the highest probability is more than 1/number of terms and
            # the highest term matches the target or the feature type
            # then it is relevant, let it through
            # even if it is not, it shall be decided on the later stage
            if (associated_terms[0] == self.args.target or
                associated_terms[0] in [self.args.feature_type.lower(), self.args.feature_type_plural.lower()]) and \
               ambiguous_context_count[0] > 1 / len(associated_terms):
                return True
        # otherwise filter it out
        return False

    def select_excerpts(self, text):
        """
        for every instance of feature name in the text, a window of 129 tokens around it
        is identified for further analysis

        :param text:
        :return:
        """
        center = self.args.feature_name.replace(' ', '\s')
        # do case sensitive, per Alberto the USGS terms are Capitalized
        rgx = regex.compile(RegExPattern % (center, self.wnd, center, self.wnd))
        results = [RegExResult(m, self.feature_types_and_target) for m in rgx.finditer(text)]

        excerpts = []
        for result in results:
            if result.include:
                result.excerpt = self.unicode.replace_control_chars(result.excerpt)
                excerpts.append(result)

        # make sure excerpts are unique, and still keep the order
        seen = set()
        excerpts = [x for x in excerpts if not (x.excerpt in seen or seen.add(x.excerpt))]

        return excerpts

    def is_context_non_planetary(self, text):
        """
        this is called when processing non usgs record
        if there are any mention of target or feature type in the text
        then it cannot be used as a negative excerpt

        :param fulltext:
        :return: return True if can proceed with processing non usgs record
        """
        if self.args.target in text:
            return False
        if len(regex.findall(r'\b(%s)\b' % '|'.join([self.args.feature_type, self.args.feature_type_plural]), text, flags=regex.IGNORECASE)) > 0:
            return False
        return True

    def validate_feature_name(self, text, entity_span, usgs_term):
        """

        :param text:
        :param entity_span:
        :param usgs_term:
        :return:
        """

        valid = self.spacy.validate_feature_name(text, self.args, entity_span, usgs_term) and \
                self.yake.validate_feature_name(text, self.args, entity_span, usgs_term)

        return valid
