import regex

from adsplanetnamepipe.utils.common import EntityArgs, Synonyms


class PaperRelevance():

    def __init__(self, args: EntityArgs):
        """

        :param args:
        """
        self.synonyms = Synonyms()
        self.re_match_target = regex.compile(r'\b(%s)\b' % self.synonyms.get_target_terms(args.target), flags=regex.IGNORECASE)
        self.re_match_feature_type = regex.compile(r'\b(%s)\b' % self.synonyms.get_feature_type_terms([args.feature_type, args.feature_type_plural]), flags=regex.IGNORECASE)

    def forward(self, text, bibstem, databases, astronomy_main_journals, len_existing_wikidata):
        """

        :param text:
        :param bibstem:
        :param databases:
        :param astronomy_main_journals:
        :param len_existing_wikidata:
        :return:
        """
        # threshold for num of targets, feature type, and wikidata terms
        threshold = text.count(' ') * 0.001

        # get number of times target and feature type appeared in the text,
        # each is worth 0.2 in scoring if above threshold
        target_terms_len = len(self.re_match_target.findall(text))
        feature_type_terms_len = len(self.re_match_feature_type.findall(text))

        in_astronomy_main_journals = any(journal == bibstem for journal in astronomy_main_journals)

        # the threshold and weights have been determined empirically, by analyzing several thousands scores
        return int('astronomy' in databases) * 0.2 + \
               int(in_astronomy_main_journals) * 0.2 + \
               int(target_terms_len > threshold) * 0.2 + \
               int(feature_type_terms_len > 0) * 0.2 + \
               int(len_existing_wikidata > threshold) * 0.2

