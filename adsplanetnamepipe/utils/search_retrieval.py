import requests
import re

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

from adsplanetnamepipe.utils.common import EntityArgs


class SearchRetrieval():

    # per Mike, for now, querying planetary journals only for the identification step
    planetry_journal_filter = ' bibstem:("%s") ' % '" OR "'.join(['Icar', 'JGRE', 'P&SS', 'M&PS', 'E&PSL', 'SSRv', 'PSJ'])

    # for collecting usgs term
    astronomy_journal_filter = ' bibstem:("%s") ' % '" OR "'.join([
        "A&A", "AdSpR", "ApJ", "ApJL", "ApJS", "ChGeo", "E&PSL", "EM&P", "GeCoA", "Geo", "GeoJI", "GeoRL", "GGG", "Icar",
        "JGRA", "JGRB", "JGRE", "LPI", "M&PS", "MNRAS", "NatGe", "Natur", "P&SS", "PASJ", "PASP", "PEPI", "PSJ", "Sci",
        "SSRv"
    ])

    other_filters = 'database:astronomy fulltext_mtime:["2000-01-01t00:00:00.000Z" TO *]'

    re_replace_html = [
        (re.compile(r'<i>|</i>|<b>|</b>', flags=re.IGNORECASE), ''),
        (re.compile(r'<'), '&#60;'),
        (re.compile(r'>'), '&#62;')
    ]
    re_references = re.compile(r'.(?=References[\W\s]*[A-Z\[\(0-9]+)')

    def __init__(self, args: EntityArgs):
        """

        :param args:
        """

        self.args = args
        self.feature_types_ored = '" OR "'.join([self.args.feature_type, self.args.feature_type_plural]) if self.args.feature_type_plural \
                                                                                                         else self.args.feature_type

    def single_solr_query(self, start, rows, query):
        """

        :param start:
        :param rows:
        :param query:
        :return:
        """
        params = {
            'q': query,
            'start': start,
            'rows': rows,
            'sort': 'bibcode desc',
            'fl': 'bibcode, title, abstract, body, database, keyword',
        }

        try:
            response = requests.get(
                url=config['PLANETARYNAMES_PIPELINE_SOLR_URL'],
                params=params,
                headers={'Authorization': 'Bearer %s' % config['PLANETARYNAMES_PIPELINE_ADSWS_API_TOKEN']},
                timeout=60
            )
            if response.status_code == 200:
                # make sure solr found the documents
                from_solr = response.json()
                if (from_solr.get('response')):
                    docs = from_solr['response']['docs']
                    for doc in docs:
                        # replace any html entities in all fields
                        for field in ['title', 'abstract', 'body']:
                            if field in doc:
                                field_str = doc.get(field)
                                if isinstance(field_str, list):
                                    for (compiled_re, replace_str) in self.re_replace_html:
                                        field_str[0] = compiled_re.sub(replace_str, field_str[0])
                                elif isinstance(field_str, str):
                                    for (compiled_re, replace_str) in self.re_replace_html:
                                        field_str = compiled_re.sub(replace_str, field_str)
                                doc[field] = field_str
                        # attempt to remove the references section from the body
                        body_split = self.re_references.split(doc.get('body', ''))
                        if len(body_split) > 1:
                            doc['body'] = ' '.join(body_split[:-1])
                    return docs, 200
            return None, response.status_code
        except requests.exceptions.RequestException as e:
            return None, e

    def solr_query(self, query):
        """

        :param query:
        :return:
        """
        index = 1
        rows = 2000

        # go through the loop and get 2000 records at a time
        docs = []
        while True:
            docs_from_solr, status_code = self.single_solr_query(start=index, rows=rows, query=query)
            if status_code == 200:
                if len(docs_from_solr) > 0:
                    docs += docs_from_solr
                else:
                    logger.info(f"Got {len(docs)} docs from solr.")
                    break
                index += rows
            else:
                logger.error(f"From solr status code {status_code}.")
                break
        return docs

    def identify_terms_query(self):
        """

        :return:
        """
        query = f'full:(="{self.args.feature_name}") full:("{self.args.target}") full:("{self.feature_types_ored}") year:[2000 TO *]'
        query += self.planetry_journal_filter + self.other_filters

        return self.solr_query(query)

    def collect_usgs_terms_query(self):
        """

        :return:
        """
        query = f'full:("{self.args.feature_name}") full:("{self.args.target}") full:("{self.feature_types_ored}") '
        query += self.other_filters

        # multi level query
        # see if can extract enough records with top filters, if not keep removing filters
        # until either get enough records, or run out of filters
        queries = [
            f'{query} {self.astronomy_journal_filter} property:refereed year:[2000 TO *]', # all conditions included
            f'{query} property:refereed year:[2000 TO *]', # without top astronomy journals
            f'{query} property:refereed', # without top astronomy journals and year
            query  # bare-bone query
        ]

        docs = []
        for query, min_num_docs in zip(queries, [5, 5, 5, 1]):
            docs = self.solr_query(query)
            if len(docs) >= min_num_docs:
                return docs
        if len(docs) == 0:
            logger.error(f"Unable to get data from solr for {self.args.feature_name}/{self.args.target}.")
        return docs

    def collect_non_usgs_terms_query(self):
        """

        :return:
        """
        # exclude any record with mention of any of the celestial bodies
        all_targets = '" OR "'.join(self.args.all_targets)
        query = f'full:(="{self.args.feature_name}") -full:("{all_targets}") '
        query += f'-{self.other_filters}'

        # 200 is enough for the negative side
        docs, status_code = self.single_solr_query(query=query, start=1, rows=500)
        if status_code == 200:
            return docs
        else:
            logger.error(f"Error querying non usgs terms, got status code: {status_code} from solr.")
        return []



