import requests
import re
from typing import List, Dict, Tuple

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

from adsplanetnamepipe.utils.common import EntityArgs


class SearchRetrieval():

    """
    a class that handles retrieval of scientific documents from a Solr search engine

    this class constructs and executes queries to retrieve documents related to
    planetary features, handles pagination, and processes the retrieved documents
    """

    # filter for collecting usgs term querying astronomy journals
    # per Mike, for now, querying these journals (only for the identification step)
    astronomy_journal_filter = 'bibstem:("%s") ' % '" OR "'.join(config['PLANETARYNAMES_PIPELINE_TOP_ASTRONOMY_JOURNALS'])

    # additional filters applied to queries for the identification step
    other_usgs_filters = 'database:astronomy'

    # list of regex patterns and replacement strings for cleaning HTML from text
    re_replace_html = [
        (re.compile(r'<i>|</i>|<b>|</b>', flags=re.IGNORECASE), ''),
        (re.compile(r'<'), '&#60;'),
        (re.compile(r'>'), '&#62;')
    ]
    # regex pattern for identifying the references section in document text
    re_references = re.compile(r'.(?=References[\W\s]*[A-Z\[\(0-9]+)')

    def __init__(self, args: EntityArgs):
        """
        initialize the SearchRetrieval class

        :param args: configuration arguments for constructing queries
        """
        self.args = args
        # an OR of singular and plural feature types for the query
        self.feature_types_ored = '" OR "'.join([self.args.feature_type, self.args.feature_type_plural]) if self.args.feature_type_plural \
                                                                                                         else self.args.feature_type
        # filter for full-text modification timestamp for the query
        self.date_time_filter = f'fulltext_mtime:["{self.args.timestamp}t00:00:00.000Z" TO *]'
        # start year extracted from the timestamp for the query
        self.year_start = self.args.timestamp.split('-')[0]

    def single_solr_query(self, start: int, rows: int, query: str) -> Tuple[List[Dict], int]:
        """
        execute a single query to the Solr search engine

        :param start: starting index for pagination
        :param rows: number of rows to retrieve
        :param query: Solr query string
        :return: tuple containing list of document dictionaries and status code
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

    def solr_query(self, query: str) -> List[Dict]:
        """
        execute a paginated query to solr

        :param query: Solr query string
        :return: list of document dictionaries
        """
        index = 0
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
        construct and execute a query to collect records for identifying entities

        :return: list of document dictionaries
        """
        query = f'full:(="{self.args.feature_name}") full:("{self.args.target}") full:("{self.feature_types_ored}") '
        query += f'{self.astronomy_journal_filter} {self.other_usgs_filters} {self.date_time_filter}'
        return self.solr_query(query)

    def collect_usgs_terms_query(self) -> List[Dict]:
        """
        construct and execute a multi-level query to collect USGS terms

        :return: list of document dictionaries
        """
        query = f'full:("{self.args.feature_name}") full:("{self.args.target}") full:("{self.feature_types_ored}") '
        query += f'{self.other_usgs_filters} {self.date_time_filter}'

        # multi level query
        # see if can extract enough records with top filters, if not keep removing filters
        # until either get enough records, or run out of filters
        queries = [
            f'{query} {self.astronomy_journal_filter} property:refereed year:[{self.year_start} TO *]', # all conditions included
            f'{query} property:refereed year:[{self.year_start} TO *]', # without top astronomy journals
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

    def collect_non_usgs_terms_query(self) -> List[Dict]:
        """
        construct and execute a query to collect non-USGS terms

        :return: list of document dictionaries
        """
        # exclude any record with mention of any of the celestial bodies
        all_targets = '" OR "'.join(self.args.all_targets)
        query = f'full:(="{self.args.feature_name}") -full:("{all_targets}") '
        query += f'-{self.other_usgs_filters} year:[{self.year_start} TO *]'

        # 200 is enough for the negative side
        docs, status_code = self.single_solr_query(query=query, start=0, rows=500)
        if status_code == 200:
            return docs
        else:
            logger.error(f"Error querying non usgs terms, got status code: {status_code} from solr.")
        return []
