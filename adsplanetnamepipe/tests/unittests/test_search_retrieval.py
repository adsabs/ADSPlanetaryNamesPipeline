import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)


import unittest

from unittest.mock import MagicMock, patch, call

from requests.exceptions import RequestException

from adsplanetnamepipe.utils.search_retrieval import SearchRetrieval
from adsplanetnamepipe.utils.common import EntityArgs

from adsplanetnamepipe.tests.unittests.stubdata import solrdata


class TestSearchRetrieval(unittest.TestCase):
    """
    Tests the search retrieval module
    """

    def setUp(self):
        """ Set up the config class and create an instance of SearchRetrieval """

        self.args = EntityArgs(
            target="Mars",
            feature_type="Crater",
            feature_type_plural="Craters",
            feature_name="Rayleigh",
            context_ambiguous_feature_names=["asteroid", "main belt asteroid", "Moon", "Mars"],
            multi_token_containing_feature_names=["Rayleigh A", "Rayleigh B", "Rayleigh C", "Rayleigh D"],
            name_entity_labels=[{'label': 'planetary', 'value': 1}, {'label': 'non planetary', 'value': 0}],
            timestamp='2000-01-01',
            all_targets = ["Mars", "Mercury", "Moon", "Venus"]
        )
        self.search_retrieval = SearchRetrieval(self.args)

    @patch('adsplanetnamepipe.utils.search_retrieval.requests.get')
    def test_single_solr_query(self, mock_get):
        """ test single_solr_query method """

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': {'docs': [solrdata.doc_1, solrdata.doc_2]}}
        mock_get.return_value = mock_response

        docs, status_code = self.search_retrieval.single_solr_query(start=0, rows=10, query='test query')

        self.assertEqual(status_code, 200)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]['bibcode'], '2010JGRE..115.0F08G')
        self.assertEqual(docs[1]['bibcode'], '2023Icar..39615503S')

    @patch('adsplanetnamepipe.utils.search_retrieval.requests.get')
    def test_single_solr_query_500_error(self, mock_get):
        """ test single_solr_query method when raises a 500 error """

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        docs, status_code = self.search_retrieval.single_solr_query(start=0, rows=10, query='test query')

        self.assertIsNone(docs)
        self.assertEqual(status_code, 500)

    @patch('adsplanetnamepipe.utils.search_retrieval.requests.get')
    def test_single_solr_query_exception(self, mock_get):
        """ test single_solr_query method when raises RequestException """

        mock_get.side_effect = RequestException("Test Request Exception")

        docs, exception = self.search_retrieval.single_solr_query(start=0, rows=10, query='*:*')

        self.assertIsNone(docs)
        self.assertIsInstance(exception, RequestException)
        self.assertEqual(str(exception), "Test Request Exception")

    @patch('adsplanetnamepipe.utils.search_retrieval.SearchRetrieval.single_solr_query')
    def test_solr_query_iteration(self, mock_single_solr_query):
        """ test the behavior of solr_query to return different results in each iteration """

        mock_single_solr_query.side_effect = [
            ([{'bibcode': '2024arXiv240320332S'}, {'bibcode': '2024arXiv240320323T'}], 200),
            ([{'bibcode': '2024arXiv240320321V'}, {'bibcode': '2024arXiv240320316D'}], 200),
            ([], 200),
        ]

        docs = self.search_retrieval.solr_query('*:*')

        # the expected number of iterations
        self.assertEqual(mock_single_solr_query.call_count, 3)
        # the expected arguments passed to single_solr_query in each iteration
        mock_single_solr_query.assert_any_call(start=0, rows=2000, query='*:*')
        mock_single_solr_query.assert_any_call(start=2000, rows=2000, query='*:*')
        mock_single_solr_query.assert_any_call(start=4000, rows=2000, query='*:*')

        self.assertEqual(len(docs), 4)
        self.assertEqual(docs, [{'bibcode': '2024arXiv240320332S'}, {'bibcode': '2024arXiv240320323T'},
                                {'bibcode': '2024arXiv240320321V'}, {'bibcode': '2024arXiv240320316D'}])

    @patch('adsplanetnamepipe.utils.search_retrieval.SearchRetrieval.single_solr_query')
    def test_solr_query_iteration_error(self, mock_single_solr_query):
        """ test the behavior of single_solr_query when returning an error status code """

        mock_single_solr_query.return_value = (None, 500)

        docs = self.search_retrieval.solr_query('*:*')

        self.assertEqual(mock_single_solr_query.call_count, 1)
        self.assertEqual(len(docs), 0)

    @patch('adsplanetnamepipe.utils.search_retrieval.SearchRetrieval.solr_query')
    def test_identify_terms_query(self, mock_solr_query):
        """ test identify_terms_query which returns the result of the query for identifying entities """

        expected_query = f'full:(="Rayleigh") full:("Mars") full:("Crater" OR "Craters") '
        expected_query += f'{self.search_retrieval.astronomy_journal_filter} {self.search_retrieval.other_usgs_filters} '
        expected_query += f'{self.search_retrieval.date_time_filter}'

        expected_result = [ {'bibcode': '2024arXiv240320332S'}, {'bibcode': '2024arXiv240320323T'}]

        mock_solr_query.return_value = expected_result
        result = self.search_retrieval.identify_terms_query()

        mock_solr_query.assert_called_once_with(expected_query)
        self.assertEqual(result, expected_result)

    @patch('adsplanetnamepipe.utils.search_retrieval.SearchRetrieval.solr_query')
    def test_identify_terms_query_no_docs(self, mock_solr_query):
        """ test identify_terms_query when no documents are found or an error occurs """

        expected_query = f'full:("Rayleigh") full:("Mars") full:("Crater" OR "Craters")  '
        expected_query += f'{self.search_retrieval.astronomy_journal_filter} {self.search_retrieval.other_usgs_filters} '
        expected_query += f'{self.search_retrieval.date_time_filter}'

        # case 1: no documents found
        mock_solr_query.return_value = []
        result = self.search_retrieval.identify_terms_query()
        self.assertEqual(result, [])

        # case 2: error occurs
        mock_solr_query.return_value = []
        result = self.search_retrieval.identify_terms_query()
        self.assertEqual(result, [])

    @patch('adsplanetnamepipe.utils.search_retrieval.SearchRetrieval.solr_query')
    def test_collect_usgs_terms_query(self, mock_solr_query):
        """ test collect_usgs_terms_query which returns the result of query for the collect step positive """

        base_query = 'full:("Rayleigh") full:("Mars") full:("Crater" OR "Craters") '
        base_query += f'{self.search_retrieval.other_usgs_filters} {self.search_retrieval.date_time_filter}'
        expected_queries = [
            f'{base_query} {self.search_retrieval.astronomy_journal_filter} property:refereed year:[{self.search_retrieval.year_start} TO *]',
            f'{base_query} property:refereed year:[{self.search_retrieval.year_start} TO *]',
            f'{base_query} property:refereed',
            base_query
        ]

        # expected results from solr_query for each query,
        # note that the last query is not executed since the one before last is accepted
        expected_results = [
            [{'bibcode': '2024arXiv240320332S'}, {'bibcode': '2024arXiv240320323T'},
             {'bibcode': '2024arXiv240320321V'}],
            [],
            [{'bibcode': '2024arXiv240320315S'}, {'bibcode': '2024arXiv240320314P'},
             {'bibcode': '2024arXiv240320311B'}, {'bibcode': '2024arXiv240320303K'},
             {'bibcode': '2024arXiv240320302G'}],
            [{'bibcode': '2024arXiv240320301M'}]
        ]

        mock_solr_query.side_effect = expected_results
        result = self.search_retrieval.collect_usgs_terms_query()

        # assert that solr_query was called with the expected queries
        mock_solr_query.assert_has_calls([call(query) for query in expected_queries[:2]])
        # and that the result matches the expected result (ie, the third set of results)
        self.assertEqual(result, expected_results[2])

    @patch('adsplanetnamepipe.utils.search_retrieval.SearchRetrieval.solr_query')
    @patch('adsplanetnamepipe.utils.search_retrieval.logger')
    def test_collect_usgs_terms_query_no_docs(self, mock_logger, mock_solr_query):
        """ test collect_usgs_terms_query when no docs is found """

        base_query = 'full:("Rayleigh") full:("Mars") full:("Crater" OR "Craters") '
        base_query += f'{self.search_retrieval.other_usgs_filters} {self.search_retrieval.date_time_filter}'
        expected_queries = [
            f'{base_query} {self.search_retrieval.astronomy_journal_filter} property:refereed year:[{self.search_retrieval.year_start} TO *]',
            f'{base_query} property:refereed year:[{self.search_retrieval.year_start} TO *]',
            f'{base_query} property:refereed',
            base_query
        ]

        expected_results = [[], [], [], []]

        mock_solr_query.side_effect = expected_results
        result = self.search_retrieval.collect_usgs_terms_query()

        # assert that solr_query was called with the expected queries
        mock_solr_query.assert_has_calls([call(query) for query in expected_queries])
        self.assertEqual(result,[])
        mock_logger.error.assert_called_with("Unable to get data from solr for Rayleigh/Mars.")

    @patch('adsplanetnamepipe.utils.search_retrieval.SearchRetrieval.single_solr_query')
    def test_collect_non_usgs_terms_query(self, mock_single_solr_query):
        """ collect_non_usgs_terms_query which returns the result of query for the collect step negative """

        expected_query = 'full:(="Rayleigh") -full:("Mars" OR "Mercury" OR "Moon" OR "Venus") '
        expected_query += f"-{self.search_retrieval.other_usgs_filters} year:[{self.search_retrieval.year_start} TO *]"

        expected_result = [{'bibcode': '2024arXiv240320332S'}, {'bibcode': '2024arXiv240320323T'}]

        mock_single_solr_query.return_value = (expected_result, 200)
        result = self.search_retrieval.collect_non_usgs_terms_query()

        mock_single_solr_query.assert_called_once_with(query=expected_query, start=0, rows=500)
        self.assertEqual(result, expected_result)

    @patch('adsplanetnamepipe.utils.search_retrieval.SearchRetrieval.single_solr_query')
    @patch('adsplanetnamepipe.utils.search_retrieval.logger')
    def test_collect_non_usgs_terms_query_no_docs(self, mock_logger, mock_single_solr_query):
        """ collect_non_usgs_terms_query when no docs is found """

        expected_query = 'full:(="Rayleigh") -full:("Mars") -full:("Crater" OR "Craters") '
        expected_query += f"-{self.search_retrieval.other_usgs_filters} year:[{self.search_retrieval.year_start} TO *]"

        # no docs with status code 200
        mock_single_solr_query.return_value = ([], 200)
        result = self.search_retrieval.collect_non_usgs_terms_query()
        self.assertEqual(result, [])

        # no docs when error
        mock_single_solr_query.return_value = ([], 500)
        result = self.search_retrieval.collect_non_usgs_terms_query()
        self.assertEqual(result, [])
        mock_logger.error.assert_called_with("Error querying non usgs terms, got status code: 500 from solr.")

    @patch('adsplanetnamepipe.utils.search_retrieval.SearchRetrieval.single_solr_query')
    @patch('adsplanetnamepipe.utils.search_retrieval.logger')
    def test_single_solr_query_no_docs(self, mock_logger, mock_single_solr_query):
        """ test when solr returns no docs """

        expected_query = 'full:(="Rayleigh") -full:("Mars") -full:("Crater" OR "Craters") '
        expected_query += f"-{self.search_retrieval.other_usgs_filters}"

        mock_single_solr_query.side_effect = [
            ([], 200),  # First call returns an empty list with status code 200
            (None, None)  # Subsequent calls are not executed due to the break statement
        ]

        result = self.search_retrieval.solr_query(expected_query)

        self.assertEqual(result, [])
        mock_single_solr_query.assert_called_with(start=0, rows=2000, query=expected_query)
        mock_logger.info.assert_called_with("Got 0 docs from solr.")

    @patch('requests.get')
    def test_single_solr_query_body_with_references_to_remove(self, mock_get):
        """ test single_solr_query_body when there is a reference section in the body, to attempt to remove it """

        start = 0
        rows = 10
        query = 'full:(="Rayleigh") -full:("Mars") -full:("Crater" OR "Craters") '
        query += f"-{self.search_retrieval.other_usgs_filters}"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': {
                'docs': [
                    {
                        'title': 'Sample Title',
                        'abstract': 'Sample Abstract',
                        'body': 'Introduction: some text here. Method: some text here. References: [1] list of references here. Appendix: some text here.'
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        docs, status_code = self.search_retrieval.single_solr_query(start, rows, query)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]['body'], 'Introduction: some text here. Method: some text here.')

if __name__ == '__main__':
    unittest.main()
