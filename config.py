# -*- coding: utf-8 -*-

from enum import Enum

LOG_STDOUT = False


PLANETARYNAMES_PIPELINE_ADSWS_API_TOKEN = 'this is a secret api token!'
PLANETARYNAMES_PIPELINE_SOLR_URL = 'https://dev.adsabs.harvard.edu/v1/search/query'


# db config
SQLALCHEMY_URL = 'postgresql+psycopg2://postgres:postgres@localhost:5432/postgres'
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = False


# possible values: WARN, INFO, DEBUG
LOGGING_LEVEL = 'DEBUG'


# celery config
CELERY_INCLUDE = ['adsrefpipe.tasks']
CELERY_BROKER = 'pyamqp://'


# number of times each items is requeued if not processed unsuccessfully before quiting
MAX_QUEUE_RETRIES = 3


# types of actions that go through queue, collecting knowledge base data, identifying USGS terms, or both (end_to_end)
# other types of actions:
# remove only the last knowledge base records (remove_the_most_recent)
# remove all knowledge base records except for the last entry (remove_all_but_last)
# add a keyword manually if the excerpt contains that keyword (add_keyword_to_knowledge_graph)
# remove a keyword if it exists (remove_keyword_from_knowledge_graph)
# retrieve all the identified entities (retrieve_identified_entities)
PLANETARYNAMES_PIPELINE_ACTION = Enum('action', ['collect', 'identify', 'end_to_end',
                                                 'remove_the_most_recent', 'remove_all_but_last',
                                                 'add_keyword_to_knowledge_graph', 'remove_keyword_from_knowledge_graph',
                                                 'retrieve_identified_entities'])

PLANETARYNAMES_PIPELINE_FORMAT_SIGNIFICANT_DIGITS = 2

PLANETARYNAMES_PIPELINE_NASA_CONCEPT_URL = 'notyetactive.com'