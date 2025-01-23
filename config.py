# -*- coding: utf-8 -*-

LOG_STDOUT = False


PLANETARYNAMES_PIPELINE_ADSWS_API_TOKEN = 'this is a secret api token!'
PLANETARYNAMES_PIPELINE_SOLR_URL = 'https://dev.adsabs.harvard.edu/v1/search/query'


PLANETARYNAMES_PIPELINE_BRAIN_API_TOKEN = 'this is a secret api token for the Brain!'
PLANETARYNAMES_PIPELINE_BRAIN_URL = 'https://playground.adsabs.harvard.edu/brain/v1/chat'


# db config
SQLALCHEMY_URL = 'postgresql+psycopg2://postgres:postgres@localhost:5432/postgres'
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = False


# possible values: WARN, INFO, DEBUG
LOGGING_LEVEL = 'DEBUG'


# celery config
CELERY_INCLUDE = ['adsplanetnamepipe.tasks']
CELERY_BROKER = 'pyamqp://'


# number of times each items is requeued if not processed unsuccessfully before quiting
MAX_QUEUE_RETRIES = 3


PLANETARYNAMES_PIPELINE_FORMAT_SIGNIFICANT_DIGITS = 2

PLANETARYNAMES_PIPELINE_NASA_CONCEPT_URL = 'http://0.0.0.0:5000'

PLANETARYNAMES_PIPELINE_TOP_ASTRONOMY_JOURNALS = [
    "A&A", "AdSpR", "ApJ", "ApJL", "ApJS", "ChGeo", "E&PSL", "EM&P", "GeCoA", "Geo", "GeoJI", "GeoRL", "GGG", "Icar",
    "JGRA", "JGRB", "JGRE", "LPI", "M&PS", "MNRAS", "NatGe", "Natur", "P&SS", "PASJ", "PASP", "PEPI", "PSJ", "Sci",
    "SSRv"
]

PLANETARYNAMES_PIPELINE_DEFAULT_TIMESTAMP = '2000-01-01'