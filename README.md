# ADSPlanetaryNamesPipeline
planetary names pipeline

[![Build Status](https://github.com/adsabs/ADSPlanetaryNamesPipeline/actions/workflows/labels.yml/badge.svg)](https://github.com/adsabs/ADSPlanetaryNamesPipeline/actions/workflows/labels.yml)
[![Coverage Status](https://coveralls.io/repos/github/adsabs/ADSPlanetaryNamesPipeline/badge.svg?branch=main)](https://coveralls.io/github/adsabs/ADSPlanetaryNamesPipeline?branch=main)
    


## Short summary

This pipeline processes planetary nomenclature to identify named entities (feature names) for a target and feature type. It can collect data, predict entities, and perform end-to-end processing (collect followed by predict).
    


## Required software

    - RabbitMQ and PostgreSQL
    


## Setup (recommended)

    $ virtualenv python
    $ source python/bin/activate
    $ pip install -r requirements.txt
    $ pip install -r dev-requirements.txt
    $ vim local_config.py # edit, edit
    $ ./start-celery.sh
    


## Queues
    - task_process_planetary_nomenclature: queues one feature name at a time for processing
    


## Command Line Options

    -a, --action            Action to perform (collect, identify, etc.)
    -t, --target            Target celestial body (e.g., Moon, Mars)
    -f, --feature_type      Feature type (e.g., Crater)
    -n, --feature_name      Feature name (e.g., Apollo)
    -k, --keyword           Knowledge graph keyword to add/remove/export
    -c, --confidence_score  (optional) Filter by minimum confidence score
    -d, --days              (optional) Filter by days for retrieve_identified_entities action
    -s, --timestamp         (optional) Specify date in YYYY-MM-DD format for identification actions
    -u, --usgs_update       CSV file for USGS data update
    -o, --output_file       (optional) Specify file name for data export
    -l, --label             (optional) Specify label of the knowledge graph keywords to export (e.g, planetary or unknown)
    


## Command Lines


### To collect data for craters on Mercury:
    python run.py -a collect -t Mercury -f Crater
    

### To identify crater features names on Mercury from ADS records:
    python run.py -a identify -t Mercury -f Crater


### To perform end-to-end processing (collect and identify):
    python run.py -a end_to_end -t Mercury -f Crater


### To identify features using a specific timestamp:
    python run.py -a identify -t Mars -f Crater -s 2023-01-01


### To query database for identified entities from last 30 days with confidence >= 0.8:
    python run.py -a retrieve_identified_entities -c 0.8 -d 30
    

### To add "basin" keyword to knowledge graph records:
    python run.py -a add_keyword_to_knowledge_graph -k basin -t Moon -f Apollo
    

### To remove the "fig" keyword from knowledge graph records:
    python run.py -a remove_keyword_from_knowledge_graph -k fig -t Moon -f Apollo
    

### To remove the most recent knowledge base record:
    python run.py -a remove_the_most_recent -t Mars -f Avan
    

### To remove all knowledge base records except the most recent one:
    python run.py -a remove_all_but_last -t Mars -f Avan


### To export keywords from knowledge graph:
    python run.py -a retrieve_knowledge_graph_keywords -t Mars -f Crater -l positive -o keywords_export.csv


### To update database with new USGS gazetteer data:
    python run.py -a update_database_with_usgs_entities -u updated_usgs_terms.csv


### To process specific feature by name instead of type:
    python run.py -a collect -t Mars -n Amenthes


### To retrieve identified entities for specific feature:
    python run.py -a retrieve_identified_entities -t Mars -n Amenthes -c 0.75


### To retrieve knowledge graph keywords for specific feature:
    python run.py -a retrieve_knowledge_graph_keywords -t Mars -n Amenthes -l planetary## Maintainers
    


## Maintainers

Golnaz