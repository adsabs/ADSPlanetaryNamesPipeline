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


## Command lines:

### To collect data for craters on Mercury:

    python run.py -a collect -t Mercury -f Crater
    

### To identify crater features names on Mercury from ADS records:
    
    python run.py -a identify -t Mercury -f Crater
    

### To query database for identified entities from last 30 days with confidence >= 0.8:

    python run.py -a retrieve_identified_entities -c 0.8 -d 30
    

### To add "basin" keyword to knowledge graph records for Apollo/Moon where the keyword exists in the excerpt but not in the keywords:

    python run.py -a add_keyword_to_knowledge_graph -k basin -t Moon -f Apollo
    

### To remove the "fig" keyword from knowledge graph records for Apollo/Moon where it exists in the keywords field:
    
    python run.py -a remove_keyword_from_knowledge_graph -k fig -t Moon -f Apollo
    

### To remove the most recent knowledge base record for the given feature name (Avan) and target (Mars):
    
    python run.py -a remove_the_most_recent -t Mars -f Avan
    

### To remove all knowledge base records except the most recent one for the given feature name and target:
    
    python run.py -a remove_all_but_last -t Mars -f Avan
    


## Maintainers

Golnaz
