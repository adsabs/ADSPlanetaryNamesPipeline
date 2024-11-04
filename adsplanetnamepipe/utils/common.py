#!/usr/bin/python
# -*- coding: utf-8 -*-

import regex
import unicodedata
from typing import List, Dict, Union, TypedDict
from enum import Enum

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

class PLANETARYNAMES_PIPELINE_ACTION(Enum):
    """
    an enumeration of actions that can be performed in the planetary names pipeline

    types of actions that go through queue, collecting knowledge base data, identifying USGS terms, or both (end_to_end)
    other types of actions:
        remove only the last knowledge base records (remove_the_most_recent)
        remove all knowledge base records except for the last entry (remove_all_but_last)
        add a keyword manually if the excerpt contains that keyword (add_keyword_to_knowledge_graph)
        remove a keyword if it exists (remove_keyword_from_knowledge_graph)
        retrieve all the identified entities (retrieve_identified_entities)
        update database with any new approved target/feature type/feature name (update_database_usgs_entities)
    also included is invalid when the action is none of the above mentioned
    """

    collect = 'collect'
    identify = 'identify'
    end_to_end = 'end_to_end'
    remove_the_most_recent = 'remove_the_most_recent'
    remove_all_but_last = 'remove_all_but_last'
    add_keyword_to_knowledge_graph = 'add_keyword_to_knowledge_graph'
    remove_keyword_from_knowledge_graph = 'remove_keyword_from_knowledge_graph'
    retrieve_identified_entities = 'retrieve_identified_entities'
    update_database_with_usgs_entities = 'update_database_with_usgs_entities'
    invalid = 'invalid'


class EntityArgs():
    """
    EntityArgs is a class to encapsulate the planetry name entity information
    """
    target: str
    feature_type: str
    feature_type_plural: str
    feature_name: str
    ambiguous_feature_name_context: List[str]
    multi_token_containing_feature_names: List[str]
    name_entity_labels: List[Dict[str, Union[str, int]]]
    all_targets: List[str]
    timestamp: str

    def __init__(self, target: str, feature_type: str, feature_type_plural: str, feature_name: str,
                 context_ambiguous_feature_names: List[str], multi_token_containing_feature_names: List[str],
                 name_entity_labels: List[Dict[str, Union[str, int]]], timestamp: str, all_targets: List[str]=[]):
        """
        initialize the EntityArgs class

        :param target: the celestial body targeted
        :param feature_type: the type of the feature
        :param feature_type_plural: plural form of the feature type
        :param feature_name: name of the feature
        :param context_ambiguous_feature_names: list of contexts for ambiguous feature names
        :param multi_token_containing_feature_names: list of multi-token feature names containing the feature name
        :param name_entity_labels: list of dictionaries containing name entity labels
        :param timestamp: timestamp used for fulltext and year filtering
        :param all_targets: list of all target bodies
        """
        self.target = target
        self.feature_type = feature_type
        self.feature_type_plural = feature_type_plural
        self.feature_name = feature_name
        self.context_ambiguous_feature_names = context_ambiguous_feature_names
        self.multi_token_containing_feature_names = multi_token_containing_feature_names
        self.name_entity_labels = name_entity_labels
        self.timestamp = timestamp
        self.all_targets = all_targets


class PlanetaryNomenclatureTask(TypedDict):
    """
    a TypedDict class representing a task in the planetary nomenclature pipeline

    this class defines the structure of a task, including the type of action
    to be performed and the associated entity arguments
    """
    action_type: PLANETARYNAMES_PIPELINE_ACTION
    args: EntityArgs


class Synonyms(object):
    """
    a class to manage synonyms for planetary terms

    this class provides methods to retrieve and add synonyms for various
    planetary terms, including targets and feature types
    """

    # generated 5/20/2022
    synonym_list = {
        "amalthea": "amalthea, amathea",
        "crater": "crater, craters, cratering, cratered, cratere, craterlike, crateres, crateris, uncratered, craterlets, krater, craterform, subcrater, craterization, craterless, crateriform, craterized, noncrater, crator, crateri, craterforming, crators, crrater, noncratering, crateral",
        "ariel": "ariel, arial, ariol",
        "chasma": "chasma, chasmata",
        "callisto": "callisto, calisto, callistoan, callistro, calliso",
        "mons": "mons, montes",
        "planitia": "planitia, planitiae, planita, planitae, planitias",
        "sulcus": "sulcus, sulci",
        "charon": "charon, charonian, charonlike",
        "deimos": "deimos, diemos, deimis",
        "enceladus": "enceladus, encedalus, enceladas, encaladus",
        "epimetheus": "epimetheus, epimethius, epimethus",
        "ganymede": "ganymede, ganymed, ganymedes, ganymedean, ganimede, ganymedian, granymede",
        "patera": "patera, paterae",
        "hyperion": "hyperion, hyperionization",
        "iapetus": "iapetus, lapetus",
        "terra": "terra, terraforming, terrae, terraformed, postterraformed",
        "mars": "mars, martian, marsquakes, martians, martain, marslike, circummartian, marsshine",
        "mare": "mare, nonmare, mares, premare, postmare",
        "mercury": "mercury, hg, mercuric, mercurian, mercurous, mercurial, merkur, mecury, mercurean, meercury, protomercury",
        "mimas": "mimas, mima",
        "moon": "lunar, moons, luna, lune, mond, luni, monde, moonlets, cislunar, lunaire, moonlet, lunae, moonless, moonrise, translunar, moonset, perilune, circumlunar, lunari, lunare, extralunar, lunes, moonlike, lunarlike, prelunar, moonwide, lunary, nonlunar, circalunadian, subselenian, subselenean, oflunar, moolet, lunarbased, lunarized",
        "sinus": "sinus, sinuses",
        "phoebe": "phoebe, phebus, phoebus",
        "pluto": "pluto, plutonic, plutonism, plutonian, transplutonian, transplutonic, protoplutonian",
        "titan": "titan, titans",
        "lacuna": "lacuna, lacunas",
        "titania": "titania, titanian",
        "triton": "triton, tritons, tritton",
        "plume": "plume, plumes, superplume, megaplumes, plumelike, pluming",
        "venus": "venus, venusian, venutian, venusquakes, venuslike, venusion, venusionopause",
    }

    def get(self, term: str) -> str:
        """
        get synonyms for a given term

        :param term: the term to find synonyms for
        :return: string of the term and its synonyms, separated by '|'
        """
        term_synonyms = self.synonym_list.get(term.lower(), '')
        if not term_synonyms:
            return term
        return term + "|" + term_synonyms.replace(', ', '|')

    def get_target_terms(self, target: str) -> str:
        """
        get synonyms for a target celestial body

        :param target: the target celestial body
        :return: string of the target and its synonyms, separated by '|'
        """
        return self.get(target)

    def get_feature_type_terms(self, feature_types: List[str]) -> str:
        """
        get synonyms for feature types

        :param feature_types: list of feature types
        :return: string of feature types and their synonyms, separated by '|'
        """
        return self.get(feature_types[0]) + '|' + self.get(feature_types[1])

    def add_synonyms(self, terms: List[str]) -> List[str]:
        """
        add synonyms to a list of terms

        :param terms: list of terms to add synonyms to
        :return: list of original terms and their synonyms
        """
        # add in synonyms of the terms, if available
        # turn everything lower case
        terms = [v.lower() for v in terms]
        added_synonyms = []
        for v in terms:
            vocab_synonyms = self.synonym_list.get(v, '')
            if vocab_synonyms:
                vocab_synonyms = vocab_synonyms.replace(',', '').split(' ')
                for s in vocab_synonyms:
                    if s.lower() not in terms:
                        added_synonyms.append(s.lower())
        return terms + added_synonyms


class Unicode():
    """
    a class to handle Unicode-related operations

    this class provides methods to replace control characters in text
    """

    # a compiled regular expression to match Unicode control characters
    re_control_chars = regex.compile(r'[\0-\x1f\x7f-\x9f]')

    def replace_control_chars(self, excerpt: str) -> str:
        """
        replace control characters in a given excerpt

        :param excerpt: the text excerpt to process
        :return: processed string with control characters replaced
        """
        def replace(match):
            char = match.group(0)
            try:
                return unicodedata.lookup(char)
            except:
                # ignore the ones that are  not replaceable
                return ' '

        return self.re_control_chars.sub(replace, excerpt).strip()
