#!/usr/bin/python
# -*- coding: utf-8 -*-

import regex
import unicodedata
from typing import List, Dict, Union

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

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

    def __init__(self, target: str, feature_type: str, feature_type_plural: str, feature_name: str,
                 context_ambiguous_feature_names: List[str], multi_token_containing_feature_names: List[str],
                 name_entity_labels: List[Dict[str, Union[str, int]]], all_targets: List[str]=[]):
        """

        :param target:
        :param feature_type:
        :param feature_type_plural:
        :param feature_name:
        :param context_ambiguous_feature_names: list of context for feature name, can be empty
        :param multi_token_containing_feature_names: list of multi token feature names that contains the feature name, can be emtpy
        """
        self.target = target
        self.feature_type = feature_type
        self.feature_type_plural = feature_type_plural
        self.feature_name = feature_name
        self.context_ambiguous_feature_names = context_ambiguous_feature_names
        self.multi_token_containing_feature_names = multi_token_containing_feature_names
        self.name_entity_labels = name_entity_labels
        self.all_targets = all_targets


class Synonyms(object):

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

    def get(self, term):
        """

        :param term:
        :return:
        """
        term_synonyms = self.synonym_list.get(term.lower(), '')
        if not term_synonyms:
            return term
        return term + "|" + term_synonyms.replace(', ', '|')

    def get_target_terms(self, target):
        """

        :param target:
        :return:
        """
        return self.get(target)

    def get_feature_type_terms(self, feature_types):
        """

        :param feature_types:
        :return:
        """
        return self.get(feature_types[0]) + '|' + self.get(feature_types[1])

    def add_synonyms(self, terms):
        """

        :param terms:
        :return:
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

    re_control_chars = regex.compile(r'[\0-\x1f\x7f-\x9f]')

    def replace_control_chars(self, excerpt):
        """

        :return:
        """
        def replace(match):
            char = match.group(0)
            try:
                return unicodedata.lookup(char)
            except:
                # ignore the ones that are  not replaceable
                return ' '

        return self.re_control_chars.sub(replace, excerpt).strip()
