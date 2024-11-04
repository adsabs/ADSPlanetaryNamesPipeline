import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from datetime import datetime, timezone
import time
import re

import unittest
from unittest.mock import patch

from typing import List, Tuple

from sqlalchemy.exc import SQLAlchemyError

from adsplanetnamepipe import app
from adsplanetnamepipe.models import Base, FeatureType, KnowledgeBaseHistory, KnowledgeBase, \
    NamedEntity, NamedEntityHistory, USGSNomenclature
from adsplanetnamepipe.tests.unittests.stubdata.dbdata import collection_records, usgs_nomenclature_records, \
    target_records, feature_type_records, feature_name_records, \
    feature_name_context_records, ambiguous_feature_name_records, multi_token_feature_name_records, \
    named_entity_label_records, knowledge_base_history_records, knowledge_base_records, \
    knowledge_base_history_records_non_planetary, knowledge_base_records_non_planetary, \
    named_entity_history_records, named_entity_records


class TesADSPlanetaryNamesPipelineCelery(unittest.TestCase):

    """
    Tests the application's methods
    """

    maxDiff = None

    postgresql_url_dict = {
        'port': 5432,
        'host': '127.0.0.1',
        'user': 'postgres',
        'database': 'postgres'
    }
    postgresql_url = 'postgresql://{user}:{user}@{host}:{port}/{database}' \
        .format(user=postgresql_url_dict['user'],
                host=postgresql_url_dict['host'],
                port=postgresql_url_dict['port'],
                database=postgresql_url_dict['database']
                )

    def setUp(self):
        self.test_dir = os.path.join(project_home, 'adsplanetnamepipe/tests')
        unittest.TestCase.setUp(self)
        self.app = app.ADSPlanetaryNamesPipelineCelery('test', local_config={
            'SQLALCHEMY_URL': self.postgresql_url,
            'SQLALCHEMY_ECHO': False,
            'PROJ_HOME': project_home,
            'TEST_DIR': self.test_dir,
        })
        Base.metadata.bind = self.app._session.get_bind()
        Base.metadata.create_all()

        self.add_stub_data()

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        Base.metadata.drop_all()
        self.app.close_app()

    def add_stub_data(self):
        """ Add stub data """

        with self.app.session_scope() as session:
            session.bulk_save_objects(collection_records)
            session.bulk_save_objects(usgs_nomenclature_records)
            session.bulk_save_objects(target_records)
            session.bulk_save_objects(feature_type_records)
            session.bulk_save_objects(feature_name_records)
            session.bulk_save_objects(feature_name_context_records)
            session.bulk_save_objects(ambiguous_feature_name_records)
            session.bulk_save_objects(multi_token_feature_name_records)
            session.bulk_save_objects(named_entity_label_records)
            session.commit()

    def test_get_feature_name_entities(self):
        """ test get_feature_name_entities """

        expected_feature_names = ['Burney', 'Coughlin', 'Edgeworth', 'Elliot', 'Hardaway', 'Hardie', 'Khare', 'Kiladze', 'Oort', 'Pulfrich', 'Simonelli', 'Zagar']
        feature_names = self.app.get_feature_name_entities('Pluto','Crater')
        self.assertTrue(feature_names == expected_feature_names)

    def test_get_feature_type_entity(self):
        """ test get_feature_type_entity """

        self.assertTrue(self.app.get_feature_type_entity('Pluto','Elliot') == 'Crater')

    def test_get_plural_feature_type_entity(self):
        """ test get_plural_feature_type_entity """

        # with plural form
        self.assertTrue(self.app.get_plural_feature_type_entity('Crater') == 'Craters')
        # without plural form
        self.assertTrue(self.app.get_plural_feature_type_entity('Large ringed feature') == '')

    def test_get_context_ambiguous_feature_name(self):
        """ test get_context_ambiguous_feature_name """

        self.assertTrue(self.app.get_context_ambiguous_feature_name('Airy') == ['Moon', 'Mars'])

    def test_get_multi_token_containing_feature_name(self):
        """ test get_multi_token_containing_feature_name """

        expected_result = [
            'Airy 0', 'Airy A', 'Airy B', 'Airy C', 'Airy D', 'Airy E', 'Airy F', 'Airy G', 'Airy H', 'Airy J',
            'Airy L', 'Airy M', 'Airy N', 'Airy O', 'Airy P', 'Airy R', 'Airy S', 'Airy T', 'Airy V', 'Airy X'
        ]
        self.assertTrue(self.app.get_multi_token_containing_feature_name('Airy') == expected_result)

    def test_get_named_entity_label(self):
        """ test get_named_entity_label"""

        self.assertTrue(self.app.get_named_entity_label() == [{'label': 'planetary', 'value': 1}, {'label': 'non planetary', 'value': 0}])

    def test_get_target_entities(self):
        """ test get_target_entities"""

        expected_result = [
            'Amalthea', 'Ariel', 'Bennu', 'Callisto', 'Ceres', 'Charon', 'Dactyl', 'Deimos', 'Dione', 'Enceladus',
            'Epimetheus', 'Eros', 'Europa', 'Ganymede', 'Gaspra', 'Hyperion', 'Iapetus', 'Ida', 'Io', 'Itokawa',
            'Janus', 'Lutetia', 'Mars', 'Mathilde', 'Mercury', 'Mimas', 'Miranda', 'Moon', 'Oberon', 'Phobos',
            'Phoebe', 'Pluto', 'Proteus', 'Puck', 'Rhea', 'Ryugu', 'Steins', 'Tethys', 'Thebe', 'Titan', 'Titania',
            'Triton', 'Umbriel', 'Venus', 'Vesta'
        ]
        self.assertTrue(self.app.get_target_entities() == expected_result)

    def test_insert_knowledge_base_records(self):
        """ test insert_knowledge_base_records method """

        # id in knowledge_base_history record gets updated in the structure
        # that is not refereshed from one unittest to the next, so clone it
        history_record_lunar_antoniadi = knowledge_base_history_records[0].clone()
        history_record_martian_antoniadi = knowledge_base_history_records[1].clone()
        knowledgebase_records_lunar = knowledge_base_records[0]
        knowledgebase_records_martian = knowledge_base_records[1]

        records: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]] = [
            (history_record_lunar_antoniadi, knowledgebase_records_lunar),
            (history_record_martian_antoniadi, knowledgebase_records_martian)
        ]

        result = self.app.insert_knowledge_base_records(records)
        self.assertTrue(result)

        with self.app.session_scope() as session:
            history_records = session.query(KnowledgeBaseHistory).all()
            self.assertEqual(len(history_records), 2)
            self.assertEqual(history_records[0].feature_name_entity, 'Antoniadi')
            self.assertEqual(history_records[1].feature_name_entity, 'Antoniadi')

            knowledgebase_records = session.query(KnowledgeBase).all()
            self.assertEqual(len(knowledgebase_records), 6)

            knowledgebase_records = session.query(KnowledgeBase).filter(KnowledgeBase.history_id == history_records[0].id).all()
            self.assertEqual(len(knowledgebase_records), 3)

            knowledgebase_records = session.query(KnowledgeBase).filter(KnowledgeBase.history_id == history_records[1].id).all()
            self.assertEqual(knowledgebase_records[1].bibcode, '2023ApJ...958..171Z')
            self.assertEqual(knowledgebase_records[1].keywords, ['figures', 'radius', 'maven', 'field', 'model', 'altitude',
                                                                 'lillis', 'impact crater', 'impact', 'caused'])

    def test_get_knowledge_base_keywords(self):
        """ test get_knowledge_base_keywords method """

        # id in knowledge_base_history record gets updated in the structure
        # that is not refereshed from one unittest to the next, so clone it
        history_record_lunar_antoniadi = knowledge_base_history_records[0].clone()
        history_record_martian_antoniadi = knowledge_base_history_records[1].clone()
        knowledgebase_records_lunar = knowledge_base_records[0]
        knowledgebase_records_martian = knowledge_base_records[1]
        records: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]] = [
            (history_record_lunar_antoniadi, knowledgebase_records_lunar),
            (history_record_martian_antoniadi, knowledgebase_records_martian)
        ]
        # first populate database with a few records
        result = self.app.insert_knowledge_base_records(records)
        self.assertTrue(result)

        expected_keywords = [['crater', 'spa', 'basin', 'fig', 'ejecta', 'floor', 'rim', 'area', 'map', 'topographic',
                              'plain', 'unit', 'scr', 'nectarian', 'region', 'impact', 'data', 'imbrian', 'domain'],
                             ['series', 'hausen', 'fizeau', 'floor', 'upper', 'crater', 'lower', 'moretus',
                              'surrounding', 'csfd'],
                             ['domain', 'table', 'floor', 'graben', 'small', 'crater', 'fit', 'feo', 'map', 'spa']]
        # now get the lunar records
        result = self.app.get_knowledge_base_keywords('Antoniadi', 'Crater', 'Moon', 'planetary')
        # verify that keywords for all the records are returned
        self.assertEqual(len(result), len(knowledgebase_records_lunar))
        self.assertEqual(result, expected_keywords)

    def test_append_to_knowledge_base_keywords(self):
        """ test append_to_knowledge_base_keywords method """

        # id in knowledge_base_history record gets updated in the structure
        # that is not refereshed from one unittest to the next, so clone it
        history_record_lunar_antoniadi = knowledge_base_history_records[0].clone()
        history_record_martian_antoniadi = knowledge_base_history_records[1].clone()
        knowledgebase_records_lunar = knowledge_base_records[0]
        knowledgebase_records_martian = knowledge_base_records[1]
        records: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]] = [
            (history_record_lunar_antoniadi, knowledgebase_records_lunar),
            (history_record_martian_antoniadi, knowledgebase_records_martian)
        ]
        # first populate database with a few records
        result = self.app.insert_knowledge_base_records(records)
        self.assertTrue(result)

        # next count for each excerpt for each celestial body if it contains the keyword,
        # and that the keyword is not in the list of keywords already
        keyword = 'larger'
        for celestial_body_knowledgebase_records, expected_count in zip(knowledge_base_records, [2,0]):
            count = 0
            for celestial_body_knowledgebase_record in celestial_body_knowledgebase_records:
                if celestial_body_knowledgebase_record.excerpt:
                    if re.search(r'\b%s\b'%keyword, celestial_body_knowledgebase_record.excerpt) and \
                       keyword not in celestial_body_knowledgebase_record.keywords:
                        count += 1
            self.assertEqual(count, expected_count)

        # now attempt to add the keyword the keyword to both lunar and martian records
        for knowledge_base_history_record, expected_count in zip(knowledge_base_history_records, [2,0]):
            count = self.app.append_to_knowledge_base_keywords(knowledge_base_history_record.feature_name_entity,
                                                               knowledge_base_history_record.target_entity,
                                                               keyword)
            self.assertEqual(count, expected_count)

    def test_remove_from_knowledge_base_keywords(self):
        """ test remove_from_knowledge_base_keywords method """

        # id in knowledge_base_history record gets updated in the structure
        # that is not refereshed from one unittest to the next, so clone it
        history_record_lunar_antoniadi = knowledge_base_history_records[0].clone()
        history_record_martian_antoniadi = knowledge_base_history_records[1].clone()
        knowledgebase_records_lunar = knowledge_base_records[0]
        knowledgebase_records_martian = knowledge_base_records[1]
        records: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]] = [
            (history_record_lunar_antoniadi, knowledgebase_records_lunar),
            (history_record_martian_antoniadi, knowledgebase_records_martian)
        ]
        # first populate database with a few records
        result = self.app.insert_knowledge_base_records(records)
        self.assertTrue(result)

        # next count how many records contain the keyword
        keyword = 'crater'
        for celestial_body_knowledgebase_records, expected_count in zip(knowledge_base_records, [3,1]):
            count = 0
            for celestial_body_knowledgebase_record in celestial_body_knowledgebase_records:
                if keyword in celestial_body_knowledgebase_record.keywords:
                    count += 1
            self.assertEqual(count, expected_count)

        # now attempt to add the keyword the keyword to both lunar and martian records
        for knowledge_base_history_record, expected_count in zip(knowledge_base_history_records, [3,1]):
            count = self.app.remove_from_knowledge_base_keywords(knowledge_base_history_record.feature_name_entity,
                                                                 knowledge_base_history_record.target_entity,
                                                                 keyword)
            self.assertEqual(count, expected_count)

    def test_remove_most_recent_knowledge_base_records(self):
        """ test remove_most_recent_knowledge_base_records method """

        # queue records to alternate between planetary and non planetary
        queue_knowledge_base_history_records = [
            knowledge_base_history_records[0],
            knowledge_base_history_records_non_planetary[0],
            knowledge_base_history_records[1],
            knowledge_base_history_records_non_planetary[1],
        ]
        queue_knowledge_base_records = [
            knowledge_base_records[0],
            knowledge_base_records_non_planetary[0],
            knowledge_base_records[1],
            knowledge_base_records_non_planetary[1]
        ]
        # insert them three times
        for i in range(3):
            # lets call knowledge_base_history parent, and knowledge_base child (one parent, many child)
            for parent, child in zip(queue_knowledge_base_history_records, queue_knowledge_base_records):
                # insert them one knowledge_base_history record at a time,
                # apply some dealy, to make sure to get a different date
                result = self.app.insert_knowledge_base_records([(parent.clone(), child)])
                self.assertTrue(result)
                time.sleep(0.1)

        with self.app.session_scope() as session:
            # retrieve all record ids from KnowledgeBaseHistory and all history_ids from KnowledgeBase tables
            knowledge_base_history_ids = session.query(KnowledgeBaseHistory.id).order_by(KnowledgeBaseHistory.id).all()
            knowledge_base_ids = session.query(KnowledgeBase.history_id).distinct().order_by(KnowledgeBase.history_id).all()
            # the count of records from KnowledgeBaseHistory and KnowledgeBase tables should match
            self.assertEqual(len(knowledge_base_history_ids), len(knowledge_base_ids))

            # now remove the last two records that are for Mars and check the database
            self.app.remove_most_recent_knowledge_base_records('Antoniadi', 'Mars')

            knowledge_base_history_ids_remove_one_set = session.query(KnowledgeBaseHistory.id).order_by(KnowledgeBaseHistory.id).all()
            knowledge_base_ids_after_remove_one_set = session.query(KnowledgeBase.history_id).distinct().order_by(KnowledgeBase.history_id).all()

            self.assertEqual(knowledge_base_history_ids[:-2], knowledge_base_history_ids_remove_one_set)
            self.assertEqual(knowledge_base_ids[:-2], knowledge_base_ids_after_remove_one_set)

            # now remove the last two records that are for Moon and check the database
            self.app.remove_most_recent_knowledge_base_records('Antoniadi', 'Moon')

            knowledge_base_history_ids_remove_another_set = session.query(KnowledgeBaseHistory.id).order_by(KnowledgeBaseHistory.id).all()
            knowledge_base_ids_after_remove_another_set = session.query(KnowledgeBase.history_id).distinct().order_by(KnowledgeBase.history_id).all()

            self.assertEqual(knowledge_base_history_ids[:-4], knowledge_base_history_ids_remove_another_set)
            self.assertEqual(knowledge_base_ids[:-4], knowledge_base_ids_after_remove_another_set)

    def test_remove_all_but_most_recent_knowledge_base_records(self):
        """ remove_all_but_most_recent_knowledge_base_records method """

        # queue records to alternate between planetary and non planetary
        queue_knowledge_base_history_records = [
            knowledge_base_history_records[0],
            knowledge_base_history_records_non_planetary[0],
            knowledge_base_history_records[1],
            knowledge_base_history_records_non_planetary[1],
        ]
        queue_knowledge_base_records = [
            knowledge_base_records[0],
            knowledge_base_records_non_planetary[0],
            knowledge_base_records[1],
            knowledge_base_records_non_planetary[1]
        ]
        # insert them three times
        for i in range(3):
            # lets call knowledge_base_history parent, and knowledge_base child (one parent, many child)
            for parent, child in zip(queue_knowledge_base_history_records, queue_knowledge_base_records):
                # insert them one knowledge_base_history record at a time,
                # apply some dealy, to make sure to get a different date
                result = self.app.insert_knowledge_base_records([(parent.clone(), child)])
                self.assertTrue(result)
                time.sleep(0.1)

        with self.app.session_scope() as session:
            # retrieve all record ids from KnowledgeBaseHistory and all history_ids from KnowledgeBase tables
            knowledge_base_history_ids = session.query(KnowledgeBaseHistory.id).order_by(KnowledgeBaseHistory.id).all()
            knowledge_base_ids = session.query(KnowledgeBase.history_id).distinct().order_by(KnowledgeBase.history_id).all()
            # the count of records from KnowledgeBaseHistory and KnowledgeBase tables should match
            self.assertEqual(len(knowledge_base_history_ids), len(knowledge_base_ids))

            # now remove all the Mars records except for the most recent ones and check the database
            self.app.remove_all_but_most_recent_knowledge_base_records('Antoniadi', 'Mars')

            knowledge_base_history_ids_remove_one_set = session.query(KnowledgeBaseHistory.id).order_by(KnowledgeBaseHistory.id).all()
            knowledge_base_ids_after_remove_one_set = session.query(KnowledgeBase.history_id).distinct().order_by(KnowledgeBase.history_id).all()

            self.assertEqual(knowledge_base_history_ids[:2]+knowledge_base_history_ids[4:6]+knowledge_base_history_ids[8:],
                             knowledge_base_history_ids_remove_one_set)
            self.assertEqual(knowledge_base_ids[:2]+knowledge_base_ids[4:6]+knowledge_base_ids[8:],
                             knowledge_base_ids_after_remove_one_set)

            # now remove all the Moon records except for the most recent ones and check the database
            self.app.remove_all_but_most_recent_knowledge_base_records('Antoniadi', 'Moon')

            knowledge_base_history_ids_remove_another_set = session.query(KnowledgeBaseHistory.id).order_by(KnowledgeBaseHistory.id).all()
            knowledge_base_ids_after_remove_another_set = session.query(KnowledgeBase.history_id).distinct().order_by(KnowledgeBase.history_id).all()

            self.assertEqual(knowledge_base_history_ids[-4:], knowledge_base_history_ids_remove_another_set)
            self.assertEqual(knowledge_base_ids[-4:], knowledge_base_ids_after_remove_another_set)

    def test_insert_knowledge_base_records_no_knowledge_base_record(self):
        """ """
        knowledge_base_history = KnowledgeBaseHistory(id=None,  # Set to None for now, will be updated later
                             feature_name_entity='Antoniadi',
                             feature_type_entity='Crater',
                             target_entity='Moon',
                             named_entity_label='non planetary',
                             date=datetime.now(timezone.utc)),
        records: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]] = [(knowledge_base_history, [])]
        result = self.app.insert_knowledge_base_records(records)
        self.assertFalse(result)

    def test_insert_named_entity_records_no_named_entity_record(self):
        """ """
        named_entity_history = NamedEntityHistory(id=None,  # Set to None for now, will be updated later
                             feature_name_entity='Antoniadi',
                             feature_type_entity='Crater',
                             target_entity='Moon',
                             date=datetime.now(timezone.utc)),
        records: List[Tuple[NamedEntityHistory, List[NamedEntity]]] = [(named_entity_history, [])]
        result = self.app.insert_named_entity_records(records)
        self.assertFalse(result)

    def test_insert_named_entity_records(self):
        """ test test_insert_named_entity_records method """

        # id in named_entity_history record gets updated in the structure
        # that is not refreshed from one unittest to the next, so clone it
        history_record_lunar_antoniadi = named_entity_history_records[0].clone()
        history_record_martian_antoniadi = named_entity_history_records[1].clone()
        namedentity_records_lunar = named_entity_records[0]
        namedentity_records_martian = named_entity_records[1]

        records: List[Tuple[NamedEntityHistory, List[NamedEntity]]] = [
            (history_record_lunar_antoniadi, namedentity_records_lunar),
            (history_record_martian_antoniadi, namedentity_records_martian)
        ]

        result = self.app.insert_named_entity_records(records)
        self.assertTrue(result)

        with self.app.session_scope() as session:
            history_records = session.query(NamedEntityHistory).all()
            self.assertEqual(len(history_records), 2)
            self.assertEqual(history_records[0].feature_name_entity, 'Antoniadi')
            self.assertEqual(history_records[1].feature_name_entity, 'Antoniadi')

            namedentity_records = session.query(NamedEntity).all()
            self.assertEqual(len(namedentity_records), 6)

            namedentity_records = session.query(NamedEntity).filter(NamedEntity.history_id == history_records[0].id).all()
            self.assertEqual(len(namedentity_records), 3)

            namedentity_records = session.query(NamedEntity).filter(NamedEntity.history_id == history_records[1].id).all()
            self.assertEqual(namedentity_records[1].bibcode, '2023JGRE..12807817Y')
            self.assertEqual(namedentity_records[1].keywords, ['planum', 'lakes on mars', 'lake', 'irwin', 'fluvial',
                                                               'syrtis major planum', 'stucky', 'goudge', 'one', 'mars'])
            self.assertEqual(namedentity_records[1].special_keywords, ['craters', 'structural basins', 'cratering',
                                                                       'meteorite craters, diameters', 'planetary craters'])
            self.assertEqual(namedentity_records[1].knowledge_graph_score, 0.96)
            self.assertEqual(namedentity_records[1].paper_relevance_score, 0.8)
            self.assertEqual(namedentity_records[1].local_llm_score, 1)
            self.assertEqual(namedentity_records[1].confidence_score, 1)
            self.assertEqual(namedentity_records[1].named_entity_label, 'planetary')

    def test_get_named_entity_bibcodes(self):
        """ test get_named_entity_bibcodes method """

        # first pupulate the database
        # id in named_entity_history record gets updated in the structure
        # that is not refreshed from one unittest to the next, so clone it
        history_record_lunar_antoniadi = named_entity_history_records[0].clone()
        history_record_martian_antoniadi = named_entity_history_records[1].clone()
        namedentity_records_lunar = named_entity_records[0]
        namedentity_records_martian = named_entity_records[1]

        records: List[Tuple[NamedEntityHistory, List[NamedEntity]]] = [
            (history_record_lunar_antoniadi, namedentity_records_lunar),
            (history_record_martian_antoniadi, namedentity_records_martian)
        ]

        result = self.app.insert_named_entity_records(records)
        self.assertTrue(result)

        # next get records all of them
        results = self.app.get_named_entity_bibcodes()
        self.assertEqual(len(results), 2)

        results = self.app.get_named_entity_bibcodes(feature_name_entity='Antoniadi')
        self.assertEqual(len(results), 2)

        results = self.app.get_named_entity_bibcodes(feature_name_entity='Antoniadi', feature_type_entity='Crater')
        self.assertEqual(len(results), 2)

        results = self.app.get_named_entity_bibcodes(target_entity='Moon')
        self.assertEqual(len(results), 1)

        results = self.app.get_named_entity_bibcodes(confidence_score=1)
        self.assertEqual(len(results), 1)

        results = self.app.get_named_entity_bibcodes(date='2024-06-06')
        self.assertEqual(len(results), 2)

    def test_get_feature_ids(self):
        """ Test the get_feature_ids method """

        results = self.app.get_feature_ids()
        self.assertEqual(len(results), 2090)
        self.assertEqual(results[0], 11)
        self.assertEqual(results[2089], 16036)

    def test_add_new_usgs_entities(self):
        """ add_new_usgs_entities """

        # define dummy data with new feature names, some new and some existing targets and feature types
        data = [
            # add a new feature name for Moon Crater and Mars Albedo Feature
            {'Feature_ID': '99001', 'Clean_Feature_Name': 'new_feature_name_1', 'Target': 'Moon', 'Feature_Type': 'Crater',
             'Approval_Date': '2024', 'Approval_Status': 'Approved', 'Feature_Type_Plural': 'Craters'},
            {'Feature_ID': '99002', 'Clean_Feature_Name': 'new_feature_name_2', 'Target': 'Mars',
             'Feature_Type': 'Albedo Feature', 'Approval_Date': '2024', 'Approval_Status': 'Approved',
             'Feature_Type_Plural': ''},
            # add a new feature name for a new target and new feature type
            # one feature type has plural and the other does not
            {'Feature_ID': '99003', 'Clean_Feature_Name': 'new_feature_name_3', 'Target': 'new_target_1',
             'Feature_Type': 'new_feature_type_1', 'Approval_Date': '2024', 'Approval_Status': 'Approved',
             'Feature_Type_Plural': 'new_feature_type_1_plural'},
            {'Feature_ID': '99004', 'Clean_Feature_Name': 'new_feature_name_4', 'Target': 'new_target_2',
             'Feature_Type': 'new_feature_type_2', 'Approval_Date': '2024', 'Approval_Status': 'Approved',
             'Feature_Type_Plural': ''},
            # make a feature name ambiguous by assigning an existing feature name to another target
            # FeatureName(entity='Rayleigh', target_entity='Moon', feature_type_entity='Crater', entity_id=4966, approval_year='1964')
            {'Feature_ID': '99005', 'Clean_Feature_Name': 'Rayleigh', 'Target': 'new_target_2',
             'Feature_Type': 'Crater', 'Approval_Date': '2024', 'Approval_Status': 'Approved',
             'Feature_Type_Plural': 'Craters'}
        ]

        # call the method to add new USGS entities
        insertion_successful = self.app.add_new_usgs_entities(data)

        # check that the insertion was successful
        self.assertTrue(insertion_successful)

        # fetch the inserted targets to verify
        inserted_targets = set(self.app.get_target_entities())
        self.assertIn('new_target_1', inserted_targets)
        self.assertIn('new_target_2', inserted_targets)

        # fetch the inserted feature name
        self.assertEqual(self.app.get_feature_type_entity('Moon', 'new_feature_name_1'), 'Crater')
        self.assertEqual(self.app.get_feature_type_entity('Mars', 'new_feature_name_2'), 'Albedo Feature')
        self.assertEqual(self.app.get_feature_type_entity('new_target_1', 'new_feature_name_3'), 'new_feature_type_1')
        self.assertEqual(self.app.get_feature_type_entity('new_target_2', 'new_feature_name_4'), 'new_feature_type_2')
        self.assertEqual(self.app.get_feature_type_entity('new_target_2', 'Rayleigh'), 'Crater')

        # verify the new feature names ids are inserted
        feature_ids = self.app.get_feature_ids()
        new_feature_ids = [99001, 99002, 99003, 99004, 99005]
        for feature_id in new_feature_ids:
            self.assertIn(feature_id, feature_ids)


class TestADSPlanetaryNamesPipelineCeleryNoStubdata(unittest.TestCase):

    """
    Tests application's methods when there is no need for shared stubdata
    """

    maxDiff = None

    postgresql_url_dict = {
        'port': 5432,
        'host': '127.0.0.1',
        'user': 'postgres',
        'database': 'postgres'
    }
    postgresql_url = 'postgresql://{user}:{user}@{host}:{port}/{database}' \
        .format(user=postgresql_url_dict['user'],
                host=postgresql_url_dict['host'],
                port=postgresql_url_dict['port'],
                database=postgresql_url_dict['database']
                )

    def setUp(self):
        self.test_dir = os.path.join(project_home, 'adsplanetnamepipe/tests')
        unittest.TestCase.setUp(self)
        self.app = app.ADSPlanetaryNamesPipelineCelery('test', local_config={
            'SQLALCHEMY_URL': self.postgresql_url,
            'SQLALCHEMY_ECHO': False,
            'PROJ_HOME': project_home,
            'TEST_DIR': self.test_dir,
        })
        Base.metadata.bind = self.app._session.get_bind()
        Base.metadata.create_all()

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        Base.metadata.drop_all()
        self.app.close_app()

    def test_get_feature_name_entities_no_record(self):
        """ test get_feature_name_entities when no records are fetched """

        target_entity = "some_target"
        feature_type_entity = "some_feature_type"

        with patch.object(self.app.logger, 'error') as mock_error:
            result = self.app.get_feature_name_entities(target_entity, feature_type_entity)

            self.assertEqual(result, [])
            mock_error.assert_called_with(f"No feature name entities are found for {target_entity}/{feature_type_entity}.")

    def test_get_feature_type_entity_no_record(self):
        """ test get_feature_type_entity when no records are fetched """

        target_entity = "some_target"
        feature_name_entity = "some_feature_name"

        with patch.object(self.app.logger, 'error') as mock_error:
            result = self.app.get_feature_type_entity(target_entity, feature_name_entity)

            self.assertEqual(result, '')
            mock_error.assert_called_with(f"No feature type entity is found for {target_entity}/{feature_name_entity}.")

    def test_get_plural_feature_type_entity_no_record(self):
        """ test get_plural_feature_type_entity when no records are fetched """

        feature_name_entity = "some_feature_name"

        with patch.object(self.app.logger, 'error') as mock_error:
            result = self.app.get_plural_feature_type_entity(feature_name_entity)

            self.assertEqual(result, '')

    def test_get_context_ambiguous_feature_name_no_record(self):
        """ test get_context_ambiguous_feature_name when no records are fetched """

        feature_name_entity = "some_feature_name"

        with patch.object(self.app.logger, 'info') as mock_info:
            result = self.app.get_context_ambiguous_feature_name(feature_name_entity)

            self.assertEqual(result, [])
            mock_info.assert_called_with(f"No context is found for entity {feature_name_entity}.")

    def test_get_multi_token_containing_feature_name_no_record(self):
        """ test get_multi_token_containing_feature_name when no records are fetched """

        feature_name_entity = "some_feature_name"

        with patch.object(self.app.logger, 'info') as mock_info:
            result = self.app.get_multi_token_containing_feature_name(feature_name_entity)

            self.assertEqual(result, [])
            mock_info.assert_called_with(f"No multi token entity is found for entity {feature_name_entity}.")

    def test_get_named_entity_label_no_record(self):
        """ test get_named_entity_label when no records are fetched """

        with patch.object(self.app.logger, 'error') as mock_error:
            result = self.app.get_named_entity_label()

            self.assertEqual(result, [])
            mock_error.assert_called_with("Unable to fetch the named entity label records.")

    def test_get_target_entities_no_record(self):
        """ test get_target_entities when no records are fetched """

        with patch.object(self.app.logger, 'error') as mock_error:
            result = self.app.get_target_entities()

            self.assertEqual(result, [])
            mock_error.assert_called_with("Unable to fetch the target records.")

    def test_insert_knowledge_base_records_exception(self):
        """ test insert_knowledge_base_records method when there is a exception """

        knowledge_base_list = [(KnowledgeBaseHistory(None, None, None, None, None),
                                [KnowledgeBase(None, None, None, None, None, None, None)])]

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.flush.side_effect = SQLAlchemyError("Mocked SQLAlchemyError")

            with patch.object(self.app.logger, 'error') as mock_error:
                result = self.app.insert_knowledge_base_records(knowledge_base_list)

                self.assertFalse(result)
                mock_session.rollback.assert_called_once()
                mock_error.assert_called_once_with("Error occurred while inserting `KnowledgeBaseHistory` and `KnowledgeBase` records: Mocked SQLAlchemyError")

    def test_get_knowledge_base_keywords_no_record(self):
        """ get_knowledge_base_keywords method when there are no records """

        result = self.app.get_knowledge_base_keywords('Antoniadi', 'Crater', 'Moon', 'planetary')
        self.assertEqual(result, [])

    def test_append_to_knowledge_base_keywords_exception(self):
        """ test append_to_knowledge_base_keywords method  when there is a exception """

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.commit.side_effect = SQLAlchemyError("Mocked SQLAlchemyError")

            with patch.object(self.app.logger, 'error') as mock_error:
                count = self.app.append_to_knowledge_base_keywords('Antoniadi', 'Moon', 'larger')

                self.assertEqual(count, -1)
                mock_session.rollback.assert_called_once()
                mock_error.assert_called_once_with("Error occurred while appending keyword `larger` for feature name `Antoniadi` and target `Moon` for the `KnowledgeBase` records: Mocked SQLAlchemyError")

    def test_remove_from_knowledge_base_keywords_exception(self):
        """ test remove_from_knowledge_base_keywords method when there is a exception """

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.commit.side_effect = SQLAlchemyError("Mocked SQLAlchemyError")

            with patch.object(self.app.logger, 'error') as mock_error:
                count = self.app.remove_from_knowledge_base_keywords('Antoniadi', 'Moon', 'larger')

                self.assertEqual(count, -1)
                mock_session.rollback.assert_called_once()
                mock_error.assert_called_once_with("Error occurred while removing keyword `larger` for feature name `Antoniadi` and target `Moon` from the `KnowledgeBase` records: Mocked SQLAlchemyError")

    def test_remove_most_recent_knowledge_base_records_exception(self):
        """ test remove_most_recent_knowledge_base_records method when there is a exception """

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.commit.side_effect = SQLAlchemyError("Mocked SQLAlchemyError")

            with patch.object(self.app.logger, 'error') as mock_error:
                counts = self.app.remove_most_recent_knowledge_base_records('Antoniadi', 'Moon')

                self.assertEqual(counts, (-1, -1))
                mock_session.rollback.assert_called_once()
                mock_error.assert_called_once_with("Error occurred while deleting `KnowledgeBaseHistory` and `KnowledgeBase` records: Mocked SQLAlchemyError")

    def test_remove_all_but_most_recent_knowledge_base_records_exception(self):
        """ test remove_all_but_most_recent_knowledge_base_records method when there is a exception """

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.commit.side_effect = SQLAlchemyError("Mocked SQLAlchemyError")

            with patch.object(self.app.logger, 'error') as mock_error:
                counts = self.app.remove_all_but_most_recent_knowledge_base_records('Antoniadi', 'Moon')

                self.assertEqual(counts, (-1, -1))
                mock_session.rollback.assert_called_once()
                mock_error.assert_called_once_with("Error occurred while deleting `KnowledgeBaseHistory` and `KnowledgeBase` records: Mocked SQLAlchemyError")

    def test_insert_named_entity_records_exception(self):
        """ test insert_named_entity_records method when there is a exception """

        named_entity_list = [(NamedEntityHistory(None, None, None, None),
                                [NamedEntity(None, None, None, None, None, None, None, None, None, None, None, None)])]

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.commit.side_effect = SQLAlchemyError("Mocked SQLAlchemyError")

            with patch.object(self.app.logger, 'error') as mock_error:
                result = self.app.insert_named_entity_records(named_entity_list)

                self.assertFalse(result)
                mock_session.rollback.assert_called_once()
                mock_error.assert_called_once_with("Error occurred while inserting `NamedEntityHistory` and `NamedEntity` records: Mocked SQLAlchemyError")

    def test_get_named_entity_bibcodes_no_records(self):
        """ test get_named_entity_bibcodes method when no records """

        results = self.app.get_named_entity_bibcodes()
        self.assertEqual(results, [])

    def test_get_feature_ids_when_empty_table(self):
        """ test the get_feature_ids method when no records """

        result = self.app.get_feature_ids()
        self.assertEqual(result, [])


    def test_insert_target_entities_exception(self):
        """ test insert_target_entities method when there is an exception """

        target_list = ["target_1", "target_2"]

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.bulk_save_objects.side_effect = SQLAlchemyError("Mocked SQLAlchemyError")

            with patch.object(self.app.logger, 'error') as mock_error:
                result = self.app.insert_target_entities(target_list)

                self.assertFalse(result)
                mock_session.rollback.assert_called_once()
                mock_error.assert_called_once_with("Error occurred while inserting new target entities: Mocked SQLAlchemyError")

    def test_insert_feature_types_exception(self):
        """ test insert_feature_types method when there is an exception """

        feature_type_list = [
            {'Feature_Type': 'feature_type_1', 'Target': 'target_1', 'Feature_Type_Plural': 'feature_type_1_plural'}
        ]

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.bulk_save_objects.side_effect = SQLAlchemyError("Mocked SQLAlchemyError")

            with patch.object(self.app.logger, 'error') as mock_error:
                result = self.app.insert_feature_types(feature_type_list)

                self.assertFalse(result)
                mock_session.rollback.assert_called_once()
                mock_error.assert_called_once_with("Error occurred while inserting new feature types: Mocked SQLAlchemyError")

    def test_insert_new_usgs_nomenclature_entities_exception(self):
        """ test insert_new_usgs_nomenclature_entities method when there is an exception """

        feature_name_list = ["feature_name_1", "feature_name_2"]

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.bulk_save_objects.side_effect = SQLAlchemyError("Mocked SQLAlchemyError")

            with patch.object(self.app.logger, 'error') as mock_error:
                result = self.app.insert_new_usgs_nomenclature_entities(feature_name_list)

                self.assertFalse(result)
                mock_session.rollback.assert_called_once()
                mock_error.assert_called_once_with("Error inserting usgs_nomenclature entities: Mocked SQLAlchemyError")

    def test_insert_feature_name_records_exception(self):
        """ test insert_feature_name_records method when there is an exception """

        feature_name_list = [
            {'Clean_Feature_Name': 'new_feature_name_1', 'Target': 'target_1', 'Feature_Type': 'feature_type_1', 'Feature_ID': '99001', 'Approval_Date': '2024'}
        ]

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.bulk_save_objects.side_effect = SQLAlchemyError("Mocked SQLAlchemyError")

            with patch.object(self.app.logger, 'error') as mock_error:
                result = self.app.insert_feature_name_records(feature_name_list)

                self.assertFalse(result)
                mock_session.rollback.assert_called_once()
                mock_error.assert_called_once_with("Error inserting feature name records: Mocked SQLAlchemyError")

    def test_no_new_feature_name_entities_needed(self):
        """ test insert_new_usgs_nomenclature_entities when no new entries are needed """

        feature_name_list = ["existing_feature_name_1", "existing_feature_name_2"]

        with patch.object(self.app, "session_scope") as mock_session_scope:
            mock_session = mock_session_scope.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.all.return_value = [
                USGSNomenclature(entity="existing_feature_name_1"),
                USGSNomenclature(entity="existing_feature_name_2")
            ]

            with patch.object(self.app.logger, 'debug') as mock_debug:
                result = self.app.insert_new_usgs_nomenclature_entities(feature_name_list)

                self.assertTrue(result)
                mock_debug.assert_called_once_with("No new feature name entities needed to be added to `usgs_nomenclature`.")

    @patch.object(app.ADSPlanetaryNamesPipelineCelery, 'insert_target_entities', return_value=True)
    @patch.object(app.ADSPlanetaryNamesPipelineCelery, 'insert_new_usgs_nomenclature_entities', return_value=False)
    @patch.object(app.ADSPlanetaryNamesPipelineCelery, 'insert_feature_types')
    @patch.object(app.ADSPlanetaryNamesPipelineCelery, 'insert_feature_name_records')
    def test_add_new_usgs_entities_returns_false(self, mock_insert_feature_name_records, mock_insert_feature_types,
                                                 mock_insert_usgs_nomenclature_entities, mock_insert_target_entities):
        """ test add_new_usgs_entities method when usgs_nomenclature insertion fails, causing the function to return False """

        data = [
            {'Clean_Feature_Name': 'new_feature_name_1', 'Target': 'target_1', 'Feature_Type': 'feature_type_1', 'Feature_ID': '99001', 'Approval_Date': '2024', 'Feature_Type_Plural': 'feature_type_1_plural'}
        ]

        # call the method to add new USGS entities
        insertion_successful = self.app.add_new_usgs_entities(data)

        # check that the insertion was unsuccessful
        self.assertFalse(insertion_successful)

        # confirm that the appropriate insert methods were called
        mock_insert_target_entities.assert_called_once()
        mock_insert_usgs_nomenclature_entities.assert_called_once_with({'new_feature_name_1'})
        mock_insert_feature_types.assert_not_called()
        mock_insert_feature_name_records.assert_not_called()


if __name__ == '__main__':
    unittest.main()
