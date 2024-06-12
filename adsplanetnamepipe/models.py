# -*- coding: utf-8 -*-

import copy

from datetime import datetime, timezone

from sqlalchemy import Integer, String, Column, ForeignKey, DateTime, func, Float, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Collection(Base):
    """
    lookup table `collection`
    the column database contains the followings: astronomy, physics, general, earth science
    """
    __tablename__ = 'collection'

    database = Column(String(44), primary_key=True)

    def __init__(self, database):
        """

        :param database:
        """
        self.database = database


class SortCollection():
    """

    """
    order = ['astronomy', 'physics', 'general', 'earth science']

    def sort_and_concat(self, database):
        """
        sort the list in the accepted order and return a string of all databases

        :param database:
        :return:
        """
        try:
            return ', '.join([item for item in self.order if item in database])
        except:
            return ''


class USGSNomenclature(Base):
    """
    lookup table `usgs_nomenclature`
    the entity is a feature names, but not tied to any target/feature type
    hence, all the unique feature names
    """
    __tablename__ = 'usgs_nomenclature'

    entity = Column(String(32), primary_key=True)

    def __init__(self, entity):
        """

        :param database:
        """
        self.entity = entity


class Target(Base):
    """
    lookup table for target entities
    ie, Moon, Mars, etc
    """
    __tablename__ = 'target'

    entity = Column(String(32), primary_key=True)

    def __init__(self, entity):
        """

        :param entity:
        """
        self.entity = entity


class FeatureType(Base):
    """
    lookup table for feature type entities tied to target
    ie, Craters on the Moon, Albedo Feature on Mars, etc.
    """
    __tablename__ = 'feature_type'

    entity = Column(String(32), primary_key=True)
    target_entity = Column(String(32), ForeignKey('target.entity'), primary_key=True)
    plural_entity = Column(String(32))

    def __init__(self, entity, target_entity, plural_entity):
        """

        :param entity:
        :param target_entity:
        :param plural_entity:
        """
        self.entity = entity
        self.target_entity = target_entity
        self.plural_entity = plural_entity


class FeatureName(Base):
    """
    lookup table for feature name entities, tied to target and feature type
    ie, Apollo/Moon/Crater, Arabia/Mars/Albedo Feature, etc.
    """
    __tablename__ = 'feature_name'
    __table_args__ = (ForeignKeyConstraint(['entity'], ['usgs_nomenclature.entity']),
                      ForeignKeyConstraint(['feature_type_entity', 'target_entity'],
                                           ['feature_type.entity', 'feature_type.target_entity']),
                      )

    entity = Column(String(32), primary_key=True)
    feature_type_entity = Column(String(32), primary_key=True)
    target_entity = Column(String(32), primary_key=True)

    def __init__(self, entity, target_entity, feature_type_entity):
        """

        :param entity:
        :param target_entity:
        :param feature_type_entity:
        """
        self.entity = entity
        self.target_entity = target_entity
        self.feature_type_entity = feature_type_entity


class FeatureNameContext(Base):
    """
    lookup table `FeatureNameContext`
    the column context holds the different contexts in which ambiguous feature names can appear,
    ie, asteroid, crater, etc.
    """
    __tablename__ = 'feature_name_context'

    context = Column(String(64), primary_key=True)

    def __init__(self, context):
        """

        :param database:
        """
        self.context = context


class AmbiguousFeatureName(Base):
    """
    lookup table that lists the feature names along with their multiple associated contexts
    """
    __tablename__ = 'ambiguous_feature_name'
    __table_args__ = (ForeignKeyConstraint(['entity'], ['usgs_nomenclature.entity']),
                      ForeignKeyConstraint(['context'], ['feature_name_context']))

    entity = Column(String(32), primary_key=True)
    context = Column(String(64), primary_key=True)

    def __init__(self, entity, context):
        """

        :param entity:
        :param context:
        """
        self.entity = entity
        self.context = context


class MultiTokenFeatureName(Base):
    """
    lookup table that lists the feature names along with their multi token feature names that it contains it
    """
    __tablename__ = 'multi_token_feature_name'

    entity = Column(String(255), primary_key=True)
    multi_token_entity = Column(String(255), primary_key=True)

    def __init__(self, entity, multi_token_entity):
        """

        :param entity:
        :param multi_token_entity:
        """
        self.entity = entity
        self.multi_token_entity = multi_token_entity


class NamedEntityLabel(Base):
    """
    lookup table `named_entity_label`
    the column label contains the followings: planetary, unknown
    """
    __tablename__ = 'named_entity_label'

    label = Column(String(32), primary_key=True)

    def __init__(self, label):
        """

        :param label:
        """
        self.label = label

    def toJSON(self):
        """
        :return: values formatted as python dict
        """
        return {
            'label': self.label,
            'value': 1 if self.label == 'planetary' else 0
        }


class KnowledgeBaseHistory(Base):
    """
    This table holds the historical data for knowledge base.
    """
    __tablename__ = 'knowledge_base_history'
    __table_args__ = (ForeignKeyConstraint(
        ['feature_name_entity', 'feature_type_entity', 'target_entity'],
        ['feature_name.entity', 'feature_name.feature_type_entity', 'feature_name.target_entity']
    ), ForeignKeyConstraint(
        ['named_entity_label'], ['named_entity_label.label']
    ),)

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    feature_name_entity = Column(String(32), primary_key=True)
    feature_type_entity = Column(String(32), primary_key=True)
    target_entity = Column(String(32), primary_key=True)
    date = Column(DateTime(timezone=True), nullable=False)
    named_entity_label = Column(String(32), nullable=False)

    def __init__(self, id, feature_name_entity, feature_type_entity, target_entity, named_entity_label, date=None):
        """

        :param id:
        :param feature_name_entity:
        :param feature_type_entity:
        :param target_entity:
        :param named_entity_label:
        :param date:
        """
        self.id = id
        self.feature_name_entity = feature_name_entity
        self.feature_type_entity = feature_type_entity
        self.target_entity = target_entity
        if not date:
            self.date = datetime.now(timezone.utc)
        else:
            self.date = date
        self.named_entity_label = named_entity_label

    def __copy__(self):
        """
        create a new instance and copy the data

        :return:
        """
        new_instance = type(self)(
            self.id,
            self.feature_name_entity,
            self.feature_type_entity,
            self.target_entity,
            self.named_entity_label,
            datetime.now(timezone.utc)      # give a current date
        )
        return new_instance

    def clone(self):
        """
        provide a public method to get a shallow copy

        this is used for unittest, creating multiple copies, to then
        remove most recent, or remove all but most recent

        :return:
        """
        return copy.copy(self)

    # def toJSON(self):
    #     """
    #     :return: values formatted as python dict
    #     """
    #     return {
    #         'id': self.id,
    #         'feature_name': self.feature_name_entity,
    #         'feature_type': self.feature_type_entity,
    #         'target': self.target_entity,
    #         'named_entity_label': self.named_entity_label,
    #         'date': self.date,
    #     }


class KnowledgeBase(Base):
    """
    This table holds the data for knowledge base, obtained during the collect step to create knowledge graph.
    """
    __tablename__ = 'knowledge_base'
    __table_args__ = (ForeignKeyConstraint(
        ['history_id'], ['knowledge_base_history.id']
    ), ForeignKeyConstraint(
        ['database'], ['collection.database']
    ), )

    history_id = Column(Integer, ForeignKey('knowledge_base_history.id'), primary_key=True)
    bibcode = Column(String(19), primary_key=True)
    database = Column(String(44), nullable=False)
    excerpt = Column(String(2000), nullable=True)
    keywords_item_id = Column(Integer, primary_key=True)
    keywords = Column(ARRAY(String(64)), nullable=False)
    special_keywords = Column(ARRAY(String(64)), nullable=True)

    def __init__(self, history_id, bibcode, database, excerpt, keywords_item_id, keywords, special_keywords):
        """

        :param history_id:
        :param bibcode:
        :param database:
        :param excerpt:
        :param keywords_item_id:
        :param keywords:
        :param special_keywords:
        """
        self.history_id = history_id
        self.bibcode = bibcode
        self.database = SortCollection().sort_and_concat(database)
        self.excerpt = excerpt[:2000] if excerpt else None # just in case > 2000
        self.keywords_item_id = keywords_item_id
        self.keywords = keywords
        self.special_keywords = special_keywords

    # def toJSON(self):
    #     """
    #     :return: values formatted as python dict
    #     """
    #     return {
    #         'history_id': self.history_id,
    #         'bibcode': self.bibcode,
    #         'database': self.database,
    #         'excerpt': self.excerpt,
    #         'keywords_item_id': self.keywords_item_id,
    #         'keywords': self.keywords,
    #         'special_keywords': self.special_keywords,
    #     }


class NamedEntityHistory(Base):
    """
    This table holds the historical data for named entity.
    """
    __tablename__ = 'named_entity_history'
    __table_args__ = (ForeignKeyConstraint(
        ['feature_name_entity', 'feature_type_entity', 'target_entity'],
        ['feature_name.entity', 'feature_name.feature_type_entity', 'feature_name.target_entity']
    ),)

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    feature_name_entity = Column(String(32), primary_key=True)
    feature_type_entity = Column(String(32), primary_key=True)
    target_entity = Column(String(32), primary_key=True)
    date = Column(DateTime(timezone=True), nullable=False)

    def __init__(self, id, feature_name_entity, feature_type_entity, target_entity, date=None):
        """

        :param id:
        :param feature_name_entity:
        :param feature_type_entity:
        :param target_entity:
        :param date:
        """
        self.id = id
        self.feature_name_entity = feature_name_entity
        self.feature_type_entity = feature_type_entity
        self.target_entity = target_entity
        if not date:
            self.date = datetime.now(timezone.utc)
        else:
            self.date = date

    def __copy__(self):
        """
        create a new instance and copy the data

        :return:
        """
        new_instance = type(self)(
            self.id,
            self.feature_name_entity,
            self.feature_type_entity,
            self.target_entity,
            datetime.now(timezone.utc)      # give a current date
        )
        return new_instance

    def clone(self):
        """
        provide a public method to get a shallow copy

        this is used for unittest

        :return:
        """
        return copy.copy(self)



class NamedEntity(Base):
    """
    This table holds the data for named entities identified, obtained during the identity step.
    """
    __tablename__ = 'named_entity'
    __table_args__ = (ForeignKeyConstraint(
        ['history_id'], ['named_entity_history.id']
    ), ForeignKeyConstraint(
        ['database'], ['collection.database']
    ), ForeignKeyConstraint(
        ['named_entity_label'], ['named_entity_label.label']
    ),)

    history_id = Column(Integer, ForeignKey('named_entity_history.id'), primary_key=True)
    bibcode = Column(String(19), primary_key=True)
    database = Column(String(44), nullable=False)
    excerpt = Column(String(2000), nullable=False)
    keywords_item_id = Column(Integer, primary_key=True)
    keywords = Column(ARRAY(String(64)), nullable=False)
    special_keywords = Column(ARRAY(String(64)), nullable=True)
    knowledge_graph_score = Column(Float, nullable=False)
    paper_relevance_score = Column(Float, nullable=False)
    local_llm_score = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    named_entity_label = Column(String(32), nullable=False)

    def __init__(self, history_id, bibcode, database, excerpt, keywords_item_id, keywords, special_keywords,
                 knowledge_graph_score, paper_relevance_score, local_llm_score, confidence_score, named_entity_label):
        """

        :param history_id:
        :param bibcode:
        :param database:
        :param excerpt:
        :param keywords_item_id:
        :param keywords:
        :param special_keywords:
        :param knowledge_graph_score:
        :param paper_relevance_score:
        :param local_llm_score:
        :param confidence_score:
        :param named_entity_label:
        """
        self.history_id = history_id
        self.bibcode = bibcode
        self.database = SortCollection().sort_and_concat(database)
        self.excerpt = excerpt[:2000] if excerpt else None # just in case > 2000
        self.keywords_item_id = keywords_item_id
        self.keywords = keywords
        self.special_keywords = special_keywords
        self.knowledge_graph_score = knowledge_graph_score
        self.paper_relevance_score = paper_relevance_score
        self.local_llm_score = local_llm_score
        self.confidence_score = confidence_score
        self.named_entity_label = named_entity_label