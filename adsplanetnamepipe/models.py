# -*- coding: utf-8 -*-

import copy

from datetime import datetime, timezone
from typing import Tuple

from sqlalchemy import Integer, String, Column, ForeignKey, DateTime, Float, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Collection(Base):
    """
    lookup table `collection`
    the column database contains the followings ads collections: astronomy, physics, general, earth science
    """
    __tablename__ = 'collection'

    # the name of the ads collections, serving as the primary key
    database = Column(String(44), primary_key=True)

    def __init__(self, database: str):
        """
        initialize a new Collection instance

        :param database: the name of collection to be added
        """
        self.database = database


class SortCollection():
    """
    a utility class for sorting and concatenating database names
    """

    # a predefined order of ads collections
    order = ['astronomy', 'physics', 'general', 'earth science']

    def sort_and_concat(self, database: list) -> str:
        """
        sort the list in the accepted order and return a string of all databases

        :param database: a list of ads collections to be sorted and concatenated
        :return: a comma-separated string of sorted ads collections
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

    # the name of the entity, serving as the primary key
    entity = Column(String(32), primary_key=True)

    def __init__(self, entity: str):
        """
        initialize a new USGSNomenclature instance

        :param entity: the name of the entity to be added
        """
        self.entity = entity


class Target(Base):
    """
    lookup table for target entities
    ie, Moon, Mars, etc
    """
    __tablename__ = 'target'

    # the name of the target entity, serving as the primary key
    entity = Column(String(32), primary_key=True)

    def __init__(self, entity: str):
        """
        initialize a new Target instance

        :param entity: the name of the target entity to be added
        """
        self.entity = entity


class FeatureType(Base):
    """
    lookup table for feature type entities tied to target
    ie, Craters on the Moon, Albedo Feature on Mars, etc.
    """
    __tablename__ = 'feature_type'

    # the name of the feature type, serving as part of the primary key
    entity = Column(String(32), primary_key=True)
    # the name of the target entity, serving as part of the primary key
    target_entity = Column(String(32), ForeignKey('target.entity'), primary_key=True)
    # the plural form of the feature type entity, both singular and plural should be considered
    plural_entity = Column(String(32))

    def __init__(self, entity: str, target_entity: str, plural_entity: str):
        """
        initialize a new FeatureType instance

        :param entity: the name of the feature type
        :param target_entity: the name of the target entity associated with the feature type entity
        :param plural_entity: the plural form of the feature type entity
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

    # the name of the feature, serving as part of the primary key
    entity = Column(String(32), primary_key=True)
    # the type of the feature, serving as part of the primary key
    feature_type_entity = Column(String(32), primary_key=True)
    # the target entity, serving as part of the primary key
    target_entity = Column(String(32), primary_key=True)
    # a unique identifier for the feature name, required
    entity_id = Column(Integer, nullable=False)
    # the year the feature was approved, only the year is stored, required
    approval_year = Column(String(4), nullable=False)

    def __init__(self, entity: str, target_entity: str, feature_type_entity: str, entity_id: int, approval_year: str):
        """
        initialize a new FeatureName instance

        :param entity: the name of the feature
        :param target_entity: the name of the target entity associated with this feature name
        :param feature_type_entity: the name of feature type associated with this feature name
        :param entity_id: unique identifier associated with this feature name
        :param approval_year: the year the feature name was approved, stored as a four-digit string (e.g., '2000')
        """
        self.entity = entity
        self.target_entity = target_entity
        self.feature_type_entity = feature_type_entity
        self.entity_id = entity_id
        self.approval_year = approval_year

class FeatureNameContext(Base):
    """
    lookup table `FeatureNameContext`
    the column context holds the different contexts in which ambiguous feature names can appear,
    ie, asteroid, crater, etc.
    """
    __tablename__ = 'feature_name_context'

    # the context in which a feature name can appear, serving as the primary key
    context = Column(String(64), primary_key=True)

    def __init__(self, context: str):
        """
        initialize a new FeatureNameContext instance

        :param context: the context to be added to the lookup table
        """
        self.context = context


class AmbiguousFeatureName(Base):
    """
    lookup table that lists the feature names along with their multiple associated contexts
    """
    __tablename__ = 'ambiguous_feature_name'
    __table_args__ = (ForeignKeyConstraint(['entity'], ['usgs_nomenclature.entity']),
                      ForeignKeyConstraint(['context'], ['feature_name_context']))

    # the name of the feature, serving as part of the primary key
    entity = Column(String(32), primary_key=True)
    # the context associated with the feature name, serving as part of the primary key
    context = Column(String(64), primary_key=True)

    def __init__(self, entity: str, context: str):
        """
        initialize a new AmbiguousFeatureName instance

        :param entity: the name of the feature
        :param context: the context associated with the feature name
        """
        self.entity = entity
        self.context = context


class MultiTokenFeatureName(Base):
    """
    lookup table that lists the feature names along with their multi token feature names that it contains
    ie, Apollo and Apollo Patera or Arabia and Arabia Terra, etc
    """
    __tablename__ = 'multi_token_feature_name'

    # the single-token feature name, serving as part of the primary key
    entity = Column(String(255), primary_key=True)
    # the multi-token feature name containing the entity, serving as part of the primary key
    multi_token_entity = Column(String(255), primary_key=True)

    def __init__(self, entity: str, multi_token_entity: str):
        """
        initialize a new MultiTokenFeatureName instance

        :param entity: the single-token feature name
        :param multi_token_entity: the multi-token feature name containing the entity
        """
        self.entity = entity
        self.multi_token_entity = multi_token_entity


class NamedEntityLabel(Base):
    """
    lookup table `named_entity_label`
    the column label contains the followings: planetary, unknown
    """
    __tablename__ = 'named_entity_label'

    # the label for the named entity, serving as the primary key
    label = Column(String(32), primary_key=True)

    def __init__(self, label: str):
        """
        initialize a new NamedEntityLabel instance

        :param label: the label to be added to the lookup table
        """
        self.label = label

    def toJSON(self) -> dict:
        """
        convert the NamedEntityLabel instance to a JSON-compatible dictionary

        :return: a dictionary representation of the NamedEntityLabel instance
        """
        return {
            'label': self.label,
            'value': 1 if self.label == 'planetary' else 0
        }

    @staticmethod
    def sort_key(item: 'NamedEntityLabel') -> Tuple[int, str]:
        """
        key function for sorting NamedEntityLabel instances

        :param item: a NamedEntityLabel instance
        :return: a tuple used for sorting, ensuring 'planetary' comes first
        """
        return (0 if item.label == 'planetary' else 1, item.label)


class KnowledgeBaseHistory(Base):
    """
    this table holds the historical data for knowledge base
    """
    __tablename__ = 'knowledge_base_history'
    __table_args__ = (ForeignKeyConstraint(
        ['feature_name_entity', 'feature_type_entity', 'target_entity'],
        ['feature_name.entity', 'feature_name.feature_type_entity', 'feature_name.target_entity']
    ), ForeignKeyConstraint(
        ['named_entity_label'], ['named_entity_label.label']
    ),)

    # the unique identifier for the KB history entry, serving as the primary key
    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    # the name of the feature, serving as part of the primary key
    feature_name_entity = Column(String(32), primary_key=True)
    # the type of the feature, serving as part of the primary key
    feature_type_entity = Column(String(32), primary_key=True)
    # the target entity, serving as part of the primary key
    target_entity = Column(String(32), primary_key=True)
    # the date and time of the KB history entry
    date = Column(DateTime(timezone=True), nullable=False)
    # the label of the named entity associated with this KB history entry
    named_entity_label = Column(String(32), nullable=False)

    def __init__(self, id: int, feature_name_entity: str, feature_type_entity: str, target_entity: str,
                       named_entity_label: str, date: datetime = None):
        """
        initialize a new KnowledgeBaseHistory instance

        :param id: the unique identifier for the KB history entry
        :param feature_name_entity: the name of the feature
        :param feature_type_entity: the type of the feature
        :param target_entity: the target entity
        :param named_entity_label: the label of the named entity
        :param date: the date and time of the history entry, defaults to the current UTC time if not provided
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

    def __copy__(self) -> 'KnowledgeBaseHistory':
        """
        create a new instance and copy the data

        :return: a new instance with copied data and the current UTC time as the date
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

    def clone(self) -> 'KnowledgeBaseHistory':
        """
        provide a public method to get a shallow copy

        this is used for unittest, creating multiple copies, to then
        remove most recent, or remove all but most recent

        :return: a shallow copy of the current instance
        """
        return copy.copy(self)

    # def toJSON(self) -> dict:
    #     """
    #     convert the KnowledgeBaseHistory instance to a JSON-compatible dictionary
    #
    #     :return: a dictionary representation of the KnowledgeBaseHistory instance
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

    # the ID of the associated history entry, serving as part of the primary key
    history_id = Column(Integer, ForeignKey('knowledge_base_history.id'), primary_key=True)
    # the bibcode of the record, serving as part of the primary key
    bibcode = Column(String(19), primary_key=True)
    # the sorted and concatenated string of ads collections
    database = Column(String(44), nullable=False)
    # an excerpt extracted around the entity name from the associated record's fulltext, limited to 2000 characters
    # (collecting 128 words, assuming a word is 15 characters, on the average)
    excerpt = Column(String(2000), nullable=True)
    # the ID (order # of the excerpt extracted from the record) of the keywords/excerpt extracted from fulltext,
    # serving as part of the primary key
    keywords_item_id = Column(Integer, primary_key=True)
    # a list of keywords extracted from the excerpt
    keywords = Column(ARRAY(String(64)), nullable=False)
    # a list of keywords extracted from the excerpt by NASA Concepts library
    special_keywords = Column(ARRAY(String(64)), nullable=True)

    def __init__(self, history_id: int, bibcode: str, database: str, excerpt: str,
                       keywords_item_id: int, keywords: list, special_keywords: list):
        """
        initialize a new KnowledgeBase instance

        :param history_id: the ID of the associated history entry
        :param bibcode: the bibcode of the record
        :param database: the ads collection(s) associated with the record
        :param excerpt: an excerpt extracted around the entity name from the record's fulltext
        :param keywords_item_id: the ID (order # of the excerpt extracted from the record) of the keywords/excerpt extracted from fulltext
        :param keywords: a list of keywords extracted from the excerpt
        :param special_keywords: a list of keywords extracted from the excerpt by NASA Concepts library
        """
        self.history_id = history_id
        self.bibcode = bibcode
        self.database = SortCollection().sort_and_concat(database)
        self.excerpt = excerpt[:2000] if excerpt else None # just in case > 2000
        self.keywords_item_id = keywords_item_id
        self.keywords = keywords
        self.special_keywords = special_keywords

    # def toJSON(self) -> dict:
    #     """
    #     convert the KnowledgeBase instance to a JSON-compatible dictionary
    #
    #     :return: a dictionary representation of the KnowledgeBase instance
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
    this table holds the historical data for named entity
    """
    __tablename__ = 'named_entity_history'
    __table_args__ = (ForeignKeyConstraint(
        ['feature_name_entity', 'feature_type_entity', 'target_entity'],
        ['feature_name.entity', 'feature_name.feature_type_entity', 'feature_name.target_entity']
    ),)

    # the unique identifier for the history entry, serving as the primary key
    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    # the name of the feature, serving as part of the primary key
    feature_name_entity = Column(String(32), primary_key=True)
    # the type of the feature, serving as part of the primary key
    feature_type_entity = Column(String(32), primary_key=True)
    # the target entity, serving as part of the primary key
    target_entity = Column(String(32), primary_key=True)
    # the date and time of the history entry
    date = Column(DateTime(timezone=True), nullable=False)

    def __init__(self, id: int, feature_name_entity: str, feature_type_entity: str, target_entity: str,
                       date: datetime = None):
        """
        initialize a new NamedEntityHistory instance

        :param id: the unique identifier for the history entry
        :param feature_name_entity: the name of the feature
        :param feature_type_entity: the type of the feature
        :param target_entity: the target entity
        :param date: the date and time of the history entry, defaults to the current UTC time if not provided
        """
        self.id = id
        self.feature_name_entity = feature_name_entity
        self.feature_type_entity = feature_type_entity
        self.target_entity = target_entity
        if not date:
            self.date = datetime.now(timezone.utc)
        else:
            self.date = date

    def __copy__(self) -> 'NamedEntityHistory':
        """
        create a new instance and copy the data

        :return: a new instance with copied data and the current UTC time as the date
        """
        new_instance = type(self)(
            self.id,
            self.feature_name_entity,
            self.feature_type_entity,
            self.target_entity,
            datetime.now(timezone.utc)      # give a current date
        )
        return new_instance

    def clone(self) -> 'NamedEntityHistory':
        """
        provide a public method to get a shallow copy

        this is used for unittest

        :return: a shallow copy of the current instance
        """
        return copy.copy(self)


class NamedEntity(Base):
    """
    this table holds the data for named entities identified, obtained during the identity step
    """
    __tablename__ = 'named_entity'
    __table_args__ = (ForeignKeyConstraint(
        ['history_id'], ['named_entity_history.id']
    ), ForeignKeyConstraint(
        ['database'], ['collection.database']
    ), ForeignKeyConstraint(
        ['named_entity_label'], ['named_entity_label.label']
    ),)

    # the ID of the associated history entry, serving as part of the primary key
    history_id = Column(Integer, ForeignKey('named_entity_history.id'), primary_key=True)
    # the bibcode of the record, serving as part of the primary key
    bibcode = Column(String(19), primary_key=True)
    # the sorted and concatenated string of ads collection(s)
    database = Column(String(44), nullable=False)
    # an excerpt extracted around the entity name from the associated record's fulltext, limited to 2000 characters
    # (collecting 128 words, assuming a word is 15 characters, on the average)
    excerpt = Column(String(2000), nullable=False)
    # the ID (order # of the excerpt extracted from the record) of the keywords/excerpt extracted from fulltext,
    # serving as part of the primary key
    keywords_item_id = Column(Integer, primary_key=True)
    # a list of keywords extracted from the excerpt
    keywords = Column(ARRAY(String(64)), nullable=False)
    # a list of keywords extracted from the excerpt by NASA Concepts library
    special_keywords = Column(ARRAY(String(64)), nullable=True)
    # the relevance score computed from the knowledge graph
    knowledge_graph_score = Column(Float, nullable=False)
    # the relevance score computed from the paper characteristics
    paper_relevance_score = Column(Float, nullable=False)
    # the relevance score as reported by the local llm
    local_llm_score = Column(Float, nullable=False)
    # the overall confidence score
    confidence_score = Column(Float, nullable=False)
    # the label of the named entity as determined by the three scores (ie, planetary or unknown)
    named_entity_label = Column(String(32), nullable=False)

    def __init__(self, history_id: int, bibcode: str, database: str, excerpt: str,
                 keywords_item_id: int, keywords: list, special_keywords: list,
                 knowledge_graph_score: float, paper_relevance_score: float, local_llm_score: float,
                 confidence_score: float, named_entity_label: str):
        """
        initialize a new NamedEntity instance

        :param history_id: the ID of the associated history entry
        :param bibcode: the bibcode of the solr record
        :param database: the ads collection(s) associated with the record
        :param excerpt: an excerpt from extracted from the fulltext of the record
        :param keywords_item_id: the ID (order #) of the exceprt/keywords
        :param keywords: a list of keywords associated with the named entity extracted from the excerpt
        :param special_keywords: a list of keywords associated with the named entity extracted from the excerpt by NASA Concepts library
        :param knowledge_graph_score: the relevance score computed from the knowledge graph
        :param paper_relevance_score: the relevance score computed from the paper characteristics
        :param local_llm_score: the relevance score as reported by the local llm
        :param confidence_score: the overall confidence score
        :param named_entity_label: the label of the named entity as determined by the three scores
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