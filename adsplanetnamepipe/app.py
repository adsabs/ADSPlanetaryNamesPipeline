"""
The main application object (it has to be loaded by any worker/script)
in order to initialize the database and get a working configuration.
"""

from typing import List, Tuple
from datetime import datetime
import re

from adsputils import ADSCelery

from adsplanetnamepipe.models import FeatureName, FeatureType, AmbiguousFeatureName, MultiTokenFeatureName, \
    NamedEntityLabel, Target, KnowledgeBase, KnowledgeBaseHistory, NamedEntityHistory, NamedEntity, USGSNomenclature, \
    FeatureNameContext


from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_, desc, delete, asc
from sqlalchemy.sql import func, select


class ADSPlanetaryNamesPipelineCelery(ADSCelery):
    """
    main application object for the ADS Planetary Names Pipeline

    this class extends ADSCelery and provides methods for insert into/remove from/query the database tables
    """

    def __init__(self, app_name: str, *args, **kwargs):
        """
        initialize the ADSPlanetaryNamesPipelineCelery object

        :param app_name: the name of the application
        :param args: additional positional arguments
        :param kwargs: additional keyword arguments
        """
        ADSCelery.__init__(self, app_name, *args, **kwargs)

    def get_feature_name_entities(self, target_entity: str, feature_type_entity: str) -> List[str]:
        """
        return all the feature name entities for a given target and feature type

        :param target_entity: the target entity to filter by
        :param feature_type_entity: the feature type entity to filter by
        :return: a list of feature name entities
        """
        with self.session_scope() as session:
            rows = session.query(FeatureName).filter(and_(FeatureName.target_entity ==  target_entity,
                                                          FeatureName.feature_type_entity == feature_type_entity)) \
                                             .order_by(FeatureName.entity.asc()) \
                                             .all()
            if rows:
                feature_name_entities = []
                for row in rows:
                    feature_name_entities.append(row.entity)
                return feature_name_entities
            else:
                self.logger.error(f"No feature name entities are found for {target_entity}/{feature_type_entity}.")
        return []

    def get_feature_type_entity(self, target_entity: str, feature_name_entity: str) -> str:
        """
        return a feature type given target and feature name

        :param target_entity: the target entity to filter by
        :param feature_name_entity: the feature name entity to filter by
        :return: the feature type entity as a string
        """
        with self.session_scope() as session:
            rows = session.query(FeatureName).filter(and_(FeatureName.target_entity ==  target_entity,
                                                          FeatureName.entity == feature_name_entity)).all()
            if len(rows) == 1:
                return rows[0].feature_type_entity
            else:
                self.logger.error(f"No feature type entity is found for {target_entity}/{feature_name_entity}.")
        return ''

    def get_plural_feature_type_entity(self, feature_type_entity: str) -> str:
        """
        return the plural form of a feature type

        :param feature_type_entity: the feature type entity to get the plural form for
        :return: the plural form of the feature type entity as a string
        """
        with self.session_scope() as session:
            row = session.query(FeatureType).filter(FeatureType.entity == feature_type_entity).first()
            if row:
                return row.plural_entity
        return ''

    def get_context_ambiguous_feature_name(self, feature_name_entity: str) -> List[str]:
        """
        return the list of contexts associated with an ambiguous feature name

        :param feature_name_entity: the feature name entity to get contexts for
        :return: a list of contexts associated with the feature name, if none is found return []
        """
        with self.session_scope() as session:
            rows = session.query(AmbiguousFeatureName).filter(AmbiguousFeatureName.entity == feature_name_entity).all()
            if rows:
                context = []
                for row in rows:
                    context.append(row.context)
                return context
            else:
                self.logger.info(f"No context is found for entity {feature_name_entity}.")
        return []

    def get_multi_token_containing_feature_name(self, feature_name_entity: str) -> List[str]:
        """
        return the list of multi-token entities that contain a given feature name

        :param feature_name_entity: the feature name entity to search for
        :return: a list of multi-token entities containing the feature name, if none is found return []
        """
        with self.session_scope() as session:
            rows = session.query(MultiTokenFeatureName).filter(MultiTokenFeatureName.entity ==  feature_name_entity).all()
            if rows:
                multi_token_entity = []
                for row in rows:
                    multi_token_entity.append(row.multi_token_entity)
                return multi_token_entity
            else:
                self.logger.info(f"No multi token entity is found for entity {feature_name_entity}.")
        return []

    def get_named_entity_label(self) -> List[dict]:
        """
        return a list of dictionaries with named entity labels and their corresponding confidence values

        :return: a list of dictionaries containing named entity labels and confidence values
        """
        with self.session_scope() as session:
            rows = session.query(NamedEntityLabel).all()
            if rows:
                sorted_rows = sorted(rows, key=NamedEntityLabel.sort_key)
                named_entity_label = [row.toJSON() for row in sorted_rows]
                return named_entity_label
            else:
                self.logger.error("Unable to fetch the named entity label records.")
        return []

    def get_target_entities(self) -> List[str]:
        """
        return all the target entities

        :return: list of all target entities
        """
        with self.session_scope() as session:
            rows = session.query(Target).all()
            if rows:
                target_entities = []
                for row in rows:
                    target_entities.append(row.entity)
                return target_entities
            else:
                self.logger.error("Unable to fetch the target records.")
        return []

    def get_feature_type_entities(self, target_entity: str) -> List[str]:
        """
        return all feature types for a given target

        :param target_entity: The target entity to filter by.
        :return: List of all feature_type entities for the target.
        """
        with self.session_scope() as session:
            rows = session.query(FeatureType).filter(FeatureType.target_entity == target_entity).all()
            if rows:
                # Using list comprehension for simplicity
                feature_type_entities = [row.entity for row in rows]
                return feature_type_entities
            else:
                self.logger.error(f"Unable to fetch feature type entity for target {target_entity}.")
        return []

    def get_entity_ids(self) -> List[int]:
        """
        return all the entity IDs from the feature_name table

        :return: A list of entity IDs
        """
        with self.session_scope() as session:
            rows = session.query(FeatureName).order_by(FeatureName.entity_id.asc()).all()

            if rows:
                entity_ids = [row.entity_id for row in rows]
                return entity_ids
            else:
                self.logger.error(f"No entity IDs are found!")
        return []

    def insert_knowledge_base_records(self, knowledge_base_list: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]]) -> bool:
        """
        insert knowledge base records into the database

        :param knowledge_base_list: a list of tuples containing KnowledgeBaseHistory and associated KnowledgeBase records
        :return: true if the insertion is successful, false otherwise
        """
        with self.session_scope() as session:
            try:
                for history_entry, knowledge_base_entries in knowledge_base_list:
                    session.add(history_entry)
                    session.flush()  # flush to generate the ID for the history entry

                    for knowledge_base_entry in knowledge_base_entries:
                        knowledge_base_entry.history_id = history_entry.id  # assign the generated history ID to the KnowledgeBase entry

                    session.bulk_save_objects(knowledge_base_entries)
                session.commit()
                self.logger.debug("Added `KnowledgeBaseHistory` and `KnowledgeBase` records successfully.")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Error occurred while inserting `KnowledgeBaseHistory` and `KnowledgeBase` records: {str(e)}")
                return False

    def get_knowledge_base_keywords(self, feature_name_entity: str, feature_type_entity: str, target_entity: str,
                                          named_entity_label: str) -> List[str]:
        """
        retrieve knowledge base keywords for given parameters

        :param feature_name_entity: the feature name entity to filter by
        :param feature_type_entity: the feature type entity to filter by
        :param target_entity: the target entity to filter by
        :param named_entity_label: the named entity label to filter by
        :return: a list of keywords from the knowledge base
        """
        result = []
        with self.session_scope() as session:
            knowledge_base_history = session.query(KnowledgeBaseHistory.id.label('id')) \
                                        .filter(and_(KnowledgeBaseHistory.feature_name_entity == feature_name_entity,
                                                     KnowledgeBaseHistory.feature_type_entity == feature_type_entity,
                                                     KnowledgeBaseHistory.target_entity == target_entity,
                                                     KnowledgeBaseHistory.named_entity_label == named_entity_label)) \
                                        .order_by(desc(KnowledgeBaseHistory.date)) \
                                        .limit(1) \
                                        .subquery()

            rows = session.query(KnowledgeBase).filter(KnowledgeBase.history_id == knowledge_base_history.c.id).all()

            for row in rows:
                result.append(row.keywords)

            else:
                self.logger.error(f'Unable to fetch `KnowledgeBase` data for {feature_name_entity}/{feature_type_entity}/{target_entity}/{named_entity_label}.')

        return result

    def append_to_knowledge_base_keywords(self, feature_name_entity: str, target_entity: str, keyword: str) -> int:
        """
        append a keyword to the knowledge base keywords for a specific feature name and target

        :param feature_name_entity: the feature name entity to update
        :param target_entity: the target entity to update
        :param keyword: the keyword to append
        :return: the number of rows updated
        """
        with self.session_scope() as session:
            try:
                knowledge_base_history = session.query(KnowledgeBaseHistory.id.label('id')) \
                    .filter(and_(KnowledgeBaseHistory.feature_name_entity == feature_name_entity,
                                 KnowledgeBaseHistory.target_entity == target_entity,
                                 KnowledgeBaseHistory.named_entity_label == 'planetary')) \
                    .subquery()

                rows_updated = session.query(KnowledgeBase)\
                    .filter(and_(KnowledgeBase.history_id == knowledge_base_history.c.id,
                                 func.lower(KnowledgeBase.excerpt).like(f'%{keyword.lower()}%'),
                                ~KnowledgeBase.keywords.op('@>')('{' + keyword.lower() + '}'))
                ).update(
                    {KnowledgeBase.keywords: func.array_append(KnowledgeBase.keywords, keyword.lower())},
                    synchronize_session='fetch'
                )
                session.commit()

                self.logger.info(f"{rows_updated} rows updated with the keyword `{keyword.lower()}` for feature name `{feature_name_entity}` and target `{target_entity}` for the `KnowledgeBase` records.")
                return rows_updated

            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Error occurred while appending keyword `{keyword.lower()}` for feature name `{feature_name_entity}` and target `{target_entity}` for the `KnowledgeBase` records: {str(e)}")
                return -1

    def remove_from_knowledge_base_keywords(self, feature_name_entity: str, target_entity: str, keyword: str) -> int:
        """
        remove a keyword from the knowledge base keywords for a specific feature name and target

        :param feature_name_entity: the feature name entity to update
        :param target_entity: the target entity to update
        :param keyword: the keyword to remove
        :return: the number of rows updated
        """
        with self.session_scope() as session:
            try:
                knowledge_base_history = session.query(KnowledgeBaseHistory.id.label('id')) \
                    .filter(and_(KnowledgeBaseHistory.feature_name_entity == feature_name_entity,
                                 KnowledgeBaseHistory.target_entity == target_entity,
                                 KnowledgeBaseHistory.named_entity_label == 'planetary')) \
                    .subquery()

                rows_updated = session.query(KnowledgeBase) \
                    .filter(and_(KnowledgeBase.history_id == knowledge_base_history.c.id,
                                 KnowledgeBase.keywords.op('@>')('{' + keyword.lower() + '}'))) \
                    .update(
                        {KnowledgeBase.keywords: func.array_remove(KnowledgeBase.keywords, keyword.lower())},
                        synchronize_session='fetch'
                    )
                session.commit()

                self.logger.info(f"{rows_updated} rows updated by removing the keyword `{keyword.lower()}` for feature name `{feature_name_entity}` and target `{target_entity}` from the `KnowledgeBase` records.")
                return rows_updated

            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Error occurred while removing keyword `{keyword.lower()}` for feature name `{feature_name_entity}` and target `{target_entity}` from the `KnowledgeBase` records: {str(e)}")
                return -1

    def get_most_recent_knowledge_base_history_records(self, feature_name_entity: str, target_entity: str) -> Tuple[List[int], List[int]]:
        """
        retrieve the most recent knowledge base history record IDs for a specific feature name and target

        :param feature_name_entity: the feature name entity to filter by
        :param target_entity: the target entity to filter by
        :return: a tuple containing two lists of IDs: (planetary_ids, non_planetary_ids)
        """
        with self.session_scope() as session:
            rows = session.query(KnowledgeBaseHistory) \
                .filter(and_(KnowledgeBaseHistory.feature_name_entity == feature_name_entity,
                             KnowledgeBaseHistory.target_entity == target_entity,
                             KnowledgeBaseHistory.named_entity_label == 'planetary')) \
                .order_by(asc(KnowledgeBaseHistory.date)) \
                .all()
            knowledge_base_history_id_planetary = [row.id for row in rows]
            rows = session.query(KnowledgeBaseHistory) \
                .filter(and_(KnowledgeBaseHistory.feature_name_entity == feature_name_entity,
                             KnowledgeBaseHistory.target_entity == target_entity,
                             KnowledgeBaseHistory.named_entity_label == 'non planetary')) \
                .order_by(asc(KnowledgeBaseHistory.date)) \
                .all()
            knowledge_base_history_id_non_planetary = [row.id for row in rows]
            return knowledge_base_history_id_planetary, knowledge_base_history_id_non_planetary

    def remove_knowledge_base_records(self, ids_to_remove: List[int]) -> Tuple[int, int]:
        """
        remove knowledge base records with the given IDs

        :param ids_to_remove: a list of IDs to remove from the knowledge base
        :return: a tuple containing the number of rows deleted from (KnowledgeBase, KnowledgeBaseHistory)
        """
        with self.session_scope() as session:
            try:
                knowledge_base_rows_deleted = session.execute(delete(KnowledgeBase)
                                                              .where(KnowledgeBase.history_id.in_(ids_to_remove)) \
                                                              .execution_options(synchronize_session=False)).rowcount
                knowledge_base_history_rows_deleted = session.execute(delete(KnowledgeBaseHistory)
                                                                      .where(KnowledgeBaseHistory.id.in_(ids_to_remove)) \
                                                                      .execution_options(synchronize_session=False)).rowcount
                session.commit()
                return knowledge_base_rows_deleted, knowledge_base_history_rows_deleted
            except SQLAlchemyError as e:
                self.logger.error(f"Error occurred while deleting `KnowledgeBaseHistory` and `KnowledgeBase` records: {str(e)}")
                session.rollback()
                return -1, -1

    def remove_most_recent_knowledge_base_records(self, feature_name_entity: str, target_entity: str) -> Tuple[int, int]:
        """
        remove the most recent knowledge base records (knowledge_base_history and knowledge_base) for a specific feature name and target

        :param feature_name_entity: the feature name entity to filter by
        :param target_entity: the target entity to filter by
        :return: a tuple containing the number of rows deleted from (KnowledgeBaseHistory, KnowledgeBase)
        """
        planetary, non_planetary = self.get_most_recent_knowledge_base_history_records(feature_name_entity, target_entity)
        ids_to_remove = ([planetary[-1]] if planetary else []) + ([non_planetary[-1]] if non_planetary else [])
        knowledge_base_history_rows_deleted, knowledge_base_rows_deleted = self.remove_knowledge_base_records(ids_to_remove)
        if knowledge_base_history_rows_deleted > 0 and knowledge_base_rows_deleted > 0:
            self.logger.info(f"Removed the most recent records of knowledge graph for feature name `{feature_name_entity}` and target `{target_entity}`. "
                             f"Deleted {knowledge_base_history_rows_deleted} rows from knowledge_base and {knowledge_base_rows_deleted} rows from knowledge_base_history.")
        return knowledge_base_history_rows_deleted, knowledge_base_rows_deleted

    def remove_all_but_most_recent_knowledge_base_records(self, feature_name_entity: str, target_entity: str) -> Tuple[int, int]:
        """
        remove all but the most recent knowledge base records (knowledge_base_history and knowledge_base) for a specific feature name and target

        :param feature_name_entity: the feature name entity to filter by
        :param target_entity: the target entity to filter by
        :return: a tuple containing the number of rows deleted from (KnowledgeBaseHistory, KnowledgeBase)
        """
        planetary, non_planetary = self.get_most_recent_knowledge_base_history_records(feature_name_entity, target_entity)
        ids_to_remove = planetary[:-1] + non_planetary[:-1]
        knowledge_base_history_rows_deleted, knowledge_base_rows_deleted = self.remove_knowledge_base_records(ids_to_remove)
        if knowledge_base_history_rows_deleted > 0 and knowledge_base_rows_deleted > 0:
            self.logger.info(f"Removed all but the most recent records of knowledge graph for feature name `{feature_name_entity}` and target `{target_entity}`. "
                             f"Deleted {knowledge_base_history_rows_deleted} rows from knowledge_base_history and {knowledge_base_rows_deleted} rows from knowledge_base.")
        return knowledge_base_history_rows_deleted, knowledge_base_rows_deleted

    def insert_named_entity_records(self, named_entity_list: List[Tuple[NamedEntityHistory, List[NamedEntity]]]) -> bool:
        """
        insert named entity records into the database

        :param named_entity_list: a list of tuples containing NamedEntityHistory and associated NamedEntity records
        :return: true if the insertion is successful, false otherwise
        """
        with self.session_scope() as session:
            try:
                for history_entry, named_entity_entries in named_entity_list:
                    session.add(history_entry)
                    session.flush()  # flush to generate the ID for the history entry

                    for named_entity_entry in named_entity_entries:
                        named_entity_entry.history_id = history_entry.id  # assign the generated history ID to the KnowledgeBase entry

                    session.bulk_save_objects(named_entity_entries)

                session.commit()
                self.logger.debug("Added `NamedEntityHistory` and `NamedEntity` records successfully.")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Error occurred while inserting `NamedEntityHistory` and `NamedEntity` records: {str(e)}")
                return False

    def get_named_entity_bibcodes(self, feature_name_entity: str = None, feature_type_entity: str = None,
                                        target_entity: str = None, confidence_score: float = None, date: datetime = None) -> \
                                        List[Tuple[str, str, str, str, float, str]]:
        """
        retrieve named entity bibcodes and related information based on the given parameters

        :param feature_name_entity: the feature name entity to filter by (optional)
        :param feature_type_entity: the feature type entity to filter by (optional)
        :param target_entity: the target entity to filter by (optional)
        :param confidence_score: the minimum confidence score to filter by (optional)
        :param date: the minimum date to filter by (optional)
        :return: a list of tuples containing (bibcode, target_entity, feature_type_entity, feature_name_entity, confidence_score, date)
        """
        result = []
        with self.session_scope() as session:
            # check if each filter value is present and add the corresponding condition to the list
            conditions = []
            if feature_name_entity:
                conditions.append(NamedEntityHistory.feature_name_entity == feature_name_entity)
            if feature_type_entity:
                conditions.append(NamedEntityHistory.feature_type_entity == feature_type_entity)
            if target_entity:
                conditions.append(NamedEntityHistory.target_entity == target_entity)
            if date:
                conditions.append(NamedEntityHistory.date >= date)
            if confidence_score:
                conditions.append(NamedEntity.confidence_score >= confidence_score)

            rows = session.query(NamedEntity.bibcode,
                                 NamedEntityHistory.target_entity,
                                 NamedEntityHistory.feature_type_entity,
                                 NamedEntityHistory.feature_name_entity,
                                 NamedEntity.confidence_score,
                                 NamedEntityHistory.date) \
                .join(NamedEntity, NamedEntityHistory.id == NamedEntity.history_id) \
                .filter(and_(*conditions)) \
                .order_by(NamedEntity.bibcode.asc(), NamedEntityHistory.feature_name_entity.asc(), NamedEntityHistory.date.asc()) \
                .distinct(NamedEntity.bibcode,
                          NamedEntityHistory.target_entity,
                          NamedEntityHistory.feature_type_entity,
                          NamedEntityHistory.feature_name_entity,
                          NamedEntity.confidence_score,
                          NamedEntityHistory.date) \
                .all()

            if len(rows) > 0:
                for row in rows:
                    result.append((row.bibcode,
                                   row.target_entity,
                                   row.feature_type_entity,
                                   row.feature_name_entity,
                                   row.confidence_score,
                                   row.date.strftime("%Y-%m-%d %H:%M:%S")))
            else:
                self.logger.error(f'Unable to fetch `NamedEntity` data for {feature_name_entity}/{feature_type_entity}/{target_entity}/{confidence_score}.')

        return result

    def insert_target_entities(self, target_list: List[str]) -> bool:
        """
        insert new target entities into the database.

        :param target_list: list of new target entities to insert
        :return: True if insertion is successful, False otherwise
        """
        with self.session_scope() as session:
            try:
                # create Target objects for each new entity
                new_targets = [Target(entity=target) for target in target_list]
                session.bulk_save_objects(new_targets)
                session.commit()
                self.logger.debug("Added new target entities successfully.")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Error occurred while inserting new target entities: {str(e)}")
                return False

    def insert_feature_type_records(self, feature_type_list: List[dict]) -> bool:
        """
        insert new feature types for specific targets into the database

        :param feature_type_list: list of dictionaries, each containing `entity`, `target_entity`, and `plural_entity`
        :return: True if insertion is successful, False otherwise
        """
        with self.session_scope() as session:
            try:
                # create FeatureType objects for each new entity
                new_feature_types = [
                    FeatureType(
                        entity=feature_type['feature_type'],
                        target_entity=feature_type['target'],
                        plural_entity=feature_type['feature_type_plural'])
                    for feature_type in feature_type_list
                ]
                session.bulk_save_objects(new_feature_types)
                session.commit()
                self.logger.debug("Added new feature types successfully.")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Error occurred while inserting new feature types: {str(e)}")
                return False

    def insert_new_usgs_nomenclature_entities(self, feature_name_list: List[str]) -> bool:
        """
        checks for the presence of specified entities in the `usgs_nomenclature` table
        only add any new feature_names

        :param feature_name_list: list of future names to check and add if missing
        :return: True if insertion is successful or if all entities already exist, False otherwise
        """
        with self.session_scope() as session:
            try:
                # query only for entities that need to be checked
                existing_entities = {row.entity for row in session.query(USGSNomenclature.entity)
                    .filter(USGSNomenclature.entity.in_(feature_name_list)).all()}

                # determine which entities are missing, then create USGSNomenclature objects for each new entity to be inserted
                new_feature_names = [USGSNomenclature(entity=feature_name) for feature_name in feature_name_list if feature_name not in existing_entities]
                if new_feature_names:
                    session.bulk_save_objects(new_feature_names)
                    session.commit()
                    self.logger.debug(f"Inserted {len(new_feature_names)} new feature names into `usgs_nomenclature`.")
                else:
                    self.logger.debug("No new feature name entities needed to be added to `usgs_nomenclature`.")
                return True
            except Exception as e:
                session.rollback()
                self.logger.error(f"Error inserting usgs_nomenclature entities: {str(e)}")
                return False

    def insert_feature_name_records(self, feature_name_list: List[dict]) -> bool:
        """
        Inserts feature name records into the `feature_name` table.

        :param feature_name_list: list of dictionaries, each representing a feature name entry
        :return: True if insertion is successful, False otherwise
        """
        with self.session_scope() as session:
            try:
                # create FeatureName objects for each new entity
                new_feature_names = [
                    FeatureName(
                        entity=feature_name['feature_name'],
                        target_entity=feature_name['target'],
                        feature_type_entity=feature_name['feature_type'],
                        entity_id=feature_name['entity_id'],
                        approval_year=feature_name['approval_date']
                    )
                    for feature_name in feature_name_list
                ]
                session.bulk_save_objects(new_feature_names)
                session.commit()
                self.logger.debug("Feature name records inserted successfully.")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Error inserting feature name records: {str(e)}")
                return False

    def get_usgs_entities(self) -> List[str]:
        """
        return all the unique feature name entities

        :return: list of all unique feature names entities
        """
        with self.session_scope() as session:
            rows = session.query(USGSNomenclature).all()
            if rows:
                usgs_entities = []
                for row in rows:
                    usgs_entities.append(row.entity)
                return usgs_entities
            else:
                self.logger.error("Unable to fetch the USGSNomenclature records.")
        return []

    def get_new_ambiguous_records(self, feature_name_list: List[str]) -> List[Tuple[str, str]]:
        """
        queries the FeatureName table to extract matching USGS entities and their corresponding targets,
        while excluding entities already present in the AmbiguousFeatureName table using a single query.

        :param matching_usgs_entities: Set of matching USGS entities to be queried.
        :return: list of feature names and the associated target that because of new additions will become ambiguous
        """
        with self.session_scope() as session:
            # # get matching feature names and targets excluding those in AmbiguousFeatureName already
            # rows = session.query(FeatureName.entity, FeatureName.target_entity) \
            #     .filter(FeatureName.entity.in_(feature_name_list)) \
            #     .filter(~session.query(AmbiguousFeatureName) \
            #             .filter(and_(AmbiguousFeatureName.entity == FeatureName.entity,
            #                          AmbiguousFeatureName.context == FeatureName.target_entity)) \
            #             .exists()) \
            #     .group_by(FeatureName.entity) \
            #     .having(func.count(FeatureName.target_entity) > 1) \
            #     .all()


            # query for entities that appear more than once in FeatureName table
            subquery = session.query(FeatureName.entity) \
                .filter(FeatureName.entity.in_(feature_name_list)) \
                .group_by(FeatureName.entity) \
                .having(func.count(FeatureName.target_entity) > 1) \
                .subquery()
            # query to filter out records that already exist in AmbiguousFeatureName
            rows = session.query(FeatureName.entity, FeatureName.target_entity) \
                .filter(FeatureName.entity.in_(select(subquery))) \
                .filter(~session.query(AmbiguousFeatureName) \
                        .filter(and_(AmbiguousFeatureName.entity == FeatureName.entity, AmbiguousFeatureName.context == FeatureName.target_entity)) \
                        .exists()) \
                .all()
            if rows:
                new_ambiguous_records = []
                for row in rows:
                    new_ambiguous_records.append((row.entity, row.target_entity))
                return new_ambiguous_records
            else:
                self.logger.error("No new ambiguous records found for the provided feature names.")
        return []

    def get_context_entities(self) -> List[str]:
        """
        return all the context entities

        :return: list of all context entities
        """
        with self.session_scope() as session:
            rows = session.query(FeatureNameContext).all()
            if rows:
                context_entities = []
                for row in rows:
                    context_entities.append(row.context)
                return context_entities
            else:
                self.logger.error("Unable to fetch the FeatureNameContext records.")
        return []

    def insert_feature_name_contexts(self, context_list: List[str]) -> bool:
        """
        inserts new contexts into the FeatureNameContext table if they are not already present

        :param context_list: list of contexts to be added
        :return: True if insertion is successful or no new contexts need to be added, False otherwise
        """
        # fetch existing contexts from the FeatureNameContext table
        existing_contexts = self.get_context_entities()

        # determine which entities are missing, then create FeatureNameContext objects for each new entity to be inserted
        new_contexts = [FeatureNameContext(context=context) for context in context_list if context not in existing_contexts]

        if not new_contexts:
            self.logger.debug("No new contexts need to be added.")
            return True

        # insert all collected entries
        with self.session_scope() as session:
            try:
                session.bulk_save_objects(new_contexts)
                session.commit()
                self.logger.debug(f"Inserted {len(new_contexts)} new contexts into `FeatureNameContext`.")
                return True
            except Exception as e:
                session.rollback()
                self.logger.error(f"Error inserting into `FeatureNameContext` table: {str(e)}")
                return False

    def insert_ambiguous_feature_names(self, feature_name_target_list: List[Tuple[str, str]]) -> bool:
        """
        updates the AmbiguousFeatureName table for any new associations where feature names may refer
        to multiple celestial bodies (contexts)

        :param feature_name_target_list: list of tuples where each tuple contains a feature name and its associated target
        :return: True if insertion is successful or if no new ambiguity is detected, False otherwise.
        """
        # fetch unique USGS entities from the database and see if any are in the feature_name_target_list
        usgs_entities = self.get_usgs_entities()
        input_feature_names = [feature_name for feature_name, _ in feature_name_target_list]
        matching_usgs_entities = list(set(input_feature_names).intersection(usgs_entities))

        # list of records exists in the feature name table that because of the update will become ambiguous
        new_ambiguous_feature_names = [
            AmbiguousFeatureName(entity=feature_name, context=target)
            for feature_name, target in self.get_new_ambiguous_records(matching_usgs_entities)
        ]

        if not new_ambiguous_feature_names:
            self.logger.debug("No new ambiguous feature names need to be added.")
            return True

        # get the list of targets and verify that they exist in the FeatureNameContext table
        target_contexts = {record.context for record in new_ambiguous_feature_names}
        if not self.insert_feature_name_contexts(list(target_contexts)):
            self.logger.error("Failed to insert new contexts into `FeatureNameContext` table, preventing updates to ambiguous feature names.")
            return False

        # insert all collected entries
        with self.session_scope() as session:
            try:
                session.bulk_save_objects(new_ambiguous_feature_names)
                session.commit()
                self.logger.debug(f"Inserted {len(new_ambiguous_feature_names)} new ambiguous feature names.")
                return True
            except Exception as e:
                session.rollback()
                self.logger.error(f"Error inserting into `ambiguous_feature_names` table: {str(e)}")
                return False

    def get_matched_usgs_entities(self, token_list: List[str]) -> List[str]:
        """
        query the USGSNomenclature table for a list of tokens and return the matching entities

        :param token_list: list of tokens to search for in the USGSNomenclature table
        :return: list of matching entities
        """
        with self.session_scope() as session:
            part_of_rows = session.query(USGSNomenclature.entity).filter(USGSNomenclature.entity.in_(token_list)).all()
            like_rows = session.query(USGSNomenclature.entity).filter(or_(*[USGSNomenclature.entity.like(f"%{token}%") for token in token_list])).all()
            rows = part_of_rows + like_rows
            if rows:
                return list(set(row.entity for row in rows))
            else:
                self.logger.info("No matching USGS entities found for the provided tokens.")
        return []

    def insert_multi_token_feature_names(self, feature_name_list: List[str]) -> bool:
            """
            updates the multi_token_feature_names table if need to for the new feature_name_list entities
            it first splits the input feature names into tokens,
            querying the database for these tokens, and then compars the matches

            :param feature_name_list: List of input feature names to be checked for single/multi-token associations.
            :return: True if insertion is successful or if none of the entities had ambiguity, False otherwise
            """
            # split all feature names into individual tokens
            all_tokens = {token for feature_name in feature_name_list for token in feature_name.split()}

            # query usgs table for any matches that contains these tokens
            matched_entities = self.get_matched_usgs_entities(all_tokens)

           # collect associations to be added based on matches
            new_multi_token_entities = []
            for i, feature_name in enumerate(feature_name_list):
                for matched_entity in matched_entities:
                    # Check if matched_entities contain any tokens from feature_name_list and vice versa
                    if re.search(rf'\b{feature_name}\b', matched_entity) and feature_name != matched_entity:
                        new_multi_token_entities.append(MultiTokenFeatureName(entity=feature_name, multi_token_entity=matched_entity))
                    elif re.search(rf'\b{matched_entity}\b', feature_name) and feature_name != matched_entity:
                        new_multi_token_entities.append(MultiTokenFeatureName(entity=matched_entity, multi_token_entity=feature_name))

                for other_feature_name in feature_name_list[(i+1):]:
                    # check for associations within the feature_name_list
                    if re.search(rf'\b{feature_name}\b', other_feature_name):
                        new_multi_token_entities.append(MultiTokenFeatureName(entity=feature_name, multi_token_entity=other_feature_name))
                    elif re.search(rf'\b{other_feature_name}\b', feature_name):
                        new_multi_token_entities.append(MultiTokenFeatureName(entity=other_feature_name, multi_token_entity=feature_name))

            if not new_multi_token_entities:
                self.logger.debug("No new associations needed in `multi_token_feature_names`.")
                return True

            # insert all collected entries
            with self.session_scope() as session:
                try:
                    session.bulk_save_objects(new_multi_token_entities)
                    session.commit()
                    self.logger.debug(f"Inserted {len(new_multi_token_entities)} new associations into `multi_token_feature_names`.")
                    return True
                except Exception as e:
                    session.rollback()
                    self.logger.error(f"Error inserting into `multi_token_feature_names` table: {str(e)}")
                    return False

    def add_new_usgs_entities(self, data: List[dict]) -> bool:
        """
        updates the database with new target and feature type entries if they don't already exist

        :param data: list of dictionaries containing new feature_name entries from user input, targets and feature types might be new, might be not
        :return: True if insertion to both tables are successful, False otherwise
        """
        # extract unique targets and feature types from the new data
        new_targets = {entry['target'] for entry in data}
        new_feature_types = [(entry['feature_type'], entry['target'], entry['feature_type_plural']) for entry in data]

        # update targets if there are any new entities
        existing_targets = self.get_target_entities()
        targets_to_add = list(new_targets - set(existing_targets))
        result_targets_added = self.insert_target_entities(targets_to_add) if targets_to_add else True

        if result_targets_added:
            # determine which feature types are new and need to be added
            existing_feature_types = {target: set(self.get_feature_type_entities(target)) for target in new_targets}
            feature_types_to_add = []
            # keep track of unique feature type entries
            seen_feature_types = set()
            for feature_type, target_entity, plural_entity in new_feature_types:
                # create a tuple representing the unique combination of feature type, target, and plural entity
                feature_type_tuple = (feature_type, target_entity, plural_entity)

                if feature_type not in existing_feature_types.get(target_entity, set()) and feature_type_tuple not in seen_feature_types:
                    feature_types_to_add.append({
                        'feature_type': feature_type,
                        'target': target_entity,
                        'feature_type_plural': plural_entity
                    })
                    seen_feature_types.add(feature_type_tuple)
            result_feature_type_added = self.insert_feature_type_records(feature_types_to_add) if feature_types_to_add else True

            if result_feature_type_added:
                # add new feature_names to usgs_nomenclature_table if not already there
                # note that usgs_nomenclature_table hold all unique feature_names while
                # feature_type_table holds feature_name associated with a target
                # hence it can identify ambiguous feature_names
                new_feature_names = list(set(entry['feature_name'] for entry in data))
                result_usgs_nomenclature_added = self.insert_new_usgs_nomenclature_entities(new_feature_names) if new_feature_names else True


                if result_usgs_nomenclature_added:
                    if self.insert_feature_name_records(data):
                        new_feature_name_and_target = [(entry['feature_name'], entry['target']) for entry in data]
                        if self.insert_ambiguous_feature_names(new_feature_name_and_target):
                            return self.insert_multi_token_feature_names(new_feature_names)

        return False
