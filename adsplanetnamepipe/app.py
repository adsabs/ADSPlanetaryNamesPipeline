"""
The main application object (it has to be loaded by any worker/script)
in order to initialize the database and get a working configuration.
"""

from typing import List, Tuple
from datetime import datetime

from adsputils import ADSCelery

from adsplanetnamepipe.models import FeatureName, FeatureType, AmbiguousFeatureName, MultiTokenFeatureName, \
    NamedEntityLabel, Target, KnowledgeBase, KnowledgeBaseHistory, NamedEntityHistory, NamedEntity, USGSNomenclature


from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, desc, delete, asc
from sqlalchemy.sql.expression import func


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
                self.logger.error(f"Unable to fetch feature type entity records for {target_entity}.")
        return []

    def get_feature_ids(self) -> List[int]:
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
                # Create Target objects for each new target
                new_targets = [Target(entity=target) for target in target_list]
                session.bulk_save_objects(new_targets)
                session.commit()
                self.logger.debug("Added new target entities successfully.")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Error occurred while inserting new target entities: {str(e)}")
                return False

    def insert_feature_types(self, feature_type_list: List[dict]) -> bool:
        """
        insert new feature types for specific targets into the database

        :param feature_type_list: list of dictionaries, each containing `entity`, `target_entity`, and `plural_entity`
        :return: True if insertion is successful, False otherwise
        """
        with self.session_scope() as session:
            try:
                new_feature_types = [
                    FeatureType(
                        entity=feature_type['Feature_Type'],
                        target_entity=feature_type['Target'],
                        plural_entity=feature_type['Feature_Type_Plural'])
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

                # determine which entities are missing and prepare them for insertion
                new_feature_names = [
                    USGSNomenclature(entity=feature_name) for feature_name in feature_name_list if feature_name not in existing_entities
                ]
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
                new_feature_names = [
                    FeatureName(
                        entity=feature_name['Clean_Feature_Name'],
                        target_entity=feature_name['Target'],
                        feature_type_entity=feature_name['Feature_Type'],
                        entity_id=feature_name['Feature_ID'],
                        approval_year=feature_name['Approval_Date']
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

    def add_new_usgs_entities(self, data: List[dict]) -> bool:
        """
        updates the database with new target and feature type entries if they don't already exist

        :param data: list of dictionaries containing new feature_name entries from user input, targets and feature types might be new, might be not
        :return: True if insertion to both tables are successful, False otherwise
        """
        # extract unique targets and feature types from the new data
        new_targets = {entry['Target'] for entry in data}
        new_feature_types = {(entry['Feature_Type'], entry['Target'], entry['Feature_Type_Plural']) for entry in data}

        # update targets if there are any new entities
        existing_targets = set(self.get_target_entities())
        targets_to_add = list(new_targets - existing_targets)
        result_targets_added = self.insert_target_entities(targets_to_add) if targets_to_add else True

        # add new feature_names to usgs_nomenclature_table if not already there
        # note that usgs_nomenclature_table hold all unique feature_names while
        # feature_type_table holds feature_name associated with a target
        # hence it can identify ambiguous feature_names
        new_feature_names = {entry['Clean_Feature_Name'] for entry in data}
        result_usgs_nomenclature_added = self.insert_new_usgs_nomenclature_entities(new_feature_names) if new_feature_names else True

        if result_targets_added and result_usgs_nomenclature_added:
            # accumulate all feature types for each target
            unique_targets = {entry[1] for entry in new_feature_types}
            existing_feature_types_by_target = {target: set(self.get_feature_type_entities(target)) for target in unique_targets}

            # prepare feature types to add
            feature_types_to_add = []
            for feature_type, target_entity, plural_entity in new_feature_types:
                if feature_type not in existing_feature_types_by_target.get(target_entity, set()):
                    feature_types_to_add.append({
                        'Feature_Type': feature_type,
                        'Target': target_entity,
                        'Feature_Type_Plural': plural_entity
                    })
            result_feature_type_added = self.insert_feature_types(feature_types_to_add) if feature_types_to_add else True

            if result_feature_type_added:
                result_feature_names_added = self.insert_feature_name_records(data)

            return result_targets_added and result_usgs_nomenclature_added and result_feature_type_added and result_feature_names_added

        return False
