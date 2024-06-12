"""
The main application object (it has to be loaded by any worker/script)
in order to initialize the database and get a working configuration.
"""

from typing import List, Tuple

from adsputils import ADSCelery

from adsplanetnamepipe.models import FeatureName, FeatureType, AmbiguousFeatureName, MultiTokenFeatureName, \
    NamedEntityLabel, Target, KnowledgeBase, KnowledgeBaseHistory, NamedEntityHistory, NamedEntity


from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, desc, delete, asc
from sqlalchemy.sql.expression import func


class ADSPlanetaryNamesPipelineCelery(ADSCelery):

    def __init__(self, app_name, *args, **kwargs):
        """

        :param app_name:
        :param args:
        :param kwargs:
        """
        ADSCelery.__init__(self, app_name, *args, **kwargs)

    def get_feature_name_entities(self, target_entity, feature_type_entity):
        """
        return all the feature names for target and feature type

        :param target_entity:
        :param feature_type_entity:
        :return:
        """
        with self.session_scope() as session:
            rows = session.query(FeatureName).filter(and_(FeatureName.target_entity ==  target_entity,
                                                          FeatureName.feature_type_entity == feature_type_entity)) \
                                             .order_by(FeatureName.entity.asc()) \
                                             .all()
            if len(rows) > 1:
                feature_name_entities = []
                for row in rows:
                    feature_name_entities.append(row.entity)
                return feature_name_entities
            else:
                self.logger.error(f"No feature name entities are found for {target_entity}/{feature_type_entity}.")
        return []

    def get_feature_type_entity(self, target_entity, feature_name_entity):
        """
        return a feature type given target and feature name

        :param feature_name_entity:
        :param target_entity:
        :return:
        """
        with self.session_scope() as session:
            rows = session.query(FeatureName).filter(and_(FeatureName.target_entity ==  target_entity,
                                                          FeatureName.entity == feature_name_entity)).all()
            if len(rows) == 1:
                return rows[0].feature_type_entity
            else:
                self.logger.error(f"No feature type entity is found for {target_entity}/{feature_name_entity}.")
        return ''

    def get_plural_feature_type_entity(self, feature_type_entity):
        """
        return the plural form of feature type

        :param feature_type_entity:
        :return:
        """
        with self.session_scope() as session:
            row = session.query(FeatureType).filter(FeatureType.entity == feature_type_entity).first()
            if row:
                return row.plural_entity
        return ''

    def get_context_ambiguous_feature_name(self, feature_name_entity):
        """
        return the list of context associated with a feature name
        if none return []

        :param feature_name_entity:
        :return:
        """
        with self.session_scope() as session:
            rows = session.query(AmbiguousFeatureName).filter(AmbiguousFeatureName.entity == feature_name_entity).all()
            if len(rows) > 1:
                context = []
                for row in rows:
                    context.append(row.context)
                return context
            else:
                self.logger.info(f"No context is found for entity {feature_name_entity}.")
        return []

    def get_multi_token_containing_feature_name(self, feature_name_entity):
        """
        return the list of multi token that contains a feature name
        if none return []

        :param feature_name_entity:
        :return:
        """
        with self.session_scope() as session:
            rows = session.query(MultiTokenFeatureName).filter(MultiTokenFeatureName.entity ==  feature_name_entity).all()
            if len(rows) > 1:
                multi_token_entity = []
                for row in rows:
                    multi_token_entity.append(row.multi_token_entity)
                return multi_token_entity
            else:
                self.logger.info(f"No multi token entity is found for entity {feature_name_entity}.")
        return []

    def get_named_entity_label(self):
        """
        return list of dict with named entity label and value corresponding to confidence [0, 1]

        :return:
        """
        with self.session_scope() as session:
            rows = session.query(NamedEntityLabel).all()
            if len(rows) > 1:
                named_entity_label = []
                for row in rows:
                    named_entity_label.append(row.toJSON())
                return named_entity_label
            else:
                self.logger.error("Unable to fetch the named entity label records.")
        return []

    def get_target_entities(self):
        """
        return all the target entities

        :return:
        """
        with self.session_scope() as session:
            rows = session.query(Target).all()
            if len(rows) > 1:
                target_entities = []
                for row in rows:
                    target_entities.append(row.entity)
                return target_entities
            else:
                self.logger.error("Unable to fetch the target records.")
        return []

    def insert_knowledge_base_records(self, knowledge_base_list: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]]):
        """

        :param knowledge_base_list: List of tuples containing KnowledgeBaseHistory and associated KnowledgeBase records
        :return: True if the insertion is successful, False otherwise
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

    def get_knowledge_base_keywords(self, feature_name_entity, feature_type_entity, target_entity, named_entity_label):
        """

        :param feature_name_entity:
        :param feature_type_entity:
        :param target_entity:
        :param named_entity_label:
        :return:
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

    def append_to_knowledge_base_keywords(self, feature_name_entity, target_entity, keyword):
        """
        search for the keyword in the excerpts of specific feature name/target
        add it to the keywords column, if not already there
        note that this is only implemented for the `planetary` records

        :param feature_name_entity:
        :param target_entity:
        :param keyword:
        :return: number of rows updated
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

    def remove_from_knowledge_base_keywords(self, feature_name_entity, target_entity, keyword):
        """
        remove the keyword from the keywords for a specific feature name/target
        note that this is only implemented for the `planetary` records

        :param feature_name_entity:
        :param target_entity:
        :param keyword:
        :return: number of rows updated
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

    def get_most_recent_knowledge_base_history_records(self, feature_name_entity, target_entity):
        """

        :param feature_name_entity:
        :param target_entity:
        :return:
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

    def remove_knowledge_base_records(self, ids_to_remove):
        """

        :param ids_to_remove:
        :return:
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

    def remove_most_recent_knowledge_base_records(self, feature_name_entity, target_entity):
        """
        remove the last record from the knowledge_base_history table for a specific feature name/target
        also remove the corresponding records from the knowledge_base table

        :param feature_name_entity:
        :param target_entity:
        :return: number of rows deleted from knowledge_base_history and knowledge_base tables
        """
        planetary, non_planetary = self.get_most_recent_knowledge_base_history_records(feature_name_entity, target_entity)
        ids_to_remove = ([planetary[-1]] if planetary else []) + ([non_planetary[-1]] if non_planetary else [])
        knowledge_base_history_rows_deleted, knowledge_base_rows_deleted = self.remove_knowledge_base_records(ids_to_remove)
        if knowledge_base_history_rows_deleted > 0 and knowledge_base_rows_deleted > 0:
            self.logger.info(f"Removed the most recent records of knowledge graph for feature name `{feature_name_entity}` and target `{target_entity}`. "
                             f"Deleted {knowledge_base_history_rows_deleted} rows from knowledge_base and {knowledge_base_rows_deleted} rows from knowledge_base_history.")
        return knowledge_base_history_rows_deleted, knowledge_base_rows_deleted

    def remove_all_but_most_recent_knowledge_base_records(self, feature_name_entity, target_entity):
        """
        remove all but the last record from the knowledge_base_history table for a specific feature name/target
        also remove the corresponding records from the knowledge_base table

        :param feature_name_entity:
        :param target_entity:
        :return: number of rows deleted from knowledge_base_history and knowledge_base tables
        """
        planetary, non_planetary = self.get_most_recent_knowledge_base_history_records(feature_name_entity, target_entity)
        ids_to_remove = planetary[:-1] + non_planetary[:-1]
        knowledge_base_history_rows_deleted, knowledge_base_rows_deleted = self.remove_knowledge_base_records(ids_to_remove)
        if knowledge_base_history_rows_deleted > 0 and knowledge_base_rows_deleted > 0:
            self.logger.info(f"Removed all but the most recent records of knowledge graph for feature name `{feature_name_entity}` and target `{target_entity}`. "
                             f"Deleted {knowledge_base_history_rows_deleted} rows from knowledge_base_history and {knowledge_base_rows_deleted} rows from knowledge_base.")
        return knowledge_base_history_rows_deleted, knowledge_base_rows_deleted

    def insert_named_entity_records(self, named_entity_list: List[Tuple[NamedEntityHistory, List[NamedEntity]]]):
        """

        :param knowledge_base_list: List of tuples containing NamedEntityHistory and associated NamedEntity records
        :return: True if the insertion is successful, False otherwise
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

    def get_named_entity_bibcodes(self, feature_name_entity=None, feature_type_entity=None, target_entity=None, confidence_score=None, date=None):
        """

        :param feature_name_entity:
        :param feature_type_entity:
        :param target_entity:
        :param confidence_score:
        :param date:
        :return:
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
