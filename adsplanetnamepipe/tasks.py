from adsplanetnamepipe import app as app_module
from kombu import Queue

import os

from adsplanetnamepipe.utils.common import PLANETARYNAMES_PIPELINE_ACTION, EntityArgs
from adsplanetnamepipe.collect import CollectKnowldegeBase
from adsplanetnamepipe.identify import IdentifyPlanetaryEntities

from adsputils import load_config

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
config = load_config(proj_home=proj_home)
app = app_module.ADSPlanetaryNamesPipelineCelery('planetary-names-pipeline',
                                                 proj_home=proj_home,
                                                 local_config=globals().get('local_config', {}))

app.conf.CELERY_QUEUES = (
    Queue('task_process_planetary_nomenclature', app.exchange, routing_key='task_process_planetary_nomenclature'),
)

logger = app.logger


class FailedRequest(Exception):
    """
    Failed to connect to reference service.
    """
    pass


@app.task(queue='task_process_planetary_nomenclature', max_retries=config['MAX_QUEUE_RETRIES'])
def task_process_planetary_nomenclature(the_task: dict) -> bool:
    """
    processes planetary nomenclature tasks based on the provided action type

    handles the two main actions:
        1. collecting data for knowledge base setup
        2. identifying and labeling entities

    :param the_task: PlanetaryNomenclatureTask, A typed dictionary containing:
                     - 'action_type': PLANETARYNAMES_PIPELINE_ACTION enum value
                     - 'args': EntityArgs object containing task arguments
    :return: bool, returns True if the task is processed successfully, False otherwise
    """
    try:
        # deserialize
        action_type = PLANETARYNAMES_PIPELINE_ACTION(the_task['action_type'])
        entity_args = EntityArgs(**the_task["args"])

        # either: action to collect data to setup KB graph
        if action_type in [PLANETARYNAMES_PIPELINE_ACTION.collect, PLANETARYNAMES_PIPELINE_ACTION.end_to_end]:
            knowledge_base_records = CollectKnowldegeBase(entity_args).collect()
            if knowledge_base_records:
                return bool(app.insert_knowledge_base_records(knowledge_base_records))

            logger.info(f"No knowledge base records found for: {entity_args.feature_name}/{entity_args.feature_type}/{entity_args.target}")
            return False

        # or: action to identify and label entities
        if action_type in [PLANETARYNAMES_PIPELINE_ACTION.identify, PLANETARYNAMES_PIPELINE_ACTION.end_to_end]:
            keywords_positive = app.get_knowledge_base_keywords(entity_args.feature_name,
                                                                entity_args.feature_type,
                                                                entity_args.target,
                                                                entity_args.name_entity_labels[0]['label'])
            keywords_negative = app.get_knowledge_base_keywords(entity_args.feature_name,
                                                                entity_args.feature_type,
                                                                entity_args.target,
                                                                entity_args.name_entity_labels[1]['label'])
            named_entity_records = IdentifyPlanetaryEntities(entity_args, keywords_positive,
                                                             keywords_negative).identify()
            if named_entity_records:
                return bool(app.insert_named_entity_records(named_entity_records))

            logger.info(f"No records identified for: {entity_args.feature_name}/{entity_args.feature_type}/{entity_args.target}")
            return False

        logger.error(f"Unhandled action: {action_type}")
        return False

    except KeyError as e:
        logger.error(f"KeyError in task_process_planetary_nomenclature: {str(e)}")
        return False


# dont know how to unittest this part
# this (app.start()) the only line that is not unittested
# and since i want all modules to be 100% covered,
# making this line not be considered part of coverage
if __name__ == '__main__':    # pragma: no cover
    app.start()