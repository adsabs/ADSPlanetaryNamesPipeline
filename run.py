import sys
import os
import csv
from typing import List, Tuple, Dict
from datetime import datetime, timedelta

from adsputils import setup_logging, load_config

import argparse

from adsplanetnamepipe import tasks
from adsplanetnamepipe.utils.common import PLANETARYNAMES_PIPELINE_ACTION, EntityArgs

proj_home = os.path.realpath(os.path.dirname(__file__))
config = load_config(proj_home=proj_home)

app = tasks.app
logger = setup_logging('run.py')


def map_input_param_to_action_type(input_param: str) -> PLANETARYNAMES_PIPELINE_ACTION:
    """
    maps an input parameter to a corresponding action type from the PLANETARYNAMES_PIPELINE_ACTION enum

    :param input_param: input parameter to be mapped to an action type
    :return: PLANETARYNAMES_PIPELINE_ACTION, corresponding PLANETARYNAMES_PIPELINE_ACTION enum value
    """
    try:
        return PLANETARYNAMES_PIPELINE_ACTION[input_param]
    except (KeyError, ValueError):
        return PLANETARYNAMES_PIPELINE_ACTION.invalid


def verify_args(args: argparse) -> Tuple[str, str, List[str]]:
    """
    verifies and processes the command-line arguments, returning the target, feature type, and feature names

    :param args: parsed command-line arguments object
    :return: Tuple[str, str, List[str]], A tuple containing (target, feature_type, feature_names)
    """
    if args.target and args.feature_type:
        return args.target, args.feature_type, app.get_feature_name_entities(args.target, args.feature_type)
    if args.target and args.feature_name:
        return args.target, app.get_feature_type_entity(args.target, args.feature_name), [args.feature_name]
    return args.target, args.feature_type, [args.feature_name]


def get_date(days: int) -> datetime:
    """
    calculates a target date based on the number of days provided
    if days is 0, it returns the earliest possible datetime (datetime.min).

    :param days: int, the number of days to subtract from the current date. Can be 0 or a positive integer.
    :return: datetime, a datetime object representing either the target date or the earliest possible datetime if days is 0.
    """
    if days == 0:
        return datetime.min

    target_date = datetime.now().date() - timedelta(days=days)
    target_datetime = datetime.combine(target_date, datetime.now().time())
    return target_datetime


def read_updated_usgs_gazetteer(csv_file_path: str) -> List[Dict[str, str]]:
    """
    reads the updated usgs entities
    containing five columns: Feature_ID,Clean_Feature_Name,Target,Feature_Type,Approval_Date,Approval_Status

    :param csv_file_path: path to the CSV file.
    :return: list of dictionaries, each containing data from one row.
    """
    data = []
    try:
        with open(csv_file_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Approval_Status'].lower() == 'approved':
                    # extract year from `Approval_Date` and use only the first part of `Feature_Type` before comma
                    row['Approval_Date'] = row['Approval_Date'].split('-')[-1] if '-' in row['Approval_Date'] else row['Approval_Date']
                    row['Feature_Type'], row['Feature_Type_Plural'] = [x.strip() for x in (row['Feature_Type'].split(',') + [''])[:2]]
                    data.append(row)
    except FileNotFoundError:
        logger.error(f"File not found: {csv_file_path}. Please check the path and try again.")
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {e}")
    return data

# python run.py -t Mars -f "Albedo Feature" -a collect
# python run.py -t Mars -n Amenthes -a collect
# python run.py -t Mars -n Amenthes -a remove_all_but_last
# python run.py -t Mars -n Amenthes -a remove_the_most_recent
# python run.py -t Mars -n Amenthes -a add_keyword_to_knowledge_graph -k basin -t Moon -f Apollo
# python run.py -t Mars -n Amenthes -a remove_keyword_from_knowledge_graph
# python run.py -t Mars -n Amenthes -a retrieve_identified_entities -c 0.75
# TODO: add a command to run all the feature types for all celestial bodies for collect step
# TODO: add a command to run all the feature names since d days ago
# python run.py -a update_database_with_usgs_entities -u updated_usgs_terms.csv

# Main entry point of the script.
# Sets up argument parsing, processes the arguments, and executes the appropriate action based on the provided command-line arguments.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify Planetary Nomenclature')
    parser.add_argument('-a', '--action', help='one of the followings: collect, identify, end_to_end (collect followed by identify).')
    parser.add_argument('-t', '--target', help='target (ie, Moon, Mars, etc) is required. Has to be capitalized.')
    parser.add_argument('-f', '--feature_type', help='feature type (ie, Crater or Albedo Feature), is one of the options, either feature type or feature name are required, but no both. Has to be capitalized and in singular form.')
    parser.add_argument('-n', '--feature_name', help='feature name (ie Apollo, Atlas), is one of the options, either feature type or feature name are required, but no both. Has to be capitalized.')
    parser.add_argument('-k', '--keyword', help='knowledge graph keyword to add or remove manually.')
    parser.add_argument('-c', '--confidence_score', help='(optional) only applicable for action=retrieve_identified_entities, if specified only the identified entities with confidence score >= this score are returned.')
    parser.add_argument('-d', '--days', help='(optional) only applicable for action=retrieve_identified_entities, if specified only the entities identified in the past many days are returned.')
    parser.add_argument('-s', '--timestamp', help='(Optional) Applicable only when action=identify is specified. If provided, it should be in the format YYYY-MM-DD. Solr records from this date (inclusive) are considered for processing. Records with full-text modifications on or after this date will also be included, even if their original date is earlier.')
    parser.add_argument('-u', '--usgs_update', help='usgs updated list in csv format with five columns: Feature_ID,Clean_Feature_Name,Target,Feature_Type,Approval_Date,Approval_Status.')
    args = parser.parse_args()
    if args.action:
        action_type = map_input_param_to_action_type(args.action)
        if action_type == PLANETARYNAMES_PIPELINE_ACTION.invalid:
            logger.info(f"Invalid action arg `{args.action}`! Terminating!")
            sys.exit(1)
    else:
        logger.info('The action arg (-a) is needed for processing! Terminating!')
        sys.exit(1)

    # timestamp is only for identification action
    default_timestamp = config['PLANETARYNAMES_PIPELINE_DEFAULT_TIMESTAMP']
    if action_type in [PLANETARYNAMES_PIPELINE_ACTION.identify, PLANETARYNAMES_PIPELINE_ACTION.end_to_end]:
        if args.timestamp:
            try:
                # verify the format of the timestamp
                timestamp = datetime.strptime(args.timestamp, '%Y-%m-%d')
            except ValueError:
                logger.error(f"The timestamp '{args.timestamp}' is not in the correct format YYYY-MM-DD. Default 2000-01-01 is used.")
                timestamp = default_timestamp
        else:
            timestamp = default_timestamp
    else:
        logger.info("`timestamp` argument is only applicable when action=identify. Ignoring the timestamp using default 2000-01-01.")
        timestamp = default_timestamp

    # the only action command with no required parameter
    if action_type == PLANETARYNAMES_PIPELINE_ACTION.retrieve_identified_entities:
        current_target, current_feature_type, current_feature_names = verify_args(args)
        current_feature_names = current_feature_names if len(current_feature_names) > 0 else ['']
        for feature_name in current_feature_names:
            results = app.get_named_entity_bibcodes(feature_name, current_feature_type, current_target, args.confidence_score, get_date(args.days))
            # TODO: find out how the user wants this results outputted
            if results:
                logger.info(f"Total entities fetched: {len(results)}")
    # the action with only one required parameter: the csv file extracted info from usgs recently
    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.update_database_with_usgs_entities:
        if args.usgs_update:
            data = read_updated_usgs_gazetteer(args.usgs_update)
            if data:
                feature_ids = {int(row['Feature_ID']) for row in data}
                entity_ids = set(app.get_feature_ids())
                new_feature_ids = list(feature_ids - entity_ids)
                # extract entries for the new_feature_ids and send to be checked for any new target or feature_type
                # to be inserted to the database
                new_entries = [row for row in data if int(row['Feature_ID']) in new_feature_ids]
                tables_updated = app.add_new_usgs_entities(new_entries)
                if tables_updated:
                    logger.info("New targets and feature types have been successfully updated.")
                else:
                    logger.error("Failed to update new targets and feature types.")
        else:
            logger.error("CSV file for USGS data update is missing. Please provide the CSV file using the '--usgs_update' argument.")
            sys.exit(1)
    else:
        if args.target:
            current_target, current_feature_type, current_feature_names = verify_args(args)
            if current_target and (current_feature_type or current_feature_names):
                # process one to many feature names
                # one is when feature name is entered, while many is when feature type is entered
                for feature_name in current_feature_names:
                    entity_args = EntityArgs(target=current_target,
                                             feature_type=current_feature_type,
                                             feature_type_plural=app.get_plural_feature_type_entity(current_feature_type),
                                             feature_name=feature_name,
                                             context_ambiguous_feature_names=app.get_context_ambiguous_feature_name(feature_name),
                                             multi_token_containing_feature_names=app.get_multi_token_containing_feature_name(feature_name),
                                             name_entity_labels=app.get_named_entity_label(),
                                             timestamp = timestamp,
                                             all_targets=app.get_target_entities())
                    # actions that needs to go through queue
                    if action_type in [PLANETARYNAMES_PIPELINE_ACTION.collect,
                                       PLANETARYNAMES_PIPELINE_ACTION.identify,
                                       PLANETARYNAMES_PIPELINE_ACTION.end_to_end]:
                        the_task = {'action_type': action_type, 'args': entity_args}
                        tasks.task_process_planetary_nomenclature.delay(the_task)

                    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.remove_the_most_recent:
                        KBH_rows_deleted, KB_rows_deleted = app.remove_most_recent_knowledge_base_records(feature_name_entity=feature_name,
                                                                                                          target_entity=current_target)
                        logger.info(f"Removed the most recent records of knowledge graph for feature name `{feature_name}` and target `{current_target}`.\n"
                                    f"Deleted {KBH_rows_deleted} rows from knowledge_base_history and {KB_rows_deleted} rows from knowledge_base.")
                    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.remove_all_but_last:
                        KBH_rows_deleted, KB_rows_deleted = app.remove_all_but_most_recent_knowledge_base_records(feature_name_entity=feature_name,
                                                                                                                  target_entity=current_target)
                        logger.info(f"Removed all but the most recent records of knowledge graph for feature name `{feature_name}` and target `{current_target}`.\n"
                                    f"Deleted {KBH_rows_deleted} rows from knowledge_base_history and {KB_rows_deleted} rows from knowledge_base.\n")
                    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.add_keyword_to_knowledge_graph:
                        if args.keyword:
                            rows_updated = app.append_to_knowledge_base_keywords(feature_name_entity=feature_name,target_entity=current_target, keyword=args.keyword)
                            logger.info(f"{rows_updated} rows updated by adding the keyword.")
                        else:
                            logger.info('Keyword (-k) is a required parameter for this action! Terminating!')
                    elif action_type == PLANETARYNAMES_PIPELINE_ACTION.remove_keyword_from_knowledge_graph:
                        if args.keyword:
                            rows_updated = app.remove_from_knowledge_base_keywords(feature_name_entity=feature_name,target_entity=current_target, keyword=args.keyword)
                            logger.info(f"{rows_updated} rows updated by removing the keyword.")
                        else:
                            logger.info('Keyword (-k) is a required parameter for this action! Terminating!')
            else:
                logger.info('Either valid feature type (-f) or valid feature name (-n) is needed for processing! Terminating!')
                sys.exit(1)
        else:
            logger.info('The target (-t) is needed for processing! Terminating!')
            sys.exit(1)
    sys.exit(0)