import os
import csv
import re
from typing import List, Tuple, Dict
from collections import Counter
from datetime import datetime

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())


class FileIO():
    """
    FileIO is a class providing static methods for reading and writing CSV files.
    """

    @staticmethod
    def output_identified_entities(output_file: str, identified_entities: List[Tuple[str, str, str, str, int, float, str]]) -> bool:
        """
        writes identified entities to a CSV file

        :param output_file: str, the file path to write the CSV
        :param identified_entities: list of tuples, each containing:
                                    - bibcode: str, the bibcode of the entity
                                    - target: str, the target entity
                                    - feature_type: str, the feature type entity
                                    - feature_name: str, the feature name entity
                                    - feature_id: int, the feature ID from the database
                                    - confidence_score: float, the confidence score
                                    - date: str, the date in 'YYYY-MM-DD HH:MM:SS' format
        :return: bool, True if the file was written successfully, False otherwise
        """
        try:
            # Count the number of instances for each unique combination of bibcode, and feature ID
            counts = Counter((row[0], row[4]) for row in identified_entities)

            # Prepare the data for writing to the CSV
            aggregated_data = [
                {
                    'bibcode': bibcode,
                    'feature_id': feature_id,
                    'feature_name': next(row[3] for row in identified_entities if row[0] == bibcode and row[4] == feature_id),
                    'feature_type': next(row[2] for row in identified_entities if row[0] == bibcode and row[4] == feature_id),
                    'target': next(row[1] for row in identified_entities if row[0] == bibcode and row[4] == feature_id),
                    'num_instances': count
                }
                for (bibcode, feature_id), count in counts.items()
            ]

            file_exists = os.path.isfile(output_file)
            with open(output_file, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # write the header only if the file is new
                if not file_exists:
                    writer.writerow(['Feature ID', 'Feature Name', 'Feature Type', 'Target', 'Bibcode', 'Number of Instances'])

                # Write each identified entity to the file
                for entity in aggregated_data:
                    writer.writerow([
                        entity['feature_id'],
                        entity['feature_name'],
                        entity['feature_type'],
                        entity['target'],
                        entity['bibcode'],
                        entity['num_instances']
                    ])
                return True

        except Exception as e:
            logger.error(f"Failed to write identified entities to '{output_file}': {str(e)}")
            return False

    @staticmethod
    def output_knowledge_graph_keywords(output_file: str, feature_name: str, feature_type: str, target: str, label: str, keywords: List[str]) -> bool:
        """
        writes the keywords to a CSV file, appending to the file if it already exists

        :param output_file: str, the file path to write the CSV
        :param feature_name: str, the feature name entity
        :param feature_type: the feature type entity
        :param target: str, the target entity
        :param label: str, named entity label (e.g., planetary, unknown)
        :param keywords: list of keywords to write to the file
        :return: bool, True if writing is successful, False otherwise
        """
        try:
            file_exists = os.path.isfile(output_file)
            with open(output_file, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # write the header only if the file is new
                if not file_exists:
                    writer.writerow(['Feature Name', 'Feature Type', 'Target', 'Label', 'Keywords'])

                for keyword in keywords:
                    writer.writerow([feature_name, feature_type, target, label, keyword])

            return True
        except Exception as e:
            logger.error(f"Failed to write keywords to '{output_file}': {str(e)}")
            return False

    @staticmethod
    def load_usgs_entities(csv_file_path: str) -> List[Dict[str, str]]:
        """
        reads the updated usgs entities
        containing five columns: Feature_ID,Clean_Feature_Name,Target,Feature_Type,Approval_Date,Approval_Status

        :param csv_file_path: path to the CSV file.
        :return: list of dictionaries, each containing data from one row.
        """
        year_pattern = re.compile(r'\b\d{4}\b')
        data = []
        try:
            with open(csv_file_path, newline='', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                # there are lots of white space in the header line coming from the Planetary Name website
                # Feature ID	Clean          Feature Name	Target	Feature Type	Approval          Status	Approval Date
                # get current header line
                file_header = reader.fieldnames
                # use these as column headers
                new_header = ['entity_id', 'feature_name', 'target', 'feature_type', 'approval_status', 'approval_date']

                for row in reader:
                    # transform keys first
                    row = {new_key: row[old_key] for new_key, old_key in zip(new_header, file_header)}
                    # now process the row
                    if row['approval_status'].lower() == 'approved':
                        # extract year from `approval_date`
                        match = year_pattern.search(row['approval_date'])
                        if match:
                            row['approval_date'] = match.group(0)
                        else:
                            # date can be in the format such as 5-Oct-18, so extract the year
                            try:
                                parsed_date = datetime.strptime(row['approval_date'], '%d-%b-%y')
                                row['approval_date'] = str(parsed_date.year)
                            except ValueError:
                                # if parsing fails, keep the original value
                                pass
                        # split `feature_type` to have separate singular and plural (if any)
                        row['feature_type'], row['feature_type_plural'] = [x.strip() for x in (row['feature_type'].split(',') + [''])[:2]]
                        data.append(row)
        except FileNotFoundError:
            logger.error(f"File not found: {csv_file_path}. Please check the path and try again.")
            print(f"File not found: {csv_file_path}. Please check the path and try again.")
        except Exception as e:
            logger.error(f"An error occurred while reading the file: {e}")
            print(f"An error occurred while reading the file: {e}")
        return data
