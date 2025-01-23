import sys, os
project_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_home not in sys.path:
    sys.path.insert(0, project_home)


import unittest
import os
import csv
import tempfile

from unittest.mock import patch, mock_open, MagicMock

from adsplanetnamepipe.utils.file_io import FileIO
from adsplanetnamepipe.models import NamedEntityLabel

class TestFileIO(unittest.TestCase):

    def test_output_identified_entities(self):
        """ test the output_identified_entities method """

        temp_file_path = os.path.join(tempfile.gettempdir(), "test_identified_entities.csv")

        # output the data to the temp file
        result = FileIO.output_identified_entities(output_file=temp_file_path,
                                                   identified_entities=[('2002JGRE..107.5056I', 'Mars', 'Albedo Feature', 'Amenthes', 226, 0.82, '2025-01-22 18:02:49'),
                                                                        ('2012JGRE..117.4001S', 'Mars', 'Albedo Feature', 'Amenthes', 226, 0.88, '2025-01-22 23:07:02')]
                                                   )
        self.assertTrue(result)
        # call it again to add more data
        result = FileIO.output_identified_entities(output_file=temp_file_path,
                                                   identified_entities=[('2002JGRE..107.5056I', 'Mars', 'Albedo Feature', 'Arabia', 335, 0.82, '2025-01-23 11:02:49'),
                                                                        ('2012JGRE..117.4001S', 'Mars', 'Albedo Feature', 'Arabia', 335, 0.88, '2025-01-23 11:07:02')]
                                                   )
        self.assertTrue(result)


        # now read the file back and verify its contents
        with open(temp_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            rows = [header] + list(reader)

            expected_rows = [
                ['Feature ID', 'Feature Name', 'Feature Type', 'Target', 'Bibcode', 'Number of Instances'],
                ['226', 'Amenthes', 'Albedo Feature', 'Mars', '2002JGRE..107.5056I', '1'],
                ['226', 'Amenthes', 'Albedo Feature', 'Mars', '2012JGRE..117.4001S', '1'],
                ['335', 'Arabia', 'Albedo Feature', 'Mars', '2002JGRE..107.5056I', '1'],
                ['335', 'Arabia', 'Albedo Feature', 'Mars', '2012JGRE..117.4001S', '1'],
            ]

            for row, expected_row in zip(rows, expected_rows):
                print(row, expected_row, row == expected_row)
                self.assertEqual(row, expected_row)

        # remove the temp file
        os.remove(temp_file_path)

    @patch('adsplanetnamepipe.utils.file_io.logger')
    def test_output_identified_entities_exception(self, mock_logger):
        """ test output_identified_entities when there is an exception """

        with patch("csv.writer", side_effect=Exception("Generic error")):
            result = FileIO.output_identified_entities(output_file="mock_output.csv",
                                                       identified_entities=[('2002JGRE..107.5056I', 'Mars', 'Albedo Feature', 'Amenthes', 226, 0.82, '2025-01-22 18:02:49'), ('2002JGRE..107.5056I', 'Mars', 'Albedo Feature', 'Amenthes', 226, 0.81, '2025-01-22 23:07:02'), ('2012JGRE..117.4001S', 'Mars', 'Albedo Feature', 'Amenthes', 226, 0.88, '2025-01-22 18:02:49'),
                                                                            ('2012JGRE..117.4001S', 'Mars', 'Albedo Feature', 'Amenthes', 226, 0.88, '2025-01-22 23:07:02')]
            )
            self.assertFalse(result)
            mock_logger.error.assert_called_with("Failed to write identified entities to 'mock_output.csv': Generic error")

    def test_output_knowledge_graph_keywords(self):
        """ test writing and reading the output of knowledge graph keywords """

        temp_file_path = os.path.join(tempfile.gettempdir(), "test_knowledge_graph_keywords.csv")

        # output the data to the temp file
        result = FileIO.output_knowledge_graph_keywords(temp_file_path,
                                                        feature_name='Amenthes',
                                                        feature_type='Albedo Feature',
                                                        target='Mars',
                                                        label='planetary',
                                                        keywords=[['drainage', 'crater', 'valley', 'basin', 'area', 'impact', 'noachian', 'catchment', 'figure', 'dissected', 'plain', 'degraded', 'mar', 'study'],
                                                                 ['upland', 'noachian', 'crater', 'count', 'basin', 'herschel', 'area', 'floor', 'map', 'fresh'],
                                                                 ['dust', 'jul', 'hellas', 'cml', 'storm', 'opposition', 'mar', 'refi', 'jun', 'part', 'dark', 'figure', 'bright', 'cloud', 'lacus', 'image', 'elysium', 'parker', 'sct', 'mare']]
        )
        self.assertTrue(result)
        # call it again to add more data
        result = FileIO.output_knowledge_graph_keywords(temp_file_path,
                                                        feature_name='Amenthes',
                                                        feature_type='Albedo Feature',
                                                        target='Mars',
                                                        label='planetary',
                                                        keywords=[['significant', 'east', 'hale', 'ejecta', 'terminal', 'earth', 'streak', 'north', 'figure', 'site'],
                                                                 ['hale', 'occur', 'crater', 'silica', 'mars', 'isidis', 'lower', 'glass', 'greater', 'sediment'],
                                                                 ['melt', 'hale', 'crater', 'mars', 'isidis', 'weathering', 'figure', 'formation']]
        )
        self.assertTrue(result)

        # now read the file back and verify its contents
        with open(temp_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            rows = [header] + list(reader)

            expected_rows = [
                ['Feature Name', 'Feature Type', 'Target', 'Label', 'Keywords'],
                ['Amenthes', 'Albedo Feature', 'Mars', 'planetary', "['drainage', 'crater', 'valley', 'basin', 'area', 'impact', 'noachian', 'catchment', 'figure', 'dissected', 'plain', 'degraded', 'mar', 'study']"],
                ['Amenthes', 'Albedo Feature', 'Mars', 'planetary', "['upland', 'noachian', 'crater', 'count', 'basin', 'herschel', 'area', 'floor', 'map', 'fresh']"],
                ['Amenthes', 'Albedo Feature', 'Mars', 'planetary', "['dust', 'jul', 'hellas', 'cml', 'storm', 'opposition', 'mar', 'refi', 'jun', 'part', 'dark', 'figure', 'bright', 'cloud', 'lacus', 'image', 'elysium', 'parker', 'sct', 'mare']"],
                ['Amenthes', 'Albedo Feature', 'Mars', 'planetary', "['significant', 'east', 'hale', 'ejecta', 'terminal', 'earth', 'streak', 'north', 'figure', 'site']"],
                ['Amenthes', 'Albedo Feature', 'Mars', 'planetary', "['hale', 'occur', 'crater', 'silica', 'mars', 'isidis', 'lower', 'glass', 'greater', 'sediment']"],
                ['Amenthes', 'Albedo Feature', 'Mars', 'planetary', "['melt', 'hale', 'crater', 'mars', 'isidis', 'weathering', 'figure', 'formation']"],
            ]

            for row, expected_row in zip(rows, expected_rows):
                print(row, expected_row, row == expected_row)
                self.assertEqual(row, expected_row)

        # remove the temp file
        os.remove(temp_file_path)

    @patch('adsplanetnamepipe.utils.file_io.logger')
    def test_output_knowledge_graph_keywords_exception(self, mock_logger):
        """ test output_knowledge_graph_keywords when there is an exception """

        with patch("csv.writer", side_effect=Exception("Generic error")):
            result = FileIO.output_knowledge_graph_keywords(output_file="mock_output.csv",
                                                            feature_name="Amenthes",
                                                            feature_type="Albedo Feature",
                                                            target="Mars",
                                                            label="planetary",
                                                            keywords=["crater", "basin", "valley"]
            )
            self.assertFalse(result)
            mock_logger.error.assert_called_with("Failed to write keywords to 'mock_output.csv': Generic error")

    def test_input_usgs_entities(self):
        """ test reading the USGS terms from a real CSV file """

        usgs_csv_file = os.path.join(os.path.dirname(__file__), "stubdata", "usgs_terms.csv")
        results = FileIO.load_usgs_entities(usgs_csv_file)

        expected_results = [
            {'entity_id': '9340', 'feature_name': '8 Homeward', 'target': 'Moon', 'feature_type': 'Crater', 'approval_status': 'Approved', 'approval_date': '2018', 'feature_type_plural': 'craters'},
            {'entity_id': '1', 'feature_name': 'Aachen', 'target': 'Mathilde', 'feature_type': 'Crater', 'approval_status': 'Approved', 'approval_date': '2000', 'feature_type_plural': 'craters'},
            {'entity_id': '2', 'feature_name': 'Aananin', 'target': 'Rhea', 'feature_type': 'Crater', 'approval_status': 'Approved', 'approval_date': '1982', 'feature_type_plural': 'craters'},
            {'entity_id': '6981', 'feature_name': 'Aaru', 'target': 'Titan', 'feature_type': 'Albedo Feature', 'approval_status': 'Approved', 'approval_date': '2006', 'feature_type_plural': ''},
            {'entity_id': '3', 'feature_name': 'Ababinili Patera', 'target': 'Io', 'feature_type': 'Patera', 'approval_status': 'Approved', 'approval_date': '2003', 'feature_type_plural': 'paterae'}
        ]
        self.assertEqual(len(results), len(expected_results))
        for i, row in enumerate(results):
            self.assertEqual(row, expected_results[i])

    @patch('adsplanetnamepipe.utils.file_io.logger')
    def test_input_usgs_entities_when_exceptions(self, mock_logger):
        """ test load_usgs_entities when there are exceptions"""

        invalid_file_path = 'usgs_terms.csv'
        usgs_csv_file = os.path.join(os.path.dirname(__file__), "stubdata", "usgs_terms.csv")

        # case 1: invalid filename
        result = FileIO.load_usgs_entities(invalid_file_path)
        self.assertEqual(result, [])
        mock_logger.error.assert_called_with(f"File not found: {invalid_file_path}. Please check the path and try again.")

        # case 2: when something goes wrong while reading the file
        with patch("csv.DictReader", side_effect=Exception("Generic error")):
            result = FileIO.load_usgs_entities(usgs_csv_file)
            self.assertEqual(result, [])
            mock_logger.error.assert_called_with("An error occurred while reading the file: Generic error")

        # case 3: when date was not parsed properly
        with patch("csv.DictReader") as mock_dict_reader:
            mock_dict_reader_instance = MagicMock()
            mock_dict_reader_instance.fieldnames = ['Feature_ID', 'Clean_Feature_Name', 'Target', 'Feature_Type', 'Approval_Status', 'Approval_Date']
            mock_dict_reader_instance.__iter__.return_value = iter([{'Feature_ID': '9340', 'Clean_Feature_Name': '8 Homeward', 'Target': 'Moon', 'Feature_Type': 'Crater, craters', 'Approval_Status': 'Approved', 'Approval_Date': '5/Oct/18'}])
            mock_dict_reader.return_value = mock_dict_reader_instance
            result = FileIO.load_usgs_entities(usgs_csv_file)

            expected_results = [{'entity_id': '9340', 'feature_name': '8 Homeward', 'target': 'Moon', 'feature_type': 'Crater', 'approval_status': 'Approved', 'approval_date': '5/Oct/18', 'feature_type_plural': 'craters'}]
            self.assertEqual(result, expected_results)

    def test_verify_label(self):
        """ test static method verify_label of the NamedEntityLabel used to decide which knowleadge graph keywords to output """

        self.assertEqual(NamedEntityLabel.verify_label("planetary"), "planetary")
        self.assertEqual(NamedEntityLabel.verify_label("unknown"), "unknown")
        self.assertEqual(NamedEntityLabel.verify_label(""), "planetary")
        self.assertEqual(NamedEntityLabel.verify_label("invalid"), "planetary")
        self.assertEqual(NamedEntityLabel.verify_label(None), "planetary")


if __name__ == "__main__":
    unittest.main()
