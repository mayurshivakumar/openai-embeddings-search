import pandas as pd


class CsvLoader:
    def __init__(self, file_path):
        """
        Initialize a CsvLoader object with a given file path.

        Args:
            file_path (str): The path of the CSV file to load.
        """
        self.file_path = file_path
        self.data = pd.DataFrame()

    def read_csv(self):
        """
        Read the CSV file specified in the object's file_path attribute
        and load it into a pandas DataFrame.

        Raises:
            FileNotFoundError: If the file specified in file_path cannot be found.
            Exception: If any other error occurs while reading the CSV file.
        """
        try:
            self.data = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"Error: {e}")

    def get_row_count(self):
        """
        Get the number of rows in the DataFrame.

        Returns:
            int: The number of rows in the DataFrame.
        """
        return len(self.data)

    def get_column_count(self):
        """
        Get the number of columns in the DataFrame.

        Returns:
            int: The number of columns in the DataFrame.
        """
        return len(self.data.columns)

    def get_data(self):
        """
        Get the DataFrame loaded from the CSV file.

        Returns:
            pandas.DataFrame: The DataFrame loaded from the CSV file.
        """
        return self.data
