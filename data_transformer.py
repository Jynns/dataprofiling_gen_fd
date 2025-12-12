from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class DataTransformer(ABC):
    """Base data transformer"""

    def __init__(self):
        pass

    @abstractmethod
    def transform(self, data_path, metadata_path):
        """Transform the data using the metadata."""
        pass


class IdentityTransformer(DataTransformer):
    """A transformer that returns the data unchanged. Dummy Transformer."""

    def __init__(self):
        super().__init__()

    def transform(self, data_path, metadata_path=None):
        """
        Return the data unchanged.
        
        Parameters:
        - data_path: Path to the input data CSV file.
        - metadata_path: Optional path to metadata (not used in this transformer).
        
        Returns:
        - DataFrame: The unchanged input data.
        """
        df = pd.read_csv(data_path)

        # Dummy transformation: iterate through each row (tuple)
        def transform_row(row):
            # Use metadata file to transform rows if needed
            return row
        
        df = df.apply(transform_row, axis=1)
        
        return df