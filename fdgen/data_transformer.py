from pathlib import Path
import pandas as pd
import numpy as np
import json
from abc import ABC, abstractmethod
from .llm_utils import generate_column_desc

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


class LLMDescTransformer(DataTransformer):
    """A transformer that uses LLM to generate column descriptions."""

    def __init__(self):
        super().__init__()
        
    def transform(self, data_path, metadata_path):
        """
        Transform the data using the LLM function.
        
        Parameters:
        - data_path: Path to the input data CSV file.
        - metadata_path: Optional path to metadata (not used in this transformer).
        
        Returns:
        - DataFrame: The transformed data as returned by the LLM function.
        """
        
        dataset_df = pd.read_csv(data_path)
        
        # Load column descriptions from metadata file (nested by table name)
        with open(metadata_path, 'r') as f:
            all_descriptions = json.load(f)

        print("Column Descriptions loaded from metadata.")

        # Add new columns with descriptions based on table and column names
        def get_lhs_desc(row):
            """Get description for LHS column based on table name."""
            tablename = row.get('tablename')
            lhs = row.get('lhs')
            
            # Handle multi-column LHS (e.g., "col1, col2")
            if pd.isna(lhs) or pd.isna(tablename):
                return None
            
            # Strip .csv extension from tablename to match JSON keys
            # Also remove underscores to handle inconsistent naming (e.g., women_basketball -> womensbasketball)
            table_key = str(tablename).replace('.csv', '').replace('_', '')
            
            # Get table descriptions
            table_desc = all_descriptions.get(table_key)
            if not table_desc:
                return None
            
            # For composite columns, try to get individual descriptions
            if ',' in str(lhs):
                cols = [c.strip() for c in str(lhs).split(',')]
                descriptions = {}
                for col in cols:
                    if col in table_desc:
                        descriptions[col] = table_desc[col]
                return str(descriptions) if descriptions else None
            else:
                if lhs in table_desc:
                    return str({lhs: table_desc[lhs]})
            return None
        
        def get_rhs_desc(row):
            """Get description for RHS column based on table name."""
            tablename = row.get('tablename')
            rhs = row.get('rhs')
            
            if pd.isna(rhs) or pd.isna(tablename):
                return None
            
            # Strip .csv extension from tablename to match JSON keys
            # Also remove underscores to handle inconsistent naming (e.g., women_basketball -> womensbasketball)
            table_key = str(tablename).replace('.csv', '').replace('_', '')
            
            # Get table descriptions
            table_desc = all_descriptions.get(table_key)
            if not table_desc:
                return None
            
            if rhs in table_desc:
                return str({rhs: table_desc[rhs]})
            return None
        
        dataset_df['description_lhs'] = dataset_df.apply(get_lhs_desc, axis=1)
        dataset_df['description_rhs'] = dataset_df.apply(get_rhs_desc, axis=1)

        return dataset_df
        