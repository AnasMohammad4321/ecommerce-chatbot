import pandas as pd
import logging
from typing import List, Dict, Any, Union
from utils.helpers import convert_to_snake_case

class SchemaInferenceAgent:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.schema: Dict[str, Dict[str, Union[List[Any], Dict[str, Any]]]] = {}

    def infer_schema(self) -> Dict[str, Dict[str, Union[List[Any], Dict[str, Any]]]]:
        """
        Infers the schema by extracting:
        - Column names (converted to snake_case)
        - Unique values for categorical columns (limited to 10 examples)
        - Summary statistics for numerical columns (min, max, mean, std, unique count)
        
        Returns:
            dict: {column_name: {'type': 'categorical' or 'numerical', 'values' (for categorical) or 'stats' (for numerical)}}
        """
        try:
            df = pd.read_csv(self.file_path)

            # Convert column names to snake_case
            df.columns = [convert_to_snake_case(col) for col in df.columns]

            schema_info = {}

            for column in df.columns:
                if df[column].dtype == 'object' or df[column].nunique() < 20:
                    schema_info[column] = {
                        "type": "categorical",
                        "values": df[column].dropna().unique()[:10].tolist() 
                    }
                else:  
                    schema_info[column] = {
                        "type": "numerical",
                        "stats": {
                            "min": float(df[column].min()) if pd.notna(df[column].min()) else None,
                            "max": float(df[column].max()) if pd.notna(df[column].max()) else None,
                            "mean": float(df[column].mean()) if pd.notna(df[column].mean()) else None,
                            "std": float(df[column].std()) if pd.notna(df[column].std()) else None,
                            "unique_count": int(df[column].nunique())
                        }
                    }

            self.schema = schema_info
            return self.schema

        except Exception as e:
            logging.error(f"Error inferring schema: {e}")
            return {}