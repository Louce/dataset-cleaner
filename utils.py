import pandas as pd
import os
from typing import Dict, List, Union, Any


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic information about the dataset
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        A dictionary containing dataset information
    """
    info = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'numeric_columns': list(df.select_dtypes(include=['int64', 'float64']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return info


def save_dataframe(df: pd.DataFrame, file_path: str, file_format: str = 'csv') -> bool:
    """
    Save a pandas DataFrame to a file
    
    Args:
        df: The pandas DataFrame to save
        file_path: The path to save the file to
        file_format: The format to save the file in ('csv' or 'excel')
        
    Returns:
        True if saving was successful, False otherwise
    """
    try:
        if file_format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif file_format.lower() == 'excel':
            df.to_excel(file_path, index=False)
        else:
            return False
        
        # Check if file was created
        if os.path.exists(file_path):
            return True
        else:
            return False
    except Exception as e:
        print(f"Error saving dataframe: {e}")
        return False


def get_data_type_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed information about data types in the dataset
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        A DataFrame with data type information
    """
    data_types = []
    
    for column in df.columns:
        dtype = df[column].dtype
        unique_count = df[column].nunique()
        missing_count = df[column].isnull().sum()
        missing_percent = (missing_count / len(df) * 100).round(2)
        
        data_types.append({
            'Column': column,
            'Data Type': str(dtype),
            'Unique Values': unique_count,
            'Missing Values': missing_count,
            'Missing Percentage': missing_percent
        })
    
    return pd.DataFrame(data_types)


def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """
    Detect outliers in a numerical column
    
    Args:
        df: The pandas DataFrame to analyze
        column: The column to check for outliers
        method: The method to use for outlier detection ('iqr' or 'zscore')
        
    Returns:
        A boolean Series indicating which rows contain outliers
    """
    if method == 'iqr':
        # IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        # Z-score method
        from scipy import stats
        z_scores = stats.zscore(df[column])
        
        return abs(z_scores) > 3
    
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
