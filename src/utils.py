import pandas as pd
import numpy as np
import os
import io
import re
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("csv_cleaner")

# Constants
MAX_FILE_SIZE_MB = 100
SUPPORTED_ENCODINGS = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
DATE_FORMATS = [
    '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', 
    '%Y-%m-%d %H:%M:%S', '%d-%m-%Y', '%m-%d-%Y'
]

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
        'numeric_columns': list(df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'missing_values': df.isna().sum().sum(),
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
            df.to_excel(file_path, index=False, engine='openpyxl')
        else:
            return False
        
        # Check if file was created
        if os.path.exists(file_path):
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Error saving dataframe: {e}")
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
        missing_count = df[column].isna().sum()
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
        z_scores = stats.zscore(df[column], nan_policy='omit')
        
        return abs(z_scores) > 3
    
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")

def validate_file(file_buffer, file_type=None):
    """
    Validates uploaded file.
    
    Args:
        file_buffer: BytesIO object containing file data
        file_type: Optional string indicating file type ('csv' or 'excel')
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    # Check if file is empty
    file_buffer.seek(0, os.SEEK_END)
    file_size = file_buffer.tell()
    file_buffer.seek(0)
    
    if file_size == 0:
        return False, "Uploaded file is empty"
    
    # If file_type not specified, try to infer from content
    if not file_type:
        # Read first few bytes to detect file type
        header = file_buffer.read(8)
        file_buffer.seek(0)
        
        # Excel file signatures
        xlsx_sig = b'\x50\x4B\x03\x04'  # XLSX files start with PK
        xls_sig = b'\xD0\xCF\x11\xE0'   # XLS files start with D0CF11E0
        
        if header.startswith(xlsx_sig) or header.startswith(xls_sig):
            file_type = 'excel'
        else:
            file_type = 'csv'
    
    # Validate based on file type
    if file_type == 'excel':
        try:
            # Try to read as Excel file
            pd.read_excel(file_buffer, nrows=5)
            file_buffer.seek(0)
            return True, "Valid Excel file"
        except Exception as e:
            return False, f"Invalid Excel file: {str(e)}"
    else:
        try:
            # Try to read as CSV file (existing validation)
            file_buffer.seek(0)
            sample = file_buffer.read(1024).decode('utf-8', errors='ignore')
            file_buffer.seek(0)
            
            # Basic CSV validation - check if it has commas or delimiters
            if ',' not in sample and ';' not in sample and '\t' not in sample:
                return False, "File does not appear to be a valid CSV"
                
            return True, "Valid CSV file"
        except Exception as e:
            return False, f"Invalid CSV file: {str(e)}"

def detect_csv_dialect(file_obj: io.BytesIO) -> Dict:
    """
    Detect CSV dialect (delimiter, quotechar, etc.)
    
    Args:
        file_obj: The uploaded file object
        
    Returns:
        Dict with detected CSV dialect parameters
    """
    import csv
    try:
        # Read a sample to detect dialect
        sample = file_obj.read(4096).decode('utf-8')
        file_obj.seek(0)  # Reset file pointer
        
        # Count occurrences of potential delimiters
        delimiters = [',', ';', '\t', '|']
        counts = {d: sample.count(d) for d in delimiters}
        likely_delimiter = max(counts, key=counts.get)
        
        # Detect quote character
        quotes = ['"', "'"]
        quote_counts = {q: sample.count(q) for q in quotes}
        likely_quote = max(quote_counts, key=quote_counts.get) if any(quote_counts.values()) else '"'
        
        return {
            'delimiter': likely_delimiter,
            'quotechar': likely_quote,
            'escapechar': '\\',
            'doublequote': True
        }
    except Exception as e:
        logger.error(f"Error detecting CSV dialect: {str(e)}")
        # Default to standard CSV format
        return {
            'delimiter': ',',
            'quotechar': '"',
            'escapechar': '\\',
            'doublequote': True
        }

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect and categorize column types more accurately
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict mapping column names to detected types
    """
    column_types = {}
    
    for col in df.columns:
        # Skip if column is empty
        if df[col].isna().all():
            column_types[col] = 'empty'
            continue
        
        # Check if it's an ID column
        if col.lower().endswith('id') or col.lower().startswith('id'):
            column_types[col] = 'id'
            continue
        
        # Get the pandas dtype
        dtype = df[col].dtype
        
        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(dtype):
            # Check if it could be a categorical disguised as numeric
            unique_ratio = df[col].nunique() / len(df[col].dropna())
            
            if unique_ratio < 0.05 and df[col].nunique() < 10:
                column_types[col] = 'categorical'
            elif pd.api.types.is_integer_dtype(dtype) or (pd.api.types.is_float_dtype(dtype) and df[col].dropna().apply(lambda x: float(x).is_integer()).all()):
                column_types[col] = 'integer'
            else:
                column_types[col] = 'numeric'
            continue
        
        # Handle potential datetime columns
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            # Sample non-null values
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                sample = non_null_values.sample(min(10, len(non_null_values)))
                
                # Try parsing as dates
                date_success = True
                for val in sample:
                    if not try_parse_date(str(val)):
                        date_success = False
                        break
                
                if date_success:
                    column_types[col] = 'datetime'
                    continue
                
                # Check if it could be a boolean
                if set(df[col].dropna().astype(str).str.lower().unique()) <= {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0'}:
                    column_types[col] = 'boolean'
                    continue
                
                # Check if it's probably categorical or text
                unique_ratio = df[col].nunique() / len(df[col].dropna())
                avg_len = df[col].dropna().astype(str).apply(len).mean()
                
                if unique_ratio < 0.2 or df[col].nunique() < 20:
                    column_types[col] = 'categorical'
                elif avg_len > 100:
                    column_types[col] = 'text'
                else:
                    column_types[col] = 'string'
            else:
                column_types[col] = 'unknown'
        elif pd.api.types.is_categorical_dtype(dtype):
            column_types[col] = 'categorical'
        else:
            # Default for other types
            column_types[col] = str(dtype)
    
    return column_types

def try_parse_date(date_string: str) -> Optional[datetime]:
    """
    Try to parse a string as a date using various formats
    
    Args:
        date_string: String to try to parse as a date
        
    Returns:
        Parsed datetime object or None if parsing failed
    """
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_string.strip(), fmt)
        except ValueError:
            continue
    return None

def sanitize_string(input_string: str) -> str:
    """
    Sanitize a string to prevent injection attacks
    
    Args:
        input_string: String to sanitize
        
    Returns:
        Sanitized string
    """
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"/\'%;&]', '', input_string)
    return sanitized

def generate_secure_filename(original_filename: str) -> str:
    """
    Generate a secure filename based on the original name and a timestamp
    
    Args:
        original_filename: Original filename
        
    Returns:
        Secure filename
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    basename = os.path.splitext(os.path.basename(original_filename))[0]
    basename = re.sub(r'[^a-zA-Z0-9_-]', '_', basename)  # Replace non-alphanumeric
    
    # Add hash for uniqueness
    unique_hash = hashlib.md5(f"{basename}{timestamp}".encode()).hexdigest()[:8]
    
    return f"{basename}_{timestamp}_{unique_hash}"

def estimate_memory_usage(df: pd.DataFrame) -> float:
    """
    Estimate memory usage of DataFrame in MB
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Estimated memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / (1024 * 1024)

def is_percentage_column(col_name: str, values: pd.Series) -> bool:
    """
    Detect if a column likely contains percentage values
    
    Args:
        col_name: Column name
        values: Series of values
        
    Returns:
        Boolean indicating if column likely contains percentages
    """
    # Check name hints
    name_hints = ['percent', 'pct', 'rate', 'ratio', 'score']
    if any(hint in col_name.lower() for hint in name_hints):
        # Check if values are between 0 and 100 or 0 and 1
        non_null = values.dropna()
        if len(non_null) == 0:
            return False
            
        if non_null.between(0, 1).all() or non_null.between(0, 100).all():
            return True
            
    return False

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000) -> List[pd.DataFrame]:
    """
    Split a large DataFrame into chunks for processing
    
    Args:
        df: DataFrame to split
        chunk_size: Number of rows per chunk
        
    Returns:
        List of DataFrame chunks
    """
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i+chunk_size].copy())
    return chunks

def get_file_type(file_buffer, filename):
    """
    Determine file type from extension and content
    
    Args:
        file_buffer: BytesIO object containing file data
        filename: Original filename with extension
        
    Returns:
        String indicating file type ('csv', 'excel', or 'unknown')
    """
    # First check extension
    if filename.lower().endswith(('.xlsx', '.xls')):
        return 'excel'
    elif filename.lower().endswith('.csv'):
        return 'csv'
        
    # If extension is ambiguous, check content signatures
    header = file_buffer.read(8)
    file_buffer.seek(0)
    
    # Excel file signatures
    xlsx_sig = b'\x50\x4B\x03\x04'  # XLSX files start with PK
    xls_sig = b'\xD0\xCF\x11\xE0'   # XLS files start with D0CF11E0
    
    if header.startswith(xlsx_sig) or header.startswith(xls_sig):
        return 'excel'
    
    # CSV detection - check if it looks like text with delimiters
    try:
        sample = file_buffer.read(1024).decode('utf-8', errors='ignore')
        file_buffer.seek(0)
        if ',' in sample or ';' in sample or '\t' in sample:
            return 'csv'
    except:
        pass
        
    return 'unknown'
