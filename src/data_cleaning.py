import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
import re
from collections import defaultdict
from src.utils import detect_column_types, is_percentage_column

logger = logging.getLogger("csv_cleaner")

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner with a DataFrame
        
        Args:
            df: Input DataFrame to clean
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.transformations = []
        self.column_types = detect_column_types(self.df)
        self.numeric_cols = [col for col, type_ in self.column_types.items() 
                           if type_ in ['numeric', 'integer']]
        self.categorical_cols = [col for col, type_ in self.column_types.items() 
                               if type_ in ['categorical', 'boolean']]
        self.datetime_cols = [col for col, type_ in self.column_types.items() 
                            if type_ == 'datetime']
        self.id_cols = [col for col, type_ in self.column_types.items() 
                      if type_ == 'id']
        self.cleaned_df = None
        self.memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        self.is_large_dataset = self.memory_usage_mb > 100  # 100 MB threshold
        self.progress_callback = None
        
        # Initialize metadata
        self.metadata = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': self.memory_usage_mb,
            'column_types': self.column_types,
            'cleaning_done': False
        }
    
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    def update_progress(self, message, progress=None):
        """Update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            logger.info(message)
    
    def get_column_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each column in the DataFrame.
        
        Returns:
            Dictionary with column statistics
        """
        stats = {}
        
        for col in self.df.columns:
            col_type = self.column_types.get(col, 'unknown')
            col_stats = {
                'type': col_type,
                'missing_count': self.df[col].isna().sum(),
                'missing_percent': round(self.df[col].isna().sum() / len(self.df) * 100, 2),
                'unique_values': self.df[col].nunique()
            }
            
            # Add type-specific statistics
            if col_type in ['numeric', 'integer']:
                col_stats.update({
                    'min': self.df[col].min() if not self.df[col].isna().all() else None,
                    'max': self.df[col].max() if not self.df[col].isna().all() else None,
                    'mean': self.df[col].mean() if not self.df[col].isna().all() else None,
                    'median': self.df[col].median() if not self.df[col].isna().all() else None,
                    'std': self.df[col].std() if not self.df[col].isna().all() else None
                })
            elif col_type == 'categorical':
                # Get top 5 most common values
                value_counts = self.df[col].value_counts().head(5).to_dict()
                col_stats['top_values'] = value_counts
            
            stats[col] = col_stats
        
        return stats
    
    def identify_missing_values(self) -> Dict[str, Any]:
        """
        Identify missing values in the dataset without automatically filling them.
            
        Returns:
            Dictionary with missing value information
        """
        # Check for columns with missing values
        missing_info = {}
        missing_counts = self.df.isna().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        
        if len(columns_with_missing) == 0:
            return {
                'has_missing': False,
                'message': "No missing values found in the dataset.",
                'columns': {}
            }
        
        total_rows = len(self.df)
        missing_info['has_missing'] = True
        missing_info['total_missing_cells'] = missing_counts.sum()
        missing_info['total_rows'] = total_rows
        missing_info['columns'] = {}
        
        for col, count in columns_with_missing.items():
            percent = round((count / total_rows) * 100, 2)
            col_type = self.column_types.get(col, 'unknown')
            
            # Generate suggested replacement strategies based on column type
            strategies = []
            
            # Add synergy strategy to all column types
            synergy_description = "Smart fill based on data patterns and relationships"
            
            if col_type in ['numeric', 'integer']:
                # Get statistics for numerical columns
                if not self.df[col].isna().all():
                    mean_val = round(self.df[col].mean(), 2)
                    median_val = round(self.df[col].median(), 2)
                    
                    strategies = [
                        {'name': 'synergy', 'value': None, 'description': synergy_description},
                        {'name': 'mean', 'value': mean_val, 'description': f"Replace with mean ({mean_val})"},
                        {'name': 'median', 'value': median_val, 'description': f"Replace with median ({median_val})"},
                        {'name': 'zero', 'value': 0, 'description': "Replace with zero"},
                        {'name': 'custom', 'value': None, 'description': "Replace with custom value"},
                        {'name': 'remove', 'value': None, 'description': "Remove rows with missing values"}
                    ]
                else:
                    strategies = [
                        {'name': 'synergy', 'value': None, 'description': synergy_description},
                        {'name': 'zero', 'value': 0, 'description': "Replace with zero"},
                        {'name': 'custom', 'value': None, 'description': "Replace with custom value"},
                        {'name': 'remove', 'value': None, 'description': "Remove rows with missing values"}
                    ]
                    
            elif col_type in ['categorical', 'boolean']:
                # For categorical, use mode (most common value)
                if not self.df[col].isna().all():
                    mode_val = self.df[col].mode()[0]
                    strategies = [
                        {'name': 'synergy', 'value': None, 'description': synergy_description},
                        {'name': 'mode', 'value': mode_val, 'description': f"Replace with most common value ({mode_val})"},
                        {'name': 'unknown', 'value': 'Unknown', 'description': "Replace with 'Unknown'"},
                        {'name': 'custom', 'value': None, 'description': "Replace with custom value"},
                        {'name': 'remove', 'value': None, 'description': "Remove rows with missing values"}
                    ]
                else:
                    strategies = [
                        {'name': 'synergy', 'value': None, 'description': synergy_description},
                        {'name': 'unknown', 'value': 'Unknown', 'description': "Replace with 'Unknown'"},
                        {'name': 'custom', 'value': None, 'description': "Replace with custom value"},
                        {'name': 'remove', 'value': None, 'description': "Remove rows with missing values"}
                    ]
            
            else:
                # Default strategies for other types
                strategies = [
                    {'name': 'synergy', 'value': None, 'description': synergy_description},
                    {'name': 'empty_string', 'value': '', 'description': "Replace with empty string"},
                    {'name': 'unknown', 'value': 'Unknown', 'description': "Replace with 'Unknown'"},
                    {'name': 'custom', 'value': None, 'description': "Replace with custom value"},
                    {'name': 'remove', 'value': None, 'description': "Remove rows with missing values"}
                ]
            
            # Get sample of rows with missing values
            missing_indices = self.df[self.df[col].isna()].index.tolist()[:5]
            sample_rows = []
            
            for idx in missing_indices:
                row_data = self.df.loc[idx].to_dict()
                # Convert to string to handle non-serializable objects
                row_data = {k: str(v) if not pd.isna(v) else None for k, v in row_data.items()}
                sample_rows.append({"index": int(idx), "data": row_data})
            
            missing_info['columns'][col] = {
                'count': int(count),
                'percent': percent,
                'type': col_type,
                'strategies': strategies,
                'sample_rows': sample_rows
            }
        
        return missing_info
    
    def fill_missing_values(self, strategies: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fill missing values in the dataset based on the provided strategies.
        
        Args:
            strategies: Dictionary mapping column names to strategies and values
            
        Returns:
            Dictionary with results of missing value handling
        """
        results = {
            'columns_modified': [],
            'rows_modified': 0,
            'rows_removed': 0,
            'transformations': []
        }
        
        rows_to_drop = set()
        
        for col, strategy in strategies.items():
            strategy_name = strategy.get('strategy', 'none')
            custom_value = strategy.get('value', None)
            
            # Skip columns that don't exist
            if col not in self.df.columns:
                continue
            
            # Check for missing values
            missing_mask = self.df[col].isna()
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                continue
            
            if strategy_name == 'remove':
                # Mark rows for removal
                rows_to_drop.update(self.df[missing_mask].index)
                results['transformations'].append({
                    'column': col,
                    'operation': 'remove_missing',
                    'details': f"Marked {missing_count} rows with missing values in column '{col}' for removal"
                })
            else:
                # Fill missing values
                fill_value = None
                operation_name = f'fill_missing_{strategy_name}'
                details = ""
                
                if strategy_name == 'synergy':
                    # Smart synergy fill based on data relationships and patterns
                    try:
                        col_type = self.column_types.get(col, 'unknown')
                        
                        if col_type in ['numeric', 'integer']:
                            # Try KNN imputation first for numeric columns
                            if len(self.df) > 3 and len(self.numeric_cols) >= 2:
                                # Use numerical columns as features for imputation
                                feature_cols = [c for c in self.numeric_cols if c != col and self.df[c].isna().sum() / len(self.df) < 0.3]
                                
                                if len(feature_cols) >= 1:
                                    # Enough features for KNN imputation
                                    from sklearn.impute import KNNImputer
                                    
                                    # Prepare data for imputation
                                    impute_df = self.df[[col] + feature_cols].copy()
                                    
                                    # Number of neighbors - adjust based on dataset size
                                    n_neighbors = min(5, len(self.df) // 10) if len(self.df) > 50 else 2
                                    n_neighbors = max(2, n_neighbors)  # Minimum 2 neighbors
                                    
                                    # Create and fit the imputer
                                    imputer = KNNImputer(n_neighbors=n_neighbors)
                                    imputed_values = imputer.fit_transform(impute_df)
                                    
                                    # Extract imputed values for the target column
                                    imputed_series = pd.Series(imputed_values[:, 0], index=self.df.index)
                                    
                                    # Apply only to missing values
                                    self.df.loc[missing_mask, col] = imputed_series[missing_mask]
                                    
                                    details = f"Filled {missing_count} missing values in column '{col}' using KNN imputation with {len(feature_cols)} related features"
                                    operation_name = 'fill_missing_synergy_knn'
                                else:
                                    # Fall back to median for numeric with insufficient features
                                    fill_value = self.df[col].median() if not self.df[col].isna().all() else 0
                                    self.df.loc[missing_mask, col] = fill_value
                                    details = f"Filled {missing_count} missing values in column '{col}' with median ({fill_value})"
                                    operation_name = 'fill_missing_synergy_median'
                            else:
                                # Fall back to median for small datasets
                                fill_value = self.df[col].median() if not self.df[col].isna().all() else 0
                                self.df.loc[missing_mask, col] = fill_value
                                details = f"Filled {missing_count} missing values in column '{col}' with median ({fill_value})"
                                operation_name = 'fill_missing_synergy_median'
                                
                        elif col_type in ['categorical', 'boolean']:
                            # Check if we can predict categorical values
                            if len(self.categorical_cols) > 1 and self.df[col].nunique() < 10:
                                # Try to use other categorical columns to predict
                                predictor_cols = [c for c in self.categorical_cols if c != col and self.df[c].isna().sum() / len(self.df) < 0.3]
                                
                                if len(predictor_cols) >= 1:
                                    # Use mode conditioned on similar rows
                                    # Group by predictor columns and get mode for each group
                                    grp_cols = predictor_cols[:min(3, len(predictor_cols))]  # Limit to max 3 predictors
                                    
                                    # Get non-missing rows for training
                                    train_mask = ~self.df[col].isna()
                                    
                                    # For each missing value, find most common value in similar rows
                                    for idx in self.df[missing_mask].index:
                                        # Get values of predictor columns for this row
                                        row_values = self.df.loc[idx, grp_cols]
                                        
                                        # Find similar rows (exact match on predictors)
                                        similar_mask = True
                                        for gc, val in zip(grp_cols, row_values):
                                            similar_mask = similar_mask & (self.df[gc] == val)
                                        
                                        similar_mask = similar_mask & train_mask
                                        
                                        if similar_mask.sum() > 0:
                                            # Get most common value in similar rows
                                            most_common = self.df.loc[similar_mask, col].mode()[0]
                                            self.df.loc[idx, col] = most_common
                                        else:
                                            # No similar rows, fall back to overall mode
                                            most_common = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                                            self.df.loc[idx, col] = most_common
                                    
                                    details = f"Filled {missing_count} missing values in column '{col}' using conditional mode based on similar rows"
                                    operation_name = 'fill_missing_synergy_conditional'
                                else:
                                    # Fall back to overall mode
                                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                                    self.df.loc[missing_mask, col] = fill_value
                                    details = f"Filled {missing_count} missing values in column '{col}' with mode ({fill_value})"
                                    operation_name = 'fill_missing_synergy_mode'
                            else:
                                # Fall back to mode
                                fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                                self.df.loc[missing_mask, col] = fill_value
                                details = f"Filled {missing_count} missing values in column '{col}' with mode ({fill_value})"
                                operation_name = 'fill_missing_synergy_mode'
                        else:
                            # For other types, use mode or 'Unknown'
                            fill_value = 'Unknown'
                            self.df.loc[missing_mask, col] = fill_value
                            details = f"Filled {missing_count} missing values in column '{col}' with '{fill_value}'"
                            operation_name = 'fill_missing_synergy_default'
                    
                    except Exception as e:
                        # In case of error, fall back to simpler method
                        if self.column_types.get(col) in ['numeric', 'integer']:
                            fill_value = self.df[col].median() if not self.df[col].isna().all() else 0
                        else:
                            fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                            
                        self.df.loc[missing_mask, col] = fill_value
                        details = f"Filled {missing_count} missing values in column '{col}' with fallback value ({fill_value})"
                        operation_name = 'fill_missing_synergy_fallback'
                    
                elif strategy_name == 'custom' and custom_value is not None:
                    fill_value = custom_value
                    self.df.loc[missing_mask, col] = fill_value
                    details = f"Filled {missing_count} missing values in column '{col}' with custom value ({fill_value})"
                elif strategy_name == 'mean':
                    fill_value = self.df[col].mean()
                    self.df.loc[missing_mask, col] = fill_value
                    details = f"Filled {missing_count} missing values in column '{col}' with mean ({fill_value})"
                elif strategy_name == 'median':
                    fill_value = self.df[col].median()
                    self.df.loc[missing_mask, col] = fill_value
                    details = f"Filled {missing_count} missing values in column '{col}' with median ({fill_value})"
                elif strategy_name == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else None
                    self.df.loc[missing_mask, col] = fill_value
                    details = f"Filled {missing_count} missing values in column '{col}' with mode ({fill_value})"
                elif strategy_name == 'zero':
                    fill_value = 0
                    self.df.loc[missing_mask, col] = fill_value
                    details = f"Filled {missing_count} missing values in column '{col}' with zero"
                elif strategy_name == 'empty_string':
                    fill_value = ''
                    self.df.loc[missing_mask, col] = fill_value
                    details = f"Filled {missing_count} missing values in column '{col}' with empty string"
                elif strategy_name == 'unknown':
                    fill_value = 'Unknown'
                    self.df.loc[missing_mask, col] = fill_value
                    details = f"Filled {missing_count} missing values in column '{col}' with 'Unknown'"
                
                # Store the indices of affected rows
                affected_indices = self.df[missing_mask].index.tolist()
                
                # Update results
                if len(affected_indices) > 0:
                    results['columns_modified'].append(col)
                    results['rows_modified'] += missing_count
                    
                    results['transformations'].append({
                        'column': col,
                        'operation': operation_name,
                        'details': details,
                        'affected_rows': affected_indices
                    })
        
        # Apply row removal after processing all columns
        if rows_to_drop:
            before_count = len(self.df)
            self.df = self.df.drop(index=list(rows_to_drop))
            removed_count = before_count - len(self.df)
            results['rows_removed'] = removed_count
            
            if removed_count > 0:
                results['transformations'].append({
                    'operation': 'remove_rows',
                    'details': f"Removed {removed_count} rows with missing values",
                    'affected_rows': list(rows_to_drop)
                })
        
        # Add transformations to the main list
        self.transformations.extend(results['transformations'])
        
        return results
    
    def identify_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """
        Identify outliers in numerical columns without modifying the data.
        
        Args:
            method: Method to use for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection (1.5 for IQR, 3 for zscore)
            
        Returns:
            Dictionary with outlier information
        """
        outlier_info = {
            'has_outliers': False,
            'columns': {},
            'total_outliers': 0
        }
        
        # Skip outlier detection if no numerical columns
        if not self.numeric_cols:
            return {
                'has_outliers': False,
                'message': "No numerical columns found for outlier detection."
            }
        
        total_outliers = 0
        
        for col in self.numeric_cols:
            # Skip columns with too many missing values
            if self.df[col].isna().sum() / len(self.df) > 0.5:
                continue
                
            # Skip ID columns and other columns that shouldn't have outlier detection
            if col in self.id_cols or 'id' in col.lower():
                continue
            
            # Calculate bounds based on the chosen method
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Identify outliers
                outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                
            elif method == 'zscore':
                # Apply Z-score method, handling NaN values
                z_scores = stats.zscore(self.df[col], nan_policy='omit')
                outliers_mask = abs(z_scores) > threshold
                
                # Create lower and upper bounds for reporting
                mean = self.df[col].mean()
                std = self.df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                raise ValueError("Method must be either 'iqr' or 'zscore'")
            
            # Count outliers
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                outlier_info['has_outliers'] = True
                total_outliers += outlier_count
                
                # Get basic stats
                stats_data = {
                    'count': int(outlier_count),
                    'percent': round((outlier_count / len(self.df)) * 100, 2),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'mean': float(self.df[col].mean()),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'method': method
                }
                
                # Get sample of outlier rows
                outlier_indices = self.df[outliers_mask].index.tolist()[:5]
                sample_rows = []
                
                for idx in outlier_indices:
                    row_data = self.df.loc[idx].to_dict()
                    # Convert to string to handle non-serializable objects
                    row_data = {k: str(v) if not pd.isna(v) else None for k, v in row_data.items()}
                    sample_rows.append({"index": int(idx), "data": row_data, "value": float(self.df.loc[idx, col])})
                
                # Add treatment options
                treatments = [
                    {'name': 'cap', 'description': f"Cap at {lower_bound:.2f} and {upper_bound:.2f}"},
                    {'name': 'remove', 'description': "Remove rows with outliers"},
                    {'name': 'keep', 'description': "Keep outliers unchanged"}
                ]
                
                outlier_info['columns'][col] = {
                    'stats': stats_data,
                    'sample_rows': sample_rows,
                    'treatments': treatments
                }
        
        outlier_info['total_outliers'] = total_outliers
        return outlier_info
    
    def handle_outliers(self, strategies: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle outliers in numerical columns based on the provided strategies.
        
        Args:
            strategies: Dictionary mapping column names to outlier handling strategies
            
        Returns:
            Dictionary with results of outlier handling
        """
        results = {
            'columns_modified': [],
            'rows_modified': 0,
            'rows_removed': 0,
            'transformations': []
        }
        
        rows_to_drop = set()
        
        for col, strategy in strategies.items():
            strategy_name = strategy.get('strategy', 'none')
            method = strategy.get('method', 'iqr')
            threshold = strategy.get('threshold', 1.5)
            
            # Skip columns that don't exist or aren't numerical
            if col not in self.df.columns or col not in self.numeric_cols:
                continue
            
            # Calculate bounds based on the chosen method
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Identify outliers
                outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                
            elif method == 'zscore':
                # Apply Z-score method
                z_scores = stats.zscore(self.df[col], nan_policy='omit')
                outliers_mask = abs(z_scores) > threshold
                
                # Create lower and upper bounds for capping
                mean = self.df[col].mean()
                std = self.df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                continue  # Skip if invalid method
            
            # Count outliers
            outlier_count = outliers_mask.sum()
            
            if outlier_count == 0:
                continue
                
            if strategy_name == 'remove':
                # Mark rows for removal
                rows_to_drop.update(self.df[outliers_mask].index)
                results['transformations'].append({
                    'column': col,
                    'operation': 'remove_outliers',
                    'details': f"Marked {outlier_count} rows with outliers in column '{col}' for removal"
                })
            elif strategy_name == 'cap':
                # Get the indices of affected rows
                affected_indices = self.df[outliers_mask].index.tolist()
                
                # Store original values for reporting
                original_values = self.df.loc[affected_indices, col].copy()
                
                # Cap outliers at the bounds
                self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                
                results['columns_modified'].append(col)
                results['rows_modified'] += outlier_count
                
                # Create detailed transformation record
                transformations = []
                for idx in affected_indices:
                    transformations.append({
                        'row_index': int(idx),
                        'original_value': float(original_values.loc[idx]),
                        'new_value': float(self.df.loc[idx, col])
                    })
                
                results['transformations'].append({
                    'column': col,
                    'operation': 'cap_outliers',
                    'details': f"Capped {outlier_count} outliers in column '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]",
                    'affected_rows': affected_indices,
                    'changes': transformations
                })
            
            # Keep outliers unchanged - no action needed
        
        # Apply row removal after processing all columns
        if rows_to_drop:
            before_count = len(self.df)
            self.df = self.df.drop(index=list(rows_to_drop))
            removed_count = before_count - len(self.df)
            results['rows_removed'] = removed_count
            
            if removed_count > 0:
                results['transformations'].append({
                    'operation': 'remove_rows',
                    'details': f"Removed {removed_count} rows with outliers",
                    'affected_rows': list(rows_to_drop)
                })
        
        # Add transformations to the main list
        self.transformations.extend(results['transformations'])
        
        return results
    
    def standardize_categorical_variables(self) -> Dict[str, Any]:
        """
        Standardize categorical variables in the dataset.
            
        Returns:
            Dictionary with results of standardization
        """
        results = {
            'columns_modified': [],
            'values_modified': 0,
            'transformations': []
        }
        
        # Skip if no categorical columns
        if not self.categorical_cols:
            return {
                'message': "No categorical columns found for standardization."
            }
        
        total_changes = 0
        
        for col in self.categorical_cols:
            # Skip columns with too many unique values
            if self.df[col].nunique() > 100:
                continue
                
            # Get non-null values
            non_null_mask = ~self.df[col].isna()
            
            # Skip if column is all null
            if not non_null_mask.any():
                continue
            
            # Convert to string and standardize
            original_values = self.df.loc[non_null_mask, col].copy()
            
            # Convert to string first to handle mixed types
            self.df.loc[non_null_mask, col] = self.df.loc[non_null_mask, col].astype(str)
            
            # Apply string standardization: title case and strip whitespace
            self.df.loc[non_null_mask, col] = self.df.loc[non_null_mask, col].str.strip().str.title()
            
            # Check if any values changed
            changed_mask = original_values.astype(str) != self.df.loc[non_null_mask, col]
            changes_count = changed_mask.sum()
            
            if changes_count > 0:
                total_changes += changes_count
                results['columns_modified'].append(col)
                
                # Record distinct value changes
                value_map = {}
                for old_val, new_val in zip(
                    original_values[changed_mask].unique(), 
                    self.df.loc[non_null_mask, col][changed_mask].unique()
                ):
                    value_map[str(old_val)] = str(new_val)
                
                # Record transformation
                results['transformations'].append({
                    'column': col,
                    'operation': 'standardize_categorical',
                    'details': f"Standardized {changes_count} values in column '{col}'",
                    'value_mapping': value_map
                })
        
        results['values_modified'] = total_changes
        
        # Add transformations to the main list
        self.transformations.extend(results['transformations'])
        
        return results
    
    def check_data_consistency(self, options: Dict[str, bool] = None) -> Dict[str, Any]:
        """
        Check data for consistency issues.
        
        Args:
            options: Dictionary with specific checks to perform
            
        Returns:
            Dictionary with data consistency issues
        """
        issues = []
        
        # Set default options if none provided
        if options is None:
            options = {
                'check_percentage_ranges': True,
                'check_date_order': True,
                'check_duplicates': True,
                'check_formats': False,
                'check_value_relationships': False,
                'check_distribution_anomalies': False
            }
        
        # Check percentage columns for values outside 0-100 range
        if options.get('check_percentage_ranges', True):
            for col in self.numeric_cols:
                if is_percentage_column(col, self.df[col]):
                    # Determine if it's 0-1 or 0-100 range
                    max_val = self.df[col].max()
                    if max_val <= 1:
                        # Check for values outside 0-1 range
                        invalid_mask = (self.df[col] < 0) | (self.df[col] > 1)
                        range_text = "0-1"
                    else:
                        # Check for values outside 0-100 range
                        invalid_mask = (self.df[col] < 0) | (self.df[col] > 100)
                        range_text = "0-100"
                    
                    invalid_count = invalid_mask.sum()
                    if invalid_count > 0:
                        # Get sample of invalid rows
                        sample_indices = self.df[invalid_mask].index.tolist()[:5]
                        sample_values = [
                            {"index": int(idx), "value": float(self.df.loc[idx, col])}
                            for idx in sample_indices
                        ]
                        
                        issues.append({
                            'type': 'percentage_range',
                            'column': col,
                            'details': f"Found {invalid_count} values outside {range_text} range in percentage column '{col}'",
                            'count': invalid_count,
                            'sample': sample_values
                        })
        
        # Check for duplicate IDs in ID columns
        if options.get('check_duplicates', True):
            for col in self.id_cols:
                dupe_counts = self.df[col].value_counts()
                duplicates = dupe_counts[dupe_counts > 1]
                
                if not duplicates.empty:
                    # Get samples of duplicated IDs
                    samples = []
                    for id_val, count in duplicates.head(5).items():
                        samples.append({
                            'value': str(id_val),
                            'count': int(count)
                        })
                    
                    issues.append({
                        'type': 'duplicate_id',
                        'column': col,
                        'details': f"Found {len(duplicates)} duplicated values in ID column '{col}'",
                        'count': len(duplicates),
                        'sample': samples
                    })
        
        # Check for date inconsistencies if we have multiple date columns
        if options.get('check_date_order', True) and len(self.datetime_cols) >= 2:
            # Check for chronological issues (e.g., end date before start date)
            for i, col1 in enumerate(self.datetime_cols[:-1]):
                for col2 in self.datetime_cols[i+1:]:
                    # Look for common pairs like start/end, from/to
                    if ('start' in col1.lower() and 'end' in col2.lower()) or \
                       ('from' in col1.lower() and 'to' in col2.lower()) or \
                       ('begin' in col1.lower() and 'finish' in col2.lower()):
                        
                        # Convert to datetime if they're not already
                        date1 = pd.to_datetime(self.df[col1], errors='coerce')
                        date2 = pd.to_datetime(self.df[col2], errors='coerce')
                        
                        # Check which rows have valid dates in both columns
                        valid_mask = ~date1.isna() & ~date2.isna()
                        
                        if valid_mask.sum() > 0:
                            # Check for date2 < date1
                            invalid_order = (date2 < date1)
                            invalid_count = (invalid_order & valid_mask).sum()
                            
                            if invalid_count > 0:
                                # Get sample of invalid rows
                                sample_indices = self.df[invalid_order & valid_mask].index.tolist()[:5]
                                sample_values = [
                                    {
                                        "index": int(idx),
                                        "start_date": str(date1.loc[idx]),
                                        "end_date": str(date2.loc[idx])
                                    }
                                    for idx in sample_indices
                                ]
                                
                                issues.append({
                                    'type': 'date_order',
                                    'columns': [col1, col2],
                                    'details': f"Found {invalid_count} rows where '{col2}' is earlier than '{col1}'",
                                    'count': invalid_count,
                                    'sample': sample_values
                                })
        
        # Check for format inconsistencies in text columns
        if options.get('check_formats', False):
            text_cols = [col for col in self.df.columns if self.column_types.get(col) in ['string', 'text']]
            
            for col in text_cols:
                # Skip columns with too many unique values
                if self.df[col].nunique() > 100:
                    continue
                
                # Check for mixed case formats
                values = self.df[col].dropna().astype(str)
                
                if len(values) > 0:
                    # Check for inconsistent capitalization
                    lower_count = values.str.islower().sum()
                    upper_count = values.str.isupper().sum()
                    title_count = values.str.istitle().sum()
                    
                    # If we have a mix of formats with significant counts
                    total = len(values)
                    
                    if (lower_count > 0 and upper_count > 0) or (lower_count > 0 and title_count > 0) or (upper_count > 0 and title_count > 0):
                        # Only flag if each format has at least 10% representation
                        lower_pct = lower_count / total * 100
                        upper_pct = upper_count / total * 100
                        title_pct = title_count / total * 100
                        
                        if ((lower_pct >= 10 and upper_pct >= 10) or 
                            (lower_pct >= 10 and title_pct >= 10) or 
                            (upper_pct >= 10 and title_pct >= 10)):
                            
                            # Get samples of each format
                            sample_values = []
                            
                            if lower_count > 0:
                                lower_sample = self.df[col][values.str.islower()].head(2).tolist()
                                for val in lower_sample:
                                    sample_values.append({"format": "lowercase", "value": str(val)})
                            
                            if upper_count > 0:
                                upper_sample = self.df[col][values.str.isupper()].head(2).tolist()
                                for val in upper_sample:
                                    sample_values.append({"format": "UPPERCASE", "value": str(val)})
                            
                            if title_count > 0:
                                title_sample = self.df[col][values.str.istitle()].head(2).tolist()
                                for val in title_sample:
                                    sample_values.append({"format": "Title Case", "value": str(val)})
                            
                            issues.append({
                                'type': 'format_inconsistency',
                                'column': col,
                                'details': f"Found inconsistent text formats in column '{col}'",
                                'count': total,
                                'stats': {
                                    'lowercase': int(lower_count),
                                    'uppercase': int(upper_count),
                                    'titlecase': int(title_count),
                                    'other': int(total - lower_count - upper_count - title_count)
                                },
                                'sample': sample_values
                            })
        
        # Check for value relationships
        if options.get('check_value_relationships', False):
            # This would check for logical relationships between columns
            # Example: age vs. income, education vs. salary, etc.
            # For now, we'll just provide a basic implementation
            
            numeric_cols = self.numeric_cols
            
            # Check pairs of numeric columns for potential issues
            for i, col1 in enumerate(numeric_cols[:-1]):
                for col2 in numeric_cols[i+1:]:
                    # Skip if either column has too many missing values
                    if (self.df[col1].isna().sum() / len(self.df) > 0.5 or 
                        self.df[col2].isna().sum() / len(self.df) > 0.5):
                        continue
                    
                    # Check correlation - flag potentially problematic relationships
                    corr = self.df[[col1, col2]].corr().iloc[0, 1]
                    
                    # Flag perfect or near-perfect correlations (might indicate duplicated information)
                    if abs(corr) > 0.98:
                        issues.append({
                            'type': 'high_correlation',
                            'columns': [col1, col2],
                            'details': f"Found near-perfect correlation ({corr:.2f}) between '{col1}' and '{col2}'",
                            'count': 1,
                            'correlation': float(corr)
                        })
                    
                    # Add other relationship checks as needed
        
        # Check for distribution anomalies
        if options.get('check_distribution_anomalies', False):
            for col in self.numeric_cols:
                # Skip columns with too many missing values
                if self.df[col].isna().sum() / len(self.df) > 0.5:
                    continue
                
                # Check for multimodal distributions which can indicate data quality issues
                try:
                    from scipy import stats
                    
                    data = self.df[col].dropna()
                    if len(data) < 100:  # Skip if not enough data
                        continue
                    
                    # Calculate kernel density estimate
                    kde = stats.gaussian_kde(data)
                    x = np.linspace(data.min(), data.max(), 1000)
                    y = kde(x)
                    
                    # Find peaks in the density curve
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(y)
                    
                    # If we find multiple significant peaks, flag as potential issue
                    if len(peaks) > 1:
                        # Filter to peaks that are at least 10% of the max peak
                        significant_peaks = [p for p in peaks if y[p] > 0.1 * y.max()]
                        
                        if len(significant_peaks) > 1:
                            peak_values = [float(x[p]) for p in significant_peaks]
                            
                            issues.append({
                                'type': 'multimodal_distribution',
                                'column': col,
                                'details': f"Found multiple peaks in distribution of '{col}', suggesting data from different sources or populations",
                                'count': len(significant_peaks),
                                'peaks': peak_values
                            })
                except:
                    # Skip on error
                    continue
        
        return {
            'issues_found': len(issues) > 0,
            'issue_count': len(issues),
            'issues': issues
        }
        
    def fix_data_consistency_issues(self, issues_to_fix: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fix identified data consistency issues.
        
        Args:
            issues_to_fix: List of issue dictionaries to fix, with issue type and fix strategy
            
        Returns:
            Dictionary with transformation results
        """
        if not issues_to_fix:
            return {'transformations': [], 'rows_modified': 0}
        
        transformations = []
        rows_modified = set()
        df_copy = self.df.copy()
        
        for issue in issues_to_fix:
            issue_type = issue.get('type')
            fix_strategy = issue.get('fix_strategy', 'default')
            
            if issue_type == 'percentage_range':
                col = issue['column']
                # Determine if it's 0-1 or 0-100 range
                max_val = df_copy[col].max()
                if max_val <= 1:
                    # Fix for 0-1 range
                    invalid_mask = (df_copy[col] < 0) | (df_copy[col] > 1)
                    range_text = "0-1"
                    
                    if fix_strategy == 'cap':
                        # Store affected rows and original values
                        affected_indices = df_copy[invalid_mask].index.tolist()
                        original_values = df_copy.loc[affected_indices, col].copy()
                        
                        # Cap values at boundaries
                        df_copy.loc[df_copy[col] < 0, col] = 0
                        df_copy.loc[df_copy[col] > 1, col] = 1
                        
                        # Record number of values fixed
                        issues_fixed = invalid_mask.sum()
                        rows_modified.update(affected_indices)
                        
                        transformations.append({
                            'type': 'percentage_range_fix',
                            'column': col,
                            'details': f"Capped {issues_fixed} values to be within {range_text} range",
                            'issues_fixed': issues_fixed,
                            'strategy': 'cap'
                        })
                    elif fix_strategy == 'rescale':
                        # Store affected rows
                        affected_indices = df_copy.index.tolist()
                        original_values = df_copy[col].copy()
                        
                        # Rescale the entire column to be between 0 and 1
                        min_val = df_copy[col].min()
                        max_val = df_copy[col].max()
                        
                        if min_val < max_val:  # Avoid division by zero
                            df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
                        
                        # Record transformation
                        issues_fixed = invalid_mask.sum()
                        rows_modified.update(affected_indices)
                        
                        transformations.append({
                            'type': 'percentage_range_fix',
                            'column': col,
                            'details': f"Rescaled entire column to be within {range_text} range",
                            'issues_fixed': issues_fixed,
                            'strategy': 'rescale'
                        })
                else:
                    # Fix for 0-100 range
                    invalid_mask = (df_copy[col] < 0) | (df_copy[col] > 100)
                    range_text = "0-100"
                    
                    if fix_strategy == 'cap':
                        # Store affected rows and original values
                        affected_indices = df_copy[invalid_mask].index.tolist()
                        original_values = df_copy.loc[affected_indices, col].copy()
                        
                        # Cap values at boundaries
                        df_copy.loc[df_copy[col] < 0, col] = 0
                        df_copy.loc[df_copy[col] > 100, col] = 100
                        
                        # Record number of values fixed
                        issues_fixed = invalid_mask.sum()
                        rows_modified.update(affected_indices)
                        
                        transformations.append({
                            'type': 'percentage_range_fix',
                            'column': col,
                            'details': f"Capped {issues_fixed} values to be within {range_text} range",
                            'issues_fixed': issues_fixed,
                            'strategy': 'cap'
                        })
                    elif fix_strategy == 'rescale':
                        # Store affected rows
                        affected_indices = df_copy.index.tolist()
                        original_values = df_copy[col].copy()
                        
                        # Rescale the entire column to be between 0 and 100
                        min_val = df_copy[col].min()
                        max_val = df_copy[col].max()
                        
                        if min_val < max_val:  # Avoid division by zero
                            df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val) * 100
                        
                        # Record transformation
                        issues_fixed = invalid_mask.sum()
                        rows_modified.update(affected_indices)
                        
                        transformations.append({
                            'type': 'percentage_range_fix',
                            'column': col,
                            'details': f"Rescaled entire column to be within {range_text} range",
                            'issues_fixed': issues_fixed,
                            'strategy': 'rescale'
                        })
            
            elif issue_type == 'date_order':
                col1, col2 = issue['columns']
                
                # Convert to datetime if they're not already
                date1 = pd.to_datetime(df_copy[col1], errors='coerce')
                date2 = pd.to_datetime(df_copy[col2], errors='coerce')
                
                # Check which rows have valid dates in both columns
                valid_mask = ~date1.isna() & ~date2.isna()
                
                # Check for date2 < date1
                invalid_order = (date2 < date1) & valid_mask
                
                if fix_strategy == 'swap':
                    # Store affected rows
                    affected_indices = df_copy[invalid_order].index.tolist()
                    
                    # Swap the dates
                    temp = df_copy.loc[affected_indices, col1].copy()
                    df_copy.loc[affected_indices, col1] = df_copy.loc[affected_indices, col2]
                    df_copy.loc[affected_indices, col2] = temp
                    
                    # Record transformation
                    issues_fixed = invalid_order.sum()
                    rows_modified.update(affected_indices)
                    
                    transformations.append({
                        'type': 'date_order_fix',
                        'columns': [col1, col2],
                        'details': f"Swapped values in {issues_fixed} rows to ensure '{col1}' is before '{col2}'",
                        'issues_fixed': issues_fixed,
                        'strategy': 'swap'
                    })
                elif fix_strategy == 'clear':
                    # Store affected rows
                    affected_indices = df_copy[invalid_order].index.tolist()
                    
                    # Clear the invalid entries
                    df_copy.loc[affected_indices, col1] = None
                    df_copy.loc[affected_indices, col2] = None
                    
                    # Record transformation
                    issues_fixed = invalid_order.sum()
                    rows_modified.update(affected_indices)
                    
                    transformations.append({
                        'type': 'date_order_fix',
                        'columns': [col1, col2],
                        'details': f"Cleared values in {issues_fixed} rows where '{col1}' is after '{col2}'",
                        'issues_fixed': issues_fixed,
                        'strategy': 'clear'
                    })
            
            elif issue_type == 'duplicate_id':
                col = issue['column']
                
                if fix_strategy == 'keep_first':
                    # Find duplicated values
                    is_duplicate = df_copy.duplicated(subset=[col], keep='first')
                    
                    # Store affected rows
                    affected_indices = df_copy[is_duplicate].index.tolist()
                    issues_fixed = len(affected_indices)
                    
                    if issues_fixed > 0:
                        # Keep first occurrence, mark the rest as null
                        df_copy.loc[affected_indices, col] = None
                        
                        # Record transformation
                        rows_modified.update(affected_indices)
                        
                        transformations.append({
                            'type': 'duplicate_id_fix',
                            'column': col,
                            'details': f"Cleared {issues_fixed} duplicate IDs in column '{col}', keeping the first occurrence",
                            'issues_fixed': issues_fixed,
                            'strategy': 'keep_first'
                        })
                
                elif fix_strategy == 'make_unique':
                    # Find values that occur more than once
                    value_counts = df_copy[col].value_counts()
                    duplicated_values = value_counts[value_counts > 1].index.tolist()
                    
                    issues_fixed = 0
                    
                    for value in duplicated_values:
                        # Find all rows with this value
                        value_mask = df_copy[col] == value
                        value_indices = df_copy[value_mask].index.tolist()
                        
                        # Keep the first occurrence, update the rest
                        for i, idx in enumerate(value_indices[1:], start=1):
                            if isinstance(value, str):
                                df_copy.loc[idx, col] = f"{value}_{i}"
                            else:
                                # Handle numeric values by using a string representation
                                df_copy.loc[idx, col] = f"{value}_{i}"
                            
                            issues_fixed += 1
                    
                    # Record transformation
                    if issues_fixed > 0:
                        rows_modified.update(df_copy[df_copy[col].isin(duplicated_values)].index.tolist())
                        
                        transformations.append({
                            'type': 'duplicate_id_fix',
                            'column': col,
                            'details': f"Made {issues_fixed} IDs unique in column '{col}' by appending suffixes",
                            'issues_fixed': issues_fixed,
                            'strategy': 'make_unique'
                        })
                        
            elif issue_type == 'format_inconsistency':
                col = issue['column']
                format_strategy = fix_strategy  # lowercase, uppercase, titlecase
                
                # Store the original values
                affected_indices = df_copy[~df_copy[col].isna()].index.tolist()
                original_values = df_copy.loc[affected_indices, col].copy()
                
                if format_strategy == 'lowercase':
                    # Convert all values to lowercase
                    df_copy.loc[affected_indices, col] = df_copy.loc[affected_indices, col].astype(str).str.lower()
                    
                    # Record transformation
                    issues_fixed = len(affected_indices)
                    rows_modified.update(affected_indices)
                    
                    transformations.append({
                        'type': 'format_inconsistency_fix',
                        'column': col,
                        'details': f"Standardized {issues_fixed} values to lowercase in column '{col}'",
                        'issues_fixed': issues_fixed,
                        'strategy': 'lowercase'
                    })
                
                elif format_strategy == 'uppercase':
                    # Convert all values to uppercase
                    df_copy.loc[affected_indices, col] = df_copy.loc[affected_indices, col].astype(str).str.upper()
                    
                    # Record transformation
                    issues_fixed = len(affected_indices)
                    rows_modified.update(affected_indices)
                    
                    transformations.append({
                        'type': 'format_inconsistency_fix',
                        'column': col,
                        'details': f"Standardized {issues_fixed} values to UPPERCASE in column '{col}'",
                        'issues_fixed': issues_fixed,
                        'strategy': 'uppercase'
                    })
                
                elif format_strategy == 'titlecase':
                    # Convert all values to title case
                    df_copy.loc[affected_indices, col] = df_copy.loc[affected_indices, col].astype(str).str.title()
                    
                    # Record transformation
                    issues_fixed = len(affected_indices)
                    rows_modified.update(affected_indices)
                    
                    transformations.append({
                        'type': 'format_inconsistency_fix',
                        'column': col,
                        'details': f"Standardized {issues_fixed} values to Title Case in column '{col}'",
                        'issues_fixed': issues_fixed,
                        'strategy': 'titlecase'
                    })
            
            elif issue_type == 'high_correlation':
                # For high correlation issues, we typically just report them
                # but don't automatically fix them as this requires domain knowledge
                # We could provide options like dropping one column, but we'll just log for now
                col1, col2 = issue['columns']
                
                if fix_strategy == 'drop_second':
                    # Drop the second column
                    df_copy = df_copy.drop(columns=[col2])
                    
                    # Record transformation
                    issues_fixed = 1
                    
                    transformations.append({
                        'type': 'high_correlation_fix',
                        'columns': [col1, col2],
                        'details': f"Dropped column '{col2}' due to high correlation with '{col1}'",
                        'issues_fixed': issues_fixed,
                        'strategy': 'drop_second'
                    })
                    
            elif issue_type == 'multimodal_distribution':
                # For multimodal distributions, we might want to identify clusters
                # and potentially split the data
                col = issue['column']
                
                if fix_strategy == 'standardize':
                    # Standardize the column (z-score normalization)
                    affected_indices = df_copy[~df_copy[col].isna()].index.tolist()
                    original_values = df_copy.loc[affected_indices, col].copy()
                    
                    from scipy import stats
                    df_copy.loc[affected_indices, col] = stats.zscore(df_copy.loc[affected_indices, col])
                    
                    # Record transformation
                    issues_fixed = len(affected_indices)
                    rows_modified.update(affected_indices)
                    
                    transformations.append({
                        'type': 'multimodal_distribution_fix',
                        'column': col,
                        'details': f"Standardized {issues_fixed} values in column '{col}' to address multimodal distribution",
                        'issues_fixed': issues_fixed,
                        'strategy': 'standardize'
                    })
                
                elif fix_strategy == 'flag_clusters':
                    # Add a new column that flags which cluster each value belongs to
                    from sklearn.cluster import KMeans
                    
                    data = df_copy[col].dropna().values.reshape(-1, 1)
                    
                    # Determine number of clusters (using the number of peaks detected)
                    num_clusters = issue.get('count', 2)
                    num_clusters = min(max(2, num_clusters), 5)  # Between 2 and 5 clusters
                    
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    clusters = kmeans.fit_predict(data)
                    
                    # Create mapping from index to cluster
                    cluster_mapping = dict(zip(df_copy[~df_copy[col].isna()].index, clusters))
                    
                    # Create a new column with the cluster labels
                    new_col_name = f"{col}_cluster"
                    df_copy[new_col_name] = None
                    
                    for idx, cluster in cluster_mapping.items():
                        df_copy.loc[idx, new_col_name] = int(cluster)
                    
                    # Record transformation
                    issues_fixed = len(cluster_mapping)
                    
                    transformations.append({
                        'type': 'multimodal_distribution_fix',
                        'column': col,
                        'details': f"Added column '{new_col_name}' with cluster labels for the {num_clusters} modes in '{col}'",
                        'issues_fixed': issues_fixed,
                        'strategy': 'flag_clusters',
                        'new_column': new_col_name
                    })
        
        # Assign the cleaned dataframe back to self.df
        self.df = df_copy
        
        return {
            'transformations': transformations,
            'rows_modified': len(rows_modified)
        }
    
    def get_transformations_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all transformations applied to the dataset.
        
        Returns:
            Dictionary with transformation summary
        """
        summary = {
            'total_transformations': len(self.transformations),
            'rows_before': len(self.original_df),
            'rows_after': len(self.df),
            'columns_modified': set(),
            'transformations': self.transformations,
            'transformation_counts': defaultdict(int)
        }
        
        # Count transformations by type
        for trans in self.transformations:
            operation = trans.get('operation', 'unknown')
            summary['transformation_counts'][operation] += 1
            
            # Collect affected columns
            if 'column' in trans:
                summary['columns_modified'].add(trans['column'])
            elif 'columns' in trans:
                summary['columns_modified'].update(trans['columns'])
        
        # Convert sets to lists for JSON serialization
        summary['columns_modified'] = list(summary['columns_modified'])
        
        return summary
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame.
        
        Returns:
            The cleaned pandas DataFrame
        """
        return self.df.copy()
    
    def get_original_data(self) -> pd.DataFrame:
        """
        Get the original DataFrame.
        
        Returns:
            The original pandas DataFrame
        """
        return self.original_df.copy()

def is_percentage_column(column_name: str, data: pd.Series) -> bool:
    """
    Check if a column contains percentage values.
    
    Args:
        column_name: Name of the column
        data: Data in the column
    
    Returns:
        True if the column appears to contain percentage values
    """
    # Check column name for percentage indicators
    percentage_indicators = [
        'percent', 'pct', 'percentage', 'ratio', 'rate', 'proportion'
    ]
    
    name_lower = column_name.lower()
    has_percentage_name = (
        any(indicator in name_lower for indicator in percentage_indicators) or
        '%' in column_name
    )
    
    # Check data characteristics
    if data.dtype in ['float64', 'float32', 'int64', 'int32']:
        # For 0-1 range percentages
        values = data.dropna()
        if len(values) == 0:
            return False
            
        max_val = values.max()
        min_val = values.min()
        
        # If all values are between 0 and 1, likely a proportion/percentage
        if 0 <= min_val and max_val <= 1:
            return True
            
        # If all values are between 0 and 100, likely a percentage
        if 0 <= min_val and max_val <= 100:
            # Additional check: most values should be numeric with limited decimals
            # and a significant portion of values should have decimals if we're dealing
            # with precise percentages
            if has_percentage_name:
                return True
            
            # Check if values cluster around whole numbers or have many decimal places
            decimal_count = sum(1 for x in values if x % 1 != 0)
            decimal_ratio = decimal_count / len(values) if len(values) > 0 else 0
            
            # If most values have decimals and max is around 100, likely a percentage
            if decimal_ratio > 0.3 and 90 <= max_val <= 105:
                return True
    
    # Return based on name if data characteristics don't give a clear signal
    return has_percentage_name
