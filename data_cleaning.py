import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        
    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Handle missing values in the dataset
        
        Args:
            df: The dataframe to process
            
        Returns:
            Tuple containing:
            - Processed dataframe with missing values handled
            - List of transformations applied
        """
        transformations = []
        df_copy = df.copy()
        
        # Check for missing values in each column
        missing_values = df_copy.isnull().sum()
        
        for column, count in missing_values.items():
            if count > 0:
                # Record the transformation
                transformations.append(f"Found {count} missing values in column '{column}'")
                
                # Handle missing values based on column type
                if df_copy[column].dtype in ['int64', 'float64']:
                    # For numerical columns, fill with median
                    median_value = df_copy[column].median()
                    df_copy[column] = df_copy[column].fillna(median_value)
                    transformations.append(f"Filled {count} missing values in '{column}' with median value {median_value:.2f}")
                else:
                    # For categorical columns, fill with mode
                    mode_value = df_copy[column].mode()[0]
                    df_copy[column] = df_copy[column].fillna(mode_value)
                    transformations.append(f"Filled {count} missing values in '{column}' with mode value '{mode_value}'")
        
        if not transformations:
            transformations.append("No missing values found in the dataset")
            
        return df_copy, transformations
    
    def standardize_categorical_variables(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Standardize categorical variables in the dataset
        
        Args:
            df: The dataframe to process
            
        Returns:
            Tuple containing:
            - Processed dataframe with standardized categorical variables
            - List of transformations applied
        """
        transformations = []
        df_copy = df.copy()
        
        # Gender standardization
        if 'Gender' in df_copy.columns:
            # Check for inconsistent gender values
            gender_values = df_copy['Gender'].unique()
            
            # Convert to title case and strip whitespace
            df_copy['Gender'] = df_copy['Gender'].str.strip().str.title()
            
            # Map to standard values
            gender_mapping = {
                'M': 'Male',
                'F': 'Female',
                'Male': 'Male',
                'Female': 'Female',
                'Other': 'Other'
            }
            
            df_copy['Gender'] = df_copy['Gender'].map(lambda x: gender_mapping.get(x, x))
            
            transformations.append(f"Standardized 'Gender' values: {', '.join(gender_values)} → {', '.join(df_copy['Gender'].unique())}")
        
        # Education_Level standardization
        if 'Education_Level' in df_copy.columns:
            # Check for inconsistent education level values
            edu_values = df_copy['Education_Level'].unique()
            
            # Convert to title case and strip whitespace
            df_copy['Education_Level'] = df_copy['Education_Level'].str.strip().str.title()
            
            # Map to standard values
            edu_mapping = {
                'Hs': 'High School',
                'High School': 'High School',
                'Highschool': 'High School',
                'Secondary': 'High School',
                'Ug': 'Undergraduate',
                'Undergrad': 'Undergraduate',
                'Undergraduate': 'Undergraduate',
                'Bachelor': 'Undergraduate',
                'Pg': 'Postgraduate',
                'Post Graduate': 'Postgraduate',
                'Postgrad': 'Postgraduate',
                'Postgraduate': 'Postgraduate',
                'Masters': 'Postgraduate',
                'Phd': 'Postgraduate',
                'Doctorate': 'Postgraduate'
            }
            
            # Apply mapping
            df_copy['Education_Level'] = df_copy['Education_Level'].map(lambda x: edu_mapping.get(x, x))
            
            transformations.append(f"Standardized 'Education_Level' values: {', '.join(edu_values)} → {', '.join(df_copy['Education_Level'].unique())}")
        
        # Engagement_Level standardization
        if 'Engagement_Level' in df_copy.columns:
            # Check for inconsistent engagement level values
            eng_values = df_copy['Engagement_Level'].unique()
            
            # Convert to title case and strip whitespace
            df_copy['Engagement_Level'] = df_copy['Engagement_Level'].str.strip().str.title()
            
            # Map to standard values
            engagement_mapping = {
                'L': 'Low',
                'Low': 'Low',
                'M': 'Medium',
                'Medium': 'Medium',
                'Moderate': 'Medium',
                'H': 'High',
                'High': 'High'
            }
            
            # Apply mapping
            df_copy['Engagement_Level'] = df_copy['Engagement_Level'].map(lambda x: engagement_mapping.get(x, x))
            
            transformations.append(f"Standardized 'Engagement_Level' values: {', '.join(eng_values)} → {', '.join(df_copy['Engagement_Level'].unique())}")
        
        # Learning_Style standardization
        if 'Learning_Style' in df_copy.columns:
            # Check for inconsistent learning style values
            style_values = df_copy['Learning_Style'].unique()
            
            # Convert to title case and strip whitespace
            df_copy['Learning_Style'] = df_copy['Learning_Style'].str.strip().str.title()
            
            # Map to standard values
            style_mapping = {
                'V': 'Visual',
                'Visual': 'Visual',
                'A': 'Auditory',
                'Auditory': 'Auditory',
                'R': 'Reading/Writing',
                'R/W': 'Reading/Writing',
                'Reading': 'Reading/Writing',
                'Reading/Writing': 'Reading/Writing',
                'K': 'Kinesthetic',
                'Kinesthetic': 'Kinesthetic'
            }
            
            # Apply mapping
            df_copy['Learning_Style'] = df_copy['Learning_Style'].map(lambda x: style_mapping.get(x, x))
            
            transformations.append(f"Standardized 'Learning_Style' values: {', '.join(style_values)} → {', '.join(df_copy['Learning_Style'].unique())}")
        
        # Dropout_Likelihood standardization
        if 'Dropout_Likelihood' in df_copy.columns:
            # Check for inconsistent dropout likelihood values
            dropout_values = df_copy['Dropout_Likelihood'].unique()
            
            # Standardize to Yes/No format
            df_copy['Dropout_Likelihood'] = df_copy['Dropout_Likelihood'].map(
                lambda x: 'Yes' if str(x).lower() in ['y', 'yes', 'true', '1'] else 'No'
            )
            
            transformations.append(f"Standardized 'Dropout_Likelihood' values: {', '.join([str(v) for v in dropout_values])} → {', '.join(df_copy['Dropout_Likelihood'].unique())}")
        
        if not transformations:
            transformations.append("No categorical variables needed standardization")
            
        return df_copy, transformations
    
    def handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Detect and handle outliers in numerical columns
        
        Args:
            df: The dataframe to process
            
        Returns:
            Tuple containing:
            - Processed dataframe with outliers handled
            - List of transformations applied
        """
        transformations = []
        df_copy = df.copy()
        
        # List of numerical columns to check for outliers
        numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Exclude certain columns from outlier detection
        exclude_cols = ['Student_ID']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        for col in numerical_cols:
            # Calculate Q1, Q3, and IQR
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outliers = df_copy[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                transformations.append(f"Found {outlier_count} outliers in column '{col}' (values outside range {lower_bound:.2f} - {upper_bound:.2f})")
                
                # Cap outliers at the bounds (winsorization)
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
                transformations.append(f"Capped outliers in '{col}' to range {lower_bound:.2f} - {upper_bound:.2f}")
        
        if not transformations:
            transformations.append("No outliers detected in the numerical columns")
            
        return df_copy, transformations
    
    def fix_age_education_discrepancies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Check for and fix age-education level discrepancies
        
        Args:
            df: The dataframe to process
            
        Returns:
            Tuple containing:
            - Processed dataframe with age-education discrepancies fixed
            - List of transformations applied
        """
        transformations = []
        df_copy = df.copy()
        
        if 'Age' in df_copy.columns and 'Education_Level' in df_copy.columns:
            # Define age thresholds for education levels
            min_age_thresholds = {
                'High School': 14,
                'Undergraduate': 17,
                'Postgraduate': 20
            }
            
            # Flag discrepancies
            discrepancies = pd.DataFrame()
            
            for edu_level, min_age in min_age_thresholds.items():
                # Find rows where age is below the minimum threshold for the education level
                invalid_rows = df_copy[(df_copy['Education_Level'] == edu_level) & (df_copy['Age'] < min_age)]
                
                if not invalid_rows.empty:
                    discrepancies = pd.concat([discrepancies, invalid_rows])
            
            discrepancy_count = len(discrepancies)
            
            if discrepancy_count > 0:
                transformations.append(f"Found {discrepancy_count} age-education level discrepancies")
                
                # Fix discrepancies by adjusting education level to match age
                for idx, row in discrepancies.iterrows():
                    age = row['Age']
                    
                    # Determine appropriate education level based on age
                    if age < min_age_thresholds['High School']:
                        # Too young for any level, assume data error - set to most common education level
                        new_edu_level = df_copy['Education_Level'].mode()[0]
                    elif age < min_age_thresholds['Undergraduate']:
                        new_edu_level = 'High School'
                    elif age < min_age_thresholds['Postgraduate']:
                        new_edu_level = 'Undergraduate'
                    else:
                        new_edu_level = row['Education_Level']
                    
                    # Update the education level
                    if new_edu_level != row['Education_Level']:
                        df_copy.at[idx, 'Education_Level'] = new_edu_level
                        transformations.append(f"Changed education level from '{row['Education_Level']}' to '{new_edu_level}' for student with Age={age}")
            else:
                transformations.append("No age-education level discrepancies found")
        else:
            transformations.append("Could not check age-education discrepancies - required columns not found")
        
        return df_copy, transformations
    
    def ensure_data_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform final consistency checks and fixes
        
        Args:
            df: The dataframe to process
            
        Returns:
            Tuple containing:
            - Processed dataframe with consistency issues fixed
            - List of transformations applied
        """
        transformations = []
        df_copy = df.copy()
        
        # Check for invalid numerical values
        numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Validate and fix percentage columns to be within 0-100
        percentage_cols = ['Quiz_Scores', 'Assignment_Completion_Rate', 'Final_Exam_Score']
        for col in percentage_cols:
            if col in df_copy.columns:
                invalid_count = len(df_copy[(df_copy[col] < 0) | (df_copy[col] > 100)])
                
                if invalid_count > 0:
                    transformations.append(f"Found {invalid_count} invalid percentage values in '{col}' (outside 0-100)")
                    
                    # Cap values to valid range
                    df_copy[col] = df_copy[col].clip(0, 100)
                    transformations.append(f"Capped '{col}' values to range 0-100")
        
        # Check for duplicated Student_ID
        if 'Student_ID' in df_copy.columns:
            duplicate_ids = df_copy['Student_ID'].duplicated()
            duplicate_count = duplicate_ids.sum()
            
            if duplicate_count > 0:
                transformations.append(f"Found {duplicate_count} duplicate Student_ID values")
                
                # Keep the first occurrence of each Student_ID
                df_copy = df_copy.drop_duplicates(subset=['Student_ID'], keep='first')
                transformations.append(f"Removed {duplicate_count} duplicate rows based on Student_ID")
        
        # Ensure consistency in related numeric features
        if 'Quiz_Attempts' in df_copy.columns and 'Quiz_Scores' in df_copy.columns:
            # Check if Quiz_Scores exists with 0 Quiz_Attempts
            invalid_quiz = df_copy[(df_copy['Quiz_Attempts'] == 0) & (df_copy['Quiz_Scores'] > 0)]
            invalid_quiz_count = len(invalid_quiz)
            
            if invalid_quiz_count > 0:
                transformations.append(f"Found {invalid_quiz_count} rows with Quiz_Scores > 0 but Quiz_Attempts = 0")
                
                # Set Quiz_Scores to 0 when Quiz_Attempts is 0
                df_copy.loc[(df_copy['Quiz_Attempts'] == 0) & (df_copy['Quiz_Scores'] > 0), 'Quiz_Scores'] = 0
                transformations.append(f"Set Quiz_Scores to 0 for {invalid_quiz_count} rows where Quiz_Attempts = 0")
        
        if not transformations:
            transformations.append("No consistency issues found in the dataset")
            
        return df_copy, transformations
