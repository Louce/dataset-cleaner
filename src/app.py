import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from data_cleaning import DataCleaner
from data_visualization import DataVisualizer
from utils import (
    validate_file, detect_csv_dialect, estimate_memory_usage,
    generate_secure_filename, sanitize_string
)

# Configure Streamlit page
st.set_page_config(
    page_title="Dataset Cleaner",   
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)   

# Initialize session state
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'cleaner' not in st.session_state:
    st.session_state.cleaner = None
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'file_info' not in st.session_state:
    st.session_state.file_info = {}
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "overview"
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'success_message' not in st.session_state:
    st.session_state.success_message = None
if 'missing_value_strategies' not in st.session_state:
    st.session_state.missing_value_strategies = {}
if 'outlier_strategies' not in st.session_state:
    st.session_state.outlier_strategies = {}
if 'progress' not in st.session_state:
    st.session_state.progress = {"message": "", "value": 0}
if 'changes_tracking' not in st.session_state:
    st.session_state.changes_tracking = {
        'missing_values': [],
        'outliers': [],
        'data_consistency': []
    }

# Function to update progress
def update_progress(message, value=None):
    st.session_state.progress = {"message": message, "value": value}

# Application header
st.title("Dataset Cleaner 🧹")
st.markdown("""
This application allows you to upload CSV or Excel files, identify data quality issues, 
clean your dataset, and export the cleaned result.
""")

# Display any error or success messages
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    st.session_state.error_message = None

if st.session_state.success_message:
    st.success(st.session_state.success_message)
    st.session_state.success_message = None

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Data Import")
    
    uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx", "xls"], help="Upload a CSV or Excel file to clean")
    
    if uploaded_file is not None:
        try:
            # First, validate the file
            file_buffer = io.BytesIO(uploaded_file.getvalue())
            is_valid, message = validate_file(file_buffer)
            
            if not is_valid:
                st.session_state.error_message = message
                st.rerun()
            
            # Detect CSV dialect (delimiter, etc.)
            dialect = detect_csv_dialect(file_buffer)
            
            # Show advanced import settings
            with st.expander("Advanced Import Settings"):
                # Determine if file is Excel
                is_excel = uploaded_file.name.endswith(('.xlsx', '.xls'))
                
                if is_excel:
                    # Excel-specific settings
                    excel_file = pd.ExcelFile(file_buffer)
                    sheet_names = excel_file.sheet_names
                    selected_sheet = st.selectbox("Select Sheet", options=sheet_names, key="sheet_selector_settings")
                    skiprows = st.number_input("Skip Rows", min_value=0, value=0)
                    excel_header_row = st.number_input("Header Row", min_value=0, value=0, 
                                                      help="Row number to use as column names (0-based)")
                    excel_use_columns = st.text_input("Use Columns (optional)", 
                                                     help="Comma-separated list of column letters or indices to import (e.g., 'A,C:F')")
                    convert_formatted = st.checkbox("Convert formatted values", value=True,
                                                   help="Convert formatted Excel values to their display format")
                    
                    # Option to view raw data sample
                    if st.button("View Data Sample"):
                        sample_df = pd.read_excel(file_buffer, sheet_name=selected_sheet, nrows=5)
                        st.dataframe(sample_df)
                else:
                    # Existing CSV settings
                    delimiter = st.text_input("Delimiter", dialect['delimiter'])
                    quotechar = st.text_input("Quote Character", dialect['quotechar'])
                    encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "iso-8859-1", "cp1252"], index=0)
                    skiprows = st.number_input("Skip Rows", min_value=0, value=0)
                    
                    # Option to view raw data sample
                    if st.button("View Raw Data Sample"):
                        file_buffer.seek(0)
                        raw_sample = file_buffer.read(5000).decode(encoding)
                        st.text_area("Raw Data Sample", raw_sample, height=200)
            
            # Import the file
            if st.button("Import Data"):
                with st.spinner("Importing data..."):
                    try:
                        # Check if file is Excel or CSV
                        if uploaded_file.name.endswith(('.xlsx', '.xls')):
                            # Handle Excel file
                            # Read Excel with settings
                            if convert_formatted:
                                import openpyxl
                                wb = openpyxl.load_workbook(file_buffer, data_only=True)
                                file_buffer.seek(0)
                            df = pd.read_excel(
                                file_buffer,
                                sheet_name=selected_sheet,
                                skiprows=skiprows,
                                header=excel_header_row,
                                # Parse use_columns if provided
                                usecols=excel_use_columns.split(',') if excel_use_columns else None
                            )
                        else:
                            # Read CSV with the settings
                            file_buffer.seek(0)
                            df = pd.read_csv(
                                file_buffer,
                                delimiter=delimiter,
                                quotechar=quotechar,
                                encoding=encoding,
                                skiprows=skiprows,
                                on_bad_lines='warn'
                            )
                        
                        # Check if dataframe is empty or too large
                        if df.empty:
                            st.session_state.error_message = "The imported data is empty"
                            st.rerun()
                            
                        memory_usage = estimate_memory_usage(df)
                        if memory_usage > 500:  # Warn if over 500MB
                            st.warning(f"Warning: Dataset is large ({memory_usage:.2f} MB). Processing may be slow.")
                            
                        # Store original data and initialize cleaner
                        st.session_state.original_df = df
                        st.session_state.cleaner = DataCleaner(df)
                        st.session_state.cleaner.set_progress_callback(update_progress)
                        
                        # Store file info
                        st.session_state.file_info = {
                            'filename': uploaded_file.name,
                            'filesize': len(file_buffer.getvalue()) / (1024 * 1024),  # Size in MB
                            'rows': len(df),
                            'columns': len(df.columns),
                            'memory_usage': memory_usage,
                            'import_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Add import event to history
                        st.session_state.processing_history.append({
                            'action': 'import',
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'details': f"Imported {uploaded_file.name} ({len(df)} rows, {len(df.columns)} columns)"
                        })
                        
                        st.session_state.success_message = f"Successfully imported {uploaded_file.name}"
                        st.session_state.current_tab = "overview"
                        st.rerun()
                        
                    except Exception as e:
                        st.session_state.error_message = f"Error importing data: {str(e)}"
                        st.rerun()
        
        except Exception as e:
            st.session_state.error_message = f"Error processing upload: {str(e)}"
            st.rerun()
    
    # Show processing history
    if st.session_state.processing_history:
        with st.expander("Processing History"):
            for i, event in enumerate(reversed(st.session_state.processing_history)):
                st.write(f"**{event['timestamp']}**: {event['details']}")
                if i < len(st.session_state.processing_history) - 1:
                    st.write("---")
    
    # Reset application
    if st.button("Reset Application"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content area - Only show if data is loaded
if st.session_state.original_df is not None:
    # Navigation tabs
    tabs = ["Overview", "Missing Values", "Outliers", "Data Consistency", "Changes Visualization", "Cleaning Report", "Export"]
    tab_icons = ["📊", "❓", "📈", "🔄", "📉", "📝", "💾"]
    
    # Use columns to create a custom tab interface with icons
    cols = st.columns(len(tabs))
    for i, (col, tab, icon) in enumerate(zip(cols, tabs, tab_icons)):
        tab_id = tab.lower().replace(" ", "_")
        if col.button(f"{icon} {tab}", key=f"tab_{tab_id}"):
            st.session_state.current_tab = tab_id
    
    # Show progress bar if processing
    if st.session_state.progress["value"] is not None:
        st.progress(st.session_state.progress["value"] / 100)
        st.info(st.session_state.progress["message"])
    
    # Display content based on current tab
    if st.session_state.current_tab == "overview":
        st.header("Dataset Overview")
        
        # Display file information
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{st.session_state.file_info['rows']:,}")
        col2.metric("Columns", st.session_state.file_info['columns'])
        col3.metric("Size", f"{st.session_state.file_info['filesize']:.2f} MB")
        
        # Display basic dataset summary
        st.subheader("Data Preview")
        st.dataframe(st.session_state.original_df.head(10))
        
        with st.expander("Column Information"):
            # Get column types and info
            df_types = pd.DataFrame({
                'Column': st.session_state.original_df.columns,
                'Type': st.session_state.original_df.dtypes,
                'Non-Null Count': st.session_state.original_df.count(),
                'Null Count': st.session_state.original_df.isna().sum(),
                'Unique Values': [st.session_state.original_df[col].nunique() for col in st.session_state.original_df.columns],
            })
            st.dataframe(df_types)
            
        with st.expander("Numeric Columns Summary"):
            # Summary statistics for numeric columns
            numeric_cols = st.session_state.original_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(st.session_state.original_df[numeric_cols].describe())
            else:
                st.info("No numeric columns found in the dataset")
        
        # Data quality overview
        st.subheader("Data Quality Overview")
    
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate missing values percentage
            missing_values = st.session_state.original_df.isna().sum()
            missing_percentage = (missing_values / len(st.session_state.original_df)) * 100
            missing_df = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Count': missing_values.values,
                'Missing Percentage': missing_percentage.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
            
            if not missing_df.empty:
                st.write("Columns with Missing Values:")
                st.dataframe(missing_df)
                
                # Plot missing values
                if st.button("Show Missing Values Plot"):
                    if len(missing_df) > 0:
                        fig = px.bar(
                            missing_df, 
                            x='Column', 
                            y='Missing Percentage',
                            title='Missing Values by Column',
                            color='Missing Percentage',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No missing values found in the dataset! 🎉")
        
        with col2:
            # Detect potential duplicates
            duplicate_count = st.session_state.original_df.duplicated().sum()
            
            st.write("Potential Data Issues:")
            issues_container = st.container()
            
            issues_found = False
            
            if duplicate_count > 0:
                issues_found = True
                issues_container.warning(f"⚠️ Found {duplicate_count} duplicate rows")
            
            # Check for outliers in numeric columns
            if hasattr(st.session_state.cleaner, 'identify_outliers'):
                outlier_info = st.session_state.cleaner.identify_outliers()
                
                # Get columns with outliers if they exist
                if isinstance(outlier_info, dict) and outlier_info.get('has_outliers', False):
                    outlier_columns = list(outlier_info.get('columns', {}).keys())
                    
                    if outlier_columns:
                        issues_found = True
                        issues_container.warning(f"⚠️ Found outliers in {len(outlier_columns)} columns")
            
            if not issues_found:
                issues_container.success("No major data issues detected! 🎉")
        
        # Actions for this tab
        st.subheader("Available Actions")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("Analyze Missing Values"):
                st.session_state.current_tab = "missing_values"
                st.rerun()
        
        with action_col2:
            if st.button("Detect Outliers"):
                st.session_state.current_tab = "outliers"
                st.rerun()
        
        with action_col3:
            if st.button("Check Data Consistency"):
                st.session_state.current_tab = "data_consistency"
                st.rerun()
                
    elif st.session_state.current_tab == "missing_values":
        st.header("Missing Values")
        
        # Identify missing values first
        if not hasattr(st.session_state, 'missing_info') or st.button("Refresh Missing Values Analysis"):
            with st.spinner("Analyzing missing values..."):
                st.session_state.cleaner.df = st.session_state.original_df.copy()
                missing_info = st.session_state.cleaner.identify_missing_values()
                st.session_state.missing_info = missing_info
        
        # Display summary
        if st.session_state.missing_info and isinstance(st.session_state.missing_info, dict):
            has_missing = st.session_state.missing_info.get('has_missing', False)
            total_missing_cells = st.session_state.missing_info.get('total_missing_cells', 0)
            total_rows = st.session_state.missing_info.get('total_rows', len(st.session_state.original_df))
            
            if total_rows > 0:
                total_missing_pct = (total_missing_cells / total_rows) * 100
            else:
                total_missing_pct = 0
                
            st.metric("Total Missing Values", f"{total_missing_cells:,} ({total_missing_pct:.2f}%)")
            
            # Display detailed missing value information
            missing_columns_data = {}
            if st.session_state.missing_info and isinstance(st.session_state.missing_info, dict):
                missing_columns_data = st.session_state.missing_info.get('columns', {})
            
            if missing_columns_data:
                st.subheader("Columns with Missing Values")
                
                # Create a dataframe for easier visualization
                missing_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Missing Count': info['count'],
                        'Missing %': info['percent'],
                        'Data Type': info['type']
                    } for col, info in missing_columns_data.items()
                ])
                
                st.dataframe(missing_df)
                
                # Interactive editor for missing values
                st.subheader("Edit Missing Values")
                
                selected_column = st.selectbox(
                    "Select Column to Edit", 
                    options=list(missing_columns_data.keys()),
                    format_func=lambda x: f"{x} ({missing_columns_data[x]['count']} missing)"
                )
                
                # Get the selected column's info
                selected_info = missing_columns_data.get(selected_column)
                
                if selected_info:
                    # Show column stats
                    st.write(f"**Column Type:** {selected_info['type']}")
                    st.write(f"**Missing Values:** {selected_info['count']} ({selected_info['percent']:.2f}%)")
                    
                    # Display sample of unique values
                    if 'sample_rows' in selected_info and selected_info['sample_rows']:
                        with st.expander("View Sample Rows with Missing Values"):
                            for i, row in enumerate(selected_info['sample_rows']):
                                st.write(f"**Row {i+1}** (Index: {row['index']})")
                                st.json(row['data'])
                    
                    # Display fill strategies
                    st.write("**Fill Strategies:**")
                    
                    fill_strategies = selected_info.get('strategies', [])
                    
                    if fill_strategies:
                        # Initialize the strategy dict for this column if it doesn't exist
                        if selected_column not in st.session_state.missing_value_strategies:
                            st.session_state.missing_value_strategies[selected_column] = {}
                        
                        # Select a fill strategy
                        strategy_options = {s['name']: s['description'] for s in fill_strategies}
                        
                        selected_strategy = st.radio(
                            "Select Fill Strategy",
                            options=list(strategy_options.keys()),
                            format_func=lambda x: f"{x} - {strategy_options[x]}"
                        )
                        
                        # Store the selected strategy
                        st.session_state.missing_value_strategies[selected_column] = {
                            'strategy': selected_strategy
                        }
                        
                        # Configure the selected strategy
                        if selected_strategy == 'synergy':
                            st.write("**Smart Fill Strategy:**")
                            st.info("""
                            The synergy strategy dynamically determines the best way to fill missing values by:
                            
                            1. For numeric columns: Uses KNN imputation to predict values based on relationships with other columns
                            2. For categorical columns: Uses conditional mode based on similar rows
                            3. Falls back to statistical methods when necessary
                            
                            This approach often produces more accurate results than simple statistical approaches.
                            """)
                            
                        elif selected_strategy == 'custom':
                            # For custom, get the value to fill
                            st.write("**Enter a custom value to fill missing values:**")
                            
                            if selected_info['type'] in ['numeric', 'integer']:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    custom_value = st.number_input(
                                        "Custom numeric value", 
                                        value=0.0, 
                                        step=0.1,
                                        help="Enter the numeric value that will replace all missing values in this column"
                                    )
                                with col2:
                                    if st.button("Use column average"):
                                        if not pd.isna(st.session_state.original_df[selected_column].mean()):
                                            custom_value = round(st.session_state.original_df[selected_column].mean(), 2)
                                            st.write(f"Using average: {custom_value}")
                            else:
                                custom_value = st.text_input(
                                    "Custom text value", 
                                    value="Unknown",
                                    help="Enter the text that will replace all missing values in this column"
                                )
                                # Add common options
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("Use 'Unknown'"):
                                        custom_value = "Unknown"
                                with col2:
                                    if st.button("Use 'N/A'"):
                                        custom_value = "N/A"
                                with col3:
                                    if st.button("Use empty string"):
                                        custom_value = ""
                            
                            if custom_value is not None:
                                st.session_state.missing_value_strategies[selected_column]['value'] = custom_value
                                st.info(f"All missing values in '{selected_column}' will be replaced with: **{custom_value}**")
                        elif selected_strategy in ['mean', 'median', 'mode', 'zero', 'unknown', 'empty_string']:
                            # These strategies are handled internally by the cleaner
                            pass
                        
                        # Apply the strategy to this column
                        if st.button("Apply Strategy to Column"):
                            with st.spinner(f"Applying {selected_strategy} to {selected_column}..."):
                                # Create a specific strategy dict for this column
                                column_strategy = {selected_column: st.session_state.missing_value_strategies[selected_column]}
                                
                                # Apply the strategy
                                df_copy = st.session_state.original_df.copy()
                                st.session_state.cleaner.df = df_copy
                                result = st.session_state.cleaner.fill_missing_values(column_strategy)
                                filled_df = st.session_state.cleaner.get_cleaned_data()
                                
                                # Record the changes for visualization
                                missing_before = st.session_state.original_df[selected_column].isna().sum()
                                missing_after = filled_df[selected_column].isna().sum()
                                
                                # Add to changes tracking
                                st.session_state.changes_tracking['missing_values'].append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'column': selected_column,
                                    'strategy': selected_strategy,
                                    'before': missing_before,
                                    'after': missing_after,
                                    'difference': missing_before - missing_after
                                })
                                
                                # Update the original_df in session_state
                                st.session_state.original_df = filled_df
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    'action': 'fill_missing',
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'details': f"Filled missing values in {selected_column} using {selected_strategy}"
                                })
                                
                                # Show success message and refresh
                                st.session_state.success_message = f"Successfully filled missing values in {selected_column}"
                                st.rerun()
                    
                    # Apply strategies to all columns at once
                    if st.button("Apply Strategies to All Columns", key="apply_all_missing"):
                        if st.session_state.missing_value_strategies:
                            with st.spinner("Filling missing values in all configured columns..."):
                                # Apply all configured strategies
                                df_copy = st.session_state.original_df.copy()
                                st.session_state.cleaner.df = df_copy
                                result = st.session_state.cleaner.fill_missing_values(st.session_state.missing_value_strategies)
                                filled_df = st.session_state.cleaner.get_cleaned_data()
                                
                                # Record the changes for visualization
                                for column, strategy in st.session_state.missing_value_strategies.items():
                                    missing_before = st.session_state.original_df[column].isna().sum()
                                    missing_after = filled_df[column].isna().sum()
                                    
                                    # Add to changes tracking
                                    st.session_state.changes_tracking['missing_values'].append({
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'column': column,
                                        'strategy': strategy['strategy'],
                                        'before': missing_before,
                                        'after': missing_after,
                                        'difference': missing_before - missing_after
                                    })
                                
                                # Update the original_df in session_state
                                st.session_state.original_df = filled_df
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    'action': 'fill_missing_all',
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'details': f"Filled missing values in {len(st.session_state.missing_value_strategies)} columns"
                                })
                                
                                # Show success message and refresh
                                st.session_state.success_message = "Successfully filled missing values in all configured columns"
                                st.rerun()
        else:
            st.success("No missing values found in the dataset! 🎉")
    
    elif st.session_state.current_tab == "outliers":
        st.header("Outliers Detection and Handling")
        
        # Detect outliers
        st.write("Detecting outliers helps identify unusual values that may represent errors or special cases.")
        
        # Select outlier detection method
        detection_method = st.radio(
            "Select outlier detection method",
            options=["IQR (Interquartile Range)", "Z-Score"],
            index=0,
            help="IQR is robust and works well for most datasets. Z-Score is better for normally distributed data."
        )
        
        method_map = {
            "IQR (Interquartile Range)": "iqr",
            "Z-Score": "zscore"
        }
        
        # Button to detect outliers
        if st.button("Detect Outliers") or st.button("Refresh Outliers Analysis") or not hasattr(st.session_state, 'outlier_info'):
            with st.spinner(f"Detecting outliers using {detection_method}..."):
                st.session_state.cleaner.df = st.session_state.original_df.copy()
                outlier_info = st.session_state.cleaner.identify_outliers(
                    method=method_map[detection_method]
                )
                st.session_state.outlier_info = outlier_info
                st.session_state.outlier_method = method_map[detection_method]
        
        # Display outlier information
        if st.session_state.outlier_info and isinstance(st.session_state.outlier_info, dict):
            has_outliers = st.session_state.outlier_info.get('has_outliers', False)
            outlier_columns_data = st.session_state.outlier_info.get('columns', {})
            total_outliers = st.session_state.outlier_info.get('total_outliers', 0)
            
            if has_outliers and outlier_columns_data:
                st.subheader("Columns with Outliers")
                
                # Create a dataframe for easier visualization
                outlier_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Outlier Count': info['stats']['count'],
                        'Outlier %': info['stats']['percent'],
                        'Method': info['stats']['method']
                    } for col, info in outlier_columns_data.items()
                ])
                
                st.dataframe(outlier_df)
                
                # Interactive outlier handling
                st.subheader("Handle Outliers")
                
                selected_column = st.selectbox(
                    "Select Column to Handle", 
                    options=list(outlier_columns_data.keys()),
                    format_func=lambda x: f"{x} ({outlier_columns_data[x]['stats']['count']} outliers)"
                )
                
                # Get the selected column's info
                selected_info = outlier_columns_data.get(selected_column)
                
                if selected_info:
                    # Show column stats
                    st.write(f"**Outlier Count:** {selected_info['stats']['count']} ({selected_info['stats']['percent']:.2f}%)")
                    
                    if 'stats' in selected_info:
                        stats = selected_info['stats']
                        mean_value = stats.get('mean', 'N/A')
                        median_value = stats.get('median', 'N/A')
                        std_value = stats.get('std', 'N/A')
                        
                        st.write(f"**Mean:** {f'{mean_value:.2f}' if isinstance(mean_value, (int, float)) else mean_value}")
                        st.write(f"**Median:** {f'{median_value:.2f}' if isinstance(median_value, (int, float)) else median_value}")
                        st.write(f"**Std Dev:** {f'{std_value:.2f}' if isinstance(std_value, (int, float)) else std_value}")
                    
                    # Show bounds for IQR method
                    if selected_info['stats']['method'] == 'iqr':
                        lower_value = selected_info['stats'].get('lower_bound', 'N/A')
                        upper_value = selected_info['stats'].get('upper_bound', 'N/A')
                        
                        st.write(f"**Lower Bound:** {f'{lower_value:.2f}' if isinstance(lower_value, (int, float)) else lower_value}")
                        st.write(f"**Upper Bound:** {f'{upper_value:.2f}' if isinstance(upper_value, (int, float)) else upper_value}")
                    
                    # Display the rows with outliers
                    with st.expander("View Sample Rows with Outliers"):
                        sample_rows = selected_info.get('sample_rows', [])
                        if sample_rows:
                            for i, row in enumerate(sample_rows):
                                st.write(f"**Row {i+1}** (Index: {row['index']}), Value: {row['value']}")
                                st.json(row['data'])
                        else:
                            st.write("No sample rows available")
                    
                    # Visualize outliers
                    if st.button("Visualize Outliers"):
                        col_data = st.session_state.original_df[selected_column].dropna()
                        
                        # Create a box plot
                        fig1 = px.box(
                            col_data, 
                            title=f"Box Plot of {selected_column}",
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Create a histogram
                        fig2 = px.histogram(
                            col_data,
                            title=f"Distribution of {selected_column}",
                            marginal="rug"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Handle outliers
                    st.write("**Outlier Handling Options:**")
                    
                    # Get recommendations
                    treatments = selected_info.get('treatments', [])
                    
                    if treatments:
                        # Initialize the strategy dict for this column if it doesn't exist
                        if selected_column not in st.session_state.outlier_strategies:
                            st.session_state.outlier_strategies[selected_column] = {
                                'strategy': 'keep'
                            }
                        
                        # Select a handling strategy
                        strategy_options = {r['name']: r['description'] for r in treatments}
                        
                        selected_strategy = st.radio(
                            "Select Handling Strategy",
                            options=list(strategy_options.keys()),
                            format_func=lambda x: f"{x} - {strategy_options[x]}"
                        )
                        
                        # Store the selected strategy
                        st.session_state.outlier_strategies[selected_column] = {
                            'strategy': selected_strategy,
                            'method': selected_info['stats']['method']
                        }
                        
                        # Configure the selected strategy
                        if selected_strategy == 'cap':
                            # For capping, allow customizing bounds
                            if selected_info['stats']['method'] == 'iqr':
                                default_lower = selected_info['stats'].get('lower_bound', 0)
                                default_upper = selected_info['stats'].get('upper_bound', 100)
                                
                                custom_bounds = st.checkbox("Customize bounds")
                                
                                if custom_bounds:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        lower_bound = st.number_input("Lower bound", value=float(default_lower))
                                    with col2:
                                        upper_bound = st.number_input("Upper bound", value=float(default_upper))
                                    
                                    st.session_state.outlier_strategies[selected_column]['lower_bound'] = lower_bound
                                    st.session_state.outlier_strategies[selected_column]['upper_bound'] = upper_bound
                        
                        elif selected_strategy == 'transform':
                            # For transformation, select the method
                            transform_type = st.selectbox(
                                "Transformation Method",
                                options=["log", "sqrt"],
                                help="Log transformation works well for right-skewed data. Square root is more moderate."
                            )
                            
                            st.session_state.outlier_strategies[selected_column]['transform_type'] = transform_type
                        
                        # Apply the strategy to this column
                        if st.button("Apply Strategy to Column"):
                            with st.spinner(f"Applying {selected_strategy} to {selected_column}..."):
                                # Create a specific strategy dict for this column
                                column_strategy = {selected_column: st.session_state.outlier_strategies[selected_column]}
                                
                                # Apply the strategy
                                df_copy = st.session_state.original_df.copy()
                                st.session_state.cleaner.df = df_copy
                                handled_results = st.session_state.cleaner.handle_outliers(column_strategy)
                                handled_df = st.session_state.cleaner.get_cleaned_data()
                                
                                # Record the changes for visualization
                                # Count outliers before
                                outlier_info = selected_info['stats']
                                outliers_before = outlier_info['count']
                                
                                # Count outliers after
                                st.session_state.cleaner.df = handled_df
                                new_outlier_info = st.session_state.cleaner.identify_outliers(
                                    method=selected_info['stats']['method']
                                )
                                
                                # Find the column in the new results
                                outliers_after = 0
                                if isinstance(new_outlier_info, dict) and new_outlier_info.get('has_outliers', False):
                                    new_column_info = new_outlier_info.get('columns', {}).get(selected_column, {})
                                    if new_column_info:
                                        outliers_after = new_column_info['stats'].get('count', 0)
                                
                                # Add to changes tracking
                                st.session_state.changes_tracking['outliers'].append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'column': selected_column,
                                    'strategy': selected_strategy,
                                    'before': outliers_before,
                                    'after': outliers_after,
                                    'difference': outliers_before - outliers_after
                                })
                                
                                # Update the original_df in session_state
                                st.session_state.original_df = handled_df
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    'action': 'handle_outliers',
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'details': f"Handled outliers in {selected_column} using {selected_strategy}"
                                })
                                
                                # Show success message and refresh
                                st.session_state.success_message = f"Successfully handled outliers in {selected_column}"
                                st.rerun()
                
                # Apply strategies to all columns at once
                if st.button("Apply Strategies to All Columns", key="apply_all_outliers"):
                    if st.session_state.outlier_strategies:
                        with st.spinner("Handling outliers in all configured columns..."):
                            # Apply all configured strategies
                            df_copy = st.session_state.original_df.copy()
                            st.session_state.cleaner.df = df_copy
                            handled_results = st.session_state.cleaner.handle_outliers(st.session_state.outlier_strategies)
                            handled_df = st.session_state.cleaner.get_cleaned_data()
                            
                            # Record changes for each column for visualization
                            for column, strategy in st.session_state.outlier_strategies.items():
                                # Get outliers count before
                                st.session_state.cleaner.df = df_copy
                                before_outlier_info = st.session_state.cleaner.identify_outliers(method=strategy['method'])
                                outliers_before = 0
                                if isinstance(before_outlier_info, dict) and before_outlier_info.get('has_outliers', False):
                                    before_column_info = before_outlier_info.get('columns', {}).get(column, {})
                                    if before_column_info:
                                        outliers_before = before_column_info['stats'].get('count', 0)
                                
                                # Get outliers count after
                                st.session_state.cleaner.df = handled_df
                                after_outlier_info = st.session_state.cleaner.identify_outliers(method=strategy['method'])
                                outliers_after = 0
                                if isinstance(after_outlier_info, dict) and after_outlier_info.get('has_outliers', False):
                                    after_column_info = after_outlier_info.get('columns', {}).get(column, {})
                                    if after_column_info:
                                        outliers_after = after_column_info['stats'].get('count', 0)
                                
                                # Add to changes tracking
                                st.session_state.changes_tracking['outliers'].append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'column': column,
                                    'strategy': strategy['strategy'],
                                    'before': outliers_before,
                                    'after': outliers_after,
                                    'difference': outliers_before - outliers_after
                                })
                            
                            # Update the original_df in session_state
                            st.session_state.original_df = handled_df
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                'action': 'handle_outliers_all',
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'details': f"Handled outliers in {len(st.session_state.outlier_strategies)} columns"
                            })
                            
                            # Show success message and refresh
                            st.session_state.success_message = "Successfully handled outliers in all configured columns"
                            st.rerun()
            else:
                message = st.session_state.outlier_info.get('message', "No outliers detected using the selected method")
                st.info(message)
        else:
            st.info("Please detect outliers first")
    
    elif st.session_state.current_tab == "data_consistency":
        st.header("Data Consistency Checks")
        st.write("Identify and fix logical inconsistencies in your dataset.")
        
        if "original_df" not in st.session_state or st.session_state.original_df is None:
            st.warning("Please upload a dataset first.")
        else:
            # Display summary metrics if fixes have been applied
            if "consistency_fixes_applied" in st.session_state:
                st.subheader("Data Quality Improvement")
                
                fixes = st.session_state.consistency_fixes_applied
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Issues Fixed", fixes.get("issues_fixed", 0))
                
                with col2:
                    st.metric("Rows Modified", fixes.get("rows_modified", 0))
                
                with col3:
                    # Calculate progress
                    original_issues = fixes.get("original_issue_count", 0)
                    remaining_issues = fixes.get("remaining_issue_count", 0)
                    
                    if original_issues > 0:
                        improvement = ((original_issues - remaining_issues) / original_issues) * 100
                        st.metric("Data Quality Improvement", f"{improvement:.1f}%")
            
            # Quick scan options
            st.subheader("Quick Scan Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                check_pct = st.checkbox("Percentage Ranges", value=True, 
                                    help="Check if percentage columns have values outside expected ranges (0-100% or 0-1)")
                check_date = st.checkbox("Date Order Issues", value=True,
                                        help="Check if date columns have logical ordering issues (e.g., end date before start date)")
                check_dupes = st.checkbox("Duplicate IDs", value=True,
                                        help="Check for duplicate values in ID columns")
            
            with col2:
                check_formats = st.checkbox("Format Inconsistencies", value=False,
                                        help="Check for inconsistent text formats (e.g., mixed case)")
                check_relationships = st.checkbox("Value Relationships", value=False,
                                                help="Check for unusual relationships between columns")
                check_distributions = st.checkbox("Distribution Anomalies", value=False,
                                                help="Check for unusual distributions that might indicate data issues")
            
            # Add this right after the Quick Scan Options section
            with st.expander("About Data Consistency Checks", expanded=False):
                st.markdown("""
                ### Types of Data Consistency Checks
                
                This tool helps identify logical inconsistencies in your dataset:
                
                * **Percentage Ranges**: Identifies numeric columns that appear to contain percentage values but have values outside the expected range (0-1 or 0-100).
                
                * **Date Order Issues**: Checks for chronological issues between date columns (e.g., where an end date occurs before a start date).
                
                * **Duplicate IDs**: Finds duplicate values in columns that appear to be identifier columns, which typically should contain unique values.
                
                * **Format Inconsistencies**: Detects when text columns have inconsistent formats (e.g., mixed case, different capitalization patterns).
                
                * **Value Relationships**: Identifies unusual relationships between numeric columns, such as perfect correlations which might indicate redundant data.
                
                * **Distribution Anomalies**: Checks for multimodal distributions in numeric columns, which might indicate data sourced from different populations.
                
                ### How to Use
                
                1. Select the types of checks you want to run using the checkboxes above
                2. Click "Run Consistency Checks" to analyze your dataset
                3. Review the issues found and select appropriate fix strategies
                4. Queue fixes for the issues you want to address
                5. Apply all queued fixes with a single click
                """)
            
            # Create options dictionary
            options = {
                'check_percentage_ranges': check_pct,
                'check_date_order': check_date,
                'check_duplicates': check_dupes,
                'check_formats': check_formats,
                'check_value_relationships': check_relationships,
                'check_distribution_anomalies': check_distributions
            }
            
            # Check data consistency when button is clicked
            if st.button("Run Consistency Checks", type="primary") or st.button("Refresh Consistency Analysis"):
                with st.spinner("Checking data consistency..."):
                    # Create data cleaner instance
                    cleaner = DataCleaner(st.session_state.original_df.copy())
                    
                    # Run checks with selected options
                    consistency_info = cleaner.check_data_consistency(options)
                    
                    # Store results in session state
                    st.session_state.consistency_info = consistency_info
                    
                    # Initialize fixes to apply queue if it doesn't exist
                    if "fixes_to_apply" not in st.session_state:
                        st.session_state.fixes_to_apply = []
            
            # Display results if available
            if "consistency_info" in st.session_state:
                st.subheader("Consistency Issues Found")
                
                consistency_info = st.session_state.consistency_info
                issues = consistency_info.get("issues", [])
                issue_count = consistency_info.get("issue_count", 0)
                
                if issue_count == 0:
                    st.success("No data consistency issues found!")
                else:
                    # Show total issues found
                    st.info(f"Found {issue_count} potential data consistency issues.")
                    
                    # Group issues by type
                    issue_types = {}
                    for issue in issues:
                        issue_type = issue.get("type", "unknown")
                        if issue_type not in issue_types:
                            issue_types[issue_type] = []
                        issue_types[issue_type].append(issue)
                    
                    # Create a pie chart of issue types
                    fig = go.Figure(
                        data=[go.Pie(
                            labels=list(issue_types.keys()),
                            values=[len(issues) for issues in issue_types.values()],
                            hole=.3,
                            textinfo="label+percent",
                            insidetextorientation="radial"
                        )]
                    )
                    fig.update_layout(
                        title="Types of Data Consistency Issues",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display issues by type with expanders
                    for issue_type, type_issues in issue_types.items():
                        with st.expander(f"{issue_type.replace('_', ' ').title()} Issues ({len(type_issues)})", expanded=True):
                            for i, issue in enumerate(type_issues):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**Issue {i+1}:** {issue.get('details', 'Unknown issue')}")
                                    
                                    # Show sample data
                                    if "sample" in issue:
                                        st.write("Sample affected data:")
                                        
                                        # Format sample data for display
                                        if issue_type == "percentage_range":
                                            sample_df = pd.DataFrame(issue["sample"])
                                            if not sample_df.empty:
                                                sample_df.columns = ["Row Index", "Value"]
                                                st.dataframe(sample_df, use_container_width=True)
                                        
                                        elif issue_type == "duplicate_id":
                                            sample_df = pd.DataFrame(issue["sample"])
                                            if not sample_df.empty:
                                                sample_df.columns = ["Value", "Count"]
                                                st.dataframe(sample_df, use_container_width=True)
                                        
                                        elif issue_type == "date_order":
                                            sample_df = pd.DataFrame(issue["sample"])
                                            if not sample_df.empty:
                                                sample_df.columns = ["Row Index", "Start Date", "End Date"]
                                                st.dataframe(sample_df, use_container_width=True)
                                                
                                        elif issue_type == "format_inconsistency":
                                            sample_df = pd.DataFrame(issue["sample"])
                                            if not sample_df.empty:
                                                sample_df.columns = ["Format", "Example"]
                                                st.dataframe(sample_df, use_container_width=True)
                                                
                                        elif issue_type == "high_correlation":
                                            st.write(f"Correlation coefficient: {issue.get('correlation', 0):.4f}")
                                            
                                        elif issue_type == "multimodal_distribution":
                                            st.write(f"Detected peaks at: {', '.join([str(round(p, 2)) for p in issue.get('peaks', [])])}")
                                            
                                with col2:
                                    # Provide fix options based on issue type
                                    fix_strategy = None
                                    
                                    if issue_type == "percentage_range":
                                        fix_strategy = st.selectbox(
                                            "Fix Strategy", 
                                            ["Select", "Cap Values", "Rescale"], 
                                            key=f"fix_{issue_type}_{i}"
                                        )
                                        
                                        if fix_strategy == "Cap Values":
                                            fix_strategy = "cap"
                                        elif fix_strategy == "Rescale":
                                            fix_strategy = "rescale"
                                        else:
                                            fix_strategy = None
                                    
                                    elif issue_type == "duplicate_id":
                                        fix_strategy = st.selectbox(
                                            "Fix Strategy", 
                                            ["Select", "Keep First", "Make Unique"], 
                                            key=f"fix_{issue_type}_{i}"
                                        )
                                        
                                        if fix_strategy == "Keep First":
                                            fix_strategy = "keep_first"
                                        elif fix_strategy == "Make Unique":
                                            fix_strategy = "make_unique"
                                        else:
                                            fix_strategy = None
                                    
                                    elif issue_type == "date_order":
                                        fix_strategy = st.selectbox(
                                            "Fix Strategy", 
                                            ["Select", "Swap Dates", "Clear Dates"], 
                                            key=f"fix_{issue_type}_{i}"
                                        )
                                        
                                        if fix_strategy == "Swap Dates":
                                            fix_strategy = "swap"
                                        elif fix_strategy == "Clear Dates":
                                            fix_strategy = "clear"
                                        else:
                                            fix_strategy = None
                                            
                                    elif issue_type == "format_inconsistency":
                                        fix_strategy = st.selectbox(
                                            "Fix Strategy", 
                                            ["Select", "lowercase", "UPPERCASE", "Title Case"], 
                                            key=f"fix_{issue_type}_{i}"
                                        )
                                        
                                        if fix_strategy == "lowercase":
                                            fix_strategy = "lowercase"
                                        elif fix_strategy == "UPPERCASE":
                                            fix_strategy = "uppercase"
                                        elif fix_strategy == "Title Case":
                                            fix_strategy = "titlecase"
                                        else:
                                            fix_strategy = None
                                            
                                    elif issue_type == "high_correlation":
                                        fix_strategy = st.selectbox(
                                            "Fix Strategy", 
                                            ["Select", "Drop Redundant Column"], 
                                            key=f"fix_{issue_type}_{i}"
                                        )
                                        
                                        if fix_strategy == "Drop Redundant Column":
                                            fix_strategy = "drop_second"
                                        else:
                                            fix_strategy = None
                                            
                                    elif issue_type == "multimodal_distribution":
                                        fix_strategy = st.selectbox(
                                            "Fix Strategy", 
                                            ["Select", "Standardize", "Flag Clusters"], 
                                            key=f"fix_{issue_type}_{i}"
                                        )
                                        
                                        if fix_strategy == "Standardize":
                                            fix_strategy = "standardize"
                                        elif fix_strategy == "Flag Clusters":
                                            fix_strategy = "flag_clusters"
                                        else:
                                            fix_strategy = None
                                    
                                    # Add to queue if a fix is selected
                                    if fix_strategy and fix_strategy != "Select":
                                        if st.button("Queue Fix", key=f"queue_{issue_type}_{i}"):
                                            issue_copy = issue.copy()
                                            issue_copy["fix_strategy"] = fix_strategy
                                            
                                            # Add to fix queue
                                            if "fixes_to_apply" not in st.session_state:
                                                st.session_state.fixes_to_apply = []
                                            
                                            st.session_state.fixes_to_apply.append(issue_copy)
                                            st.success(f"Added fix to queue.")
                                            st.rerun()
                
                # Show queued fixes
                if "fixes_to_apply" in st.session_state and len(st.session_state.fixes_to_apply) > 0:
                    st.subheader("Fixes Queued for Application")
                    
                    fixes_df = []
                    for fix in st.session_state.fixes_to_apply:
                        fixes_df.append({
                            "Issue Type": fix.get("type", "").replace("_", " ").title(),
                            "Column(s)": fix.get("column", "") if "column" in fix else ", ".join(fix.get("columns", [])),
                            "Fix Strategy": fix.get("fix_strategy", "").replace("_", " ").title(),
                        })
                    
                    st.dataframe(pd.DataFrame(fixes_df), use_container_width=True)
                    
                    # Button to apply all fixes
                    if st.button("Apply All Queued Fixes", type="primary"):
                        with st.spinner("Applying fixes..."):
                            # Create data cleaner instance
                            cleaner = DataCleaner(st.session_state.original_df.copy())
                            
                            # Store original issue count
                            original_issue_count = st.session_state.consistency_info.get("issue_count", 0)
                            
                            # Apply the fixes
                            fix_results = cleaner.fix_data_consistency_issues(st.session_state.fixes_to_apply)
                            
                            # Get the cleaned dataframe
                            cleaned_df = cleaner.get_cleaned_data()
                            
                            # Re-run consistency checks to see what issues remain
                            remaining_info = cleaner.check_data_consistency(options)
                            remaining_issue_count = remaining_info.get("issue_count", 0)
                            
                            # Calculate metrics
                            issues_fixed = sum(t.get("issues_fixed", 0) for t in fix_results.get("transformations", []))
                            rows_modified = fix_results.get("rows_modified", 0)
                            
                            # Store results for display
                            st.session_state.consistency_fixes_applied = {
                                "issues_fixed": issues_fixed,
                                "rows_modified": rows_modified,
                                "original_issue_count": original_issue_count,
                                "remaining_issue_count": remaining_issue_count
                            }
                            
                            # Add tracking for data consistency changes
                            st.session_state.changes_tracking['data_consistency'].append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'issues_fixed': issues_fixed,
                                'rows_modified': rows_modified,
                                'original_issue_count': original_issue_count,
                                'remaining_issue_count': remaining_issue_count,
                                'fix_types': [t.get('type', 'unknown') for t in fix_results.get('transformations', [])]
                            })
                            
                            # Update the original DataFrame
                            st.session_state.original_df = cleaned_df
                            
                            # Add to processing history
                            if "processing_history" not in st.session_state:
                                st.session_state.processing_history = []
                            
                            st.session_state.processing_history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "action": "fix_data_consistency",
                                "details": f"Applied {len(st.session_state.fixes_to_apply)} consistency fixes"
                            })
                            
                            # Clear the fixes queue
                            st.session_state.fixes_to_apply = []
                            
                            # Delete consistency info to refresh
                            if "consistency_info" in st.session_state:
                                del st.session_state.consistency_info
                            
                            # Show success message
                            st.success("Data consistency fixes applied successfully!")
                            st.rerun()
    elif st.session_state.current_tab == "cleaning_report":
        st.header("Data Cleaning Report")
        st.write("Summary of transformations and improvements to your dataset.")
        
        # Compare original and current dataset
        if st.session_state.original_df is not None:
            # Basic statistics
            original_shape = st.session_state.cleaner.original_df.shape
            current_shape = st.session_state.original_df.shape
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Rows", 
                    f"{current_shape[0]:,}", 
                    delta=f"{current_shape[0] - original_shape[0]:,}",
                    delta_color="normal"
                )
            
            with col2:
                missing_before = st.session_state.cleaner.original_df.isna().sum().sum()
                missing_after = st.session_state.original_df.isna().sum().sum()
                
                st.metric(
                    "Missing Values", 
                    f"{missing_after:,}", 
                    delta=f"{missing_after - missing_before:,}",
                    delta_color="inverse"  # Negative is good here
                )
            
            with col3:
                # Calculate approx memory usage change
                memory_before = st.session_state.cleaner.memory_usage_mb
                memory_after = estimate_memory_usage(st.session_state.original_df)
                
                st.metric(
                    "Memory Usage", 
                    f"{memory_after:.2f} MB", 
                    delta=f"{memory_after - memory_before:.2f} MB",
                    delta_color="normal"
                )
            
            # Processing history
            st.subheader("Processing Steps")
            
            for i, event in enumerate(st.session_state.processing_history):
                st.write(f"**{i+1}. {event['timestamp']}**: {event['details']}")
    
    elif st.session_state.current_tab == "export":
        st.header("Export Cleaned Data")
        
        # Export options - move these outside the export button click handler
        export_format = st.radio(
            "Select export format",
            options=["CSV", "Excel", "JSON", "Pickle"],
            help="Choose the format for exporting your cleaned dataset"
        )
        
        # Compression options
        compression_options = {
            "CSV": ["None", "gzip", "zip"],
            "Excel": ["None"],
            "JSON": ["None", "gzip", "zip"],
            "Pickle": ["None", "gzip"]
        }
        
        compression = st.selectbox(
            "Compression",
            options=compression_options[export_format],
            help="Compress the output file to save space"
        )
        
        # Index options
        include_index = st.checkbox("Include index in export", value=False)
        
        # Validate sheet name (Excel has limitations on sheet names)
        if export_format == "Excel":
            sheet_name = st.text_input("Sheet Name", value="Cleaned Data")
            if len(sheet_name) > 31:
                st.warning("Sheet name is too long. Maximum length is 31 characters.")
                sheet_name = sheet_name[:31]
            
            # Check for invalid characters in sheet name
            invalid_chars = [':', '\\', '/', '?', '*', '[', ']']
            if any(c in sheet_name for c in invalid_chars):
                st.warning("Sheet name contains invalid characters. They will be replaced.")
                for c in invalid_chars:
                    sheet_name = sheet_name.replace(c, '_')
            
            # Add Excel-specific export options
            with st.expander("Excel Advanced Options"):
                freeze_panes = st.checkbox("Freeze header row", value=True, 
                                          help="Freeze the first row so it remains visible when scrolling")
                auto_filter = st.checkbox("Add auto-filter to header", value=True,
                                         help="Add filter dropdown menus to column headers")
                add_table_style = st.checkbox("Format as Excel table", value=False,
                                             help="Apply table formatting with alternating row colors")
                include_stats = st.checkbox("Include statistics sheet", value=False,
                                           help="Add a second sheet with summary statistics")
        
        # Export button
        if st.button("Export Data"):
            try:
                # Generate secure filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = generate_secure_filename(st.session_state.file_info['filename'])
                
                if export_format == "CSV":
                    if compression == "None":
                        filename = f"{base_filename}_cleaned_{timestamp}.csv"
                        st.session_state.original_df.to_csv(filename, index=include_index)
                    else:
                        filename = f"{base_filename}_cleaned_{timestamp}.csv.{compression}"
                        st.session_state.original_df.to_csv(filename, compression=compression, index=include_index)
                
                elif export_format == "Excel":
                    # When exporting
                    filename = f"{base_filename}_cleaned_{timestamp}.xlsx"
                    writer = pd.ExcelWriter(filename, engine='openpyxl')
                    try:
                        # All Excel writing operations
                        st.session_state.original_df.to_excel(writer, sheet_name=sheet_name, index=include_index)
                        
                        # Add formatting if Excel Advanced Options were selected
                        worksheet = writer.sheets[sheet_name]
                        
                        # Apply Excel-specific formatting
                        freeze_panes = False  # Default value
                        auto_filter = False   # Default value
                        add_table_style = False  # Default value

                        if freeze_panes:
                            worksheet.freeze_panes = 'A2'  # Freeze the first row
                        
                        if auto_filter:
                            worksheet.auto_filter.ref = worksheet.dimensions
                        
                        if add_table_style:
                            try:
                                from openpyxl.worksheet.table import Table, TableStyleInfo
                                
                                # Create a table with unique name (avoiding spaces and special chars)
                                safe_name = sheet_name.replace(' ', '_').replace('-', '_')
                                table_name = f"Table_{safe_name}_{timestamp}"[:31]  # Excel has 31 char limit for table names
                                
                                # Get dimensions of data
                                data_range = worksheet.dimensions
                                
                                # Create table and apply style
                                tab = Table(displayName=table_name, ref=data_range)
                                style = TableStyleInfo(name="TableStyleMedium9", showRowStripes=True)
                                tab.tableStyleInfo = style
                                
                                # Add table to worksheet
                                worksheet.add_table(tab)
                            except Exception as e:
                                st.warning(f"Could not apply table style: {str(e)}")
                    finally:
                        writer.close()
                
                elif export_format == "JSON":
                    if compression == "None":
                        filename = f"{base_filename}_cleaned_{timestamp}.json"
                        st.session_state.original_df.to_json(filename, orient="records")
                    else:
                        filename = f"{base_filename}_cleaned_{timestamp}.json.{compression}"
                        st.session_state.original_df.to_json(filename, orient="records", compression=compression)
                
                elif export_format == "Pickle":
                    if compression == "None":
                        filename = f"{base_filename}_cleaned_{timestamp}.pkl"
                        st.session_state.original_df.to_pickle(filename)
                    else:
                        filename = f"{base_filename}_cleaned_{timestamp}.pkl.{compression}"
                        st.session_state.original_df.to_pickle(filename, compression=compression)
                
                # Success message
                st.success(f"Data exported to {filename}")
                
                # Download button
                with open(filename, "rb") as f:
                    st.download_button(
                        label=f"Download {export_format} file",
                        data=f,
                        file_name=filename,
                        mime={
                            "CSV": "text/csv",
                            "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            "JSON": "application/json",
                            "Pickle": "application/octet-stream"
                        }[export_format]
                    )
                
                # Cleanup file after download
                os.remove(filename)
                
                # Add export event to history
                st.session_state.processing_history.append({
                    'action': 'export',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'details': f"Exported cleaned data to {filename}"
                })
                
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")

    elif st.session_state.current_tab == "changes_visualization":
        st.header("Changes Visualization")
        st.write("Visualize the changes made to your dataset across all cleaning steps.")
        
        # Add a refresh button
        refresh_vis = st.button("Refresh Visualizations")
        
        if not st.session_state.changes_tracking['missing_values'] and not st.session_state.changes_tracking['outliers'] and not st.session_state.changes_tracking['data_consistency']:
            st.info("No changes have been made to the dataset yet. Apply some cleaning operations in the Missing Values, Outliers, or Data Consistency tabs first.")
        else:
            # Create tabs for different visualizations
            viz_tabs = st.tabs(["Missing Values", "Outliers", "Data Consistency", "Combined View"])
            
            # Missing Values tab
            with viz_tabs[0]:
                if not st.session_state.changes_tracking['missing_values']:
                    st.info("No missing values operations have been performed yet.")
                else:
                    st.subheader("Missing Values Handling")
                    
                    # Create a dataframe from the tracked changes
                    missing_df = pd.DataFrame(st.session_state.changes_tracking['missing_values'])
                    
                    # Add a column for time sequence
                    missing_df['sequence'] = range(1, len(missing_df) + 1)
                    
                    # Bar chart showing before and after for each operation
                    fig1 = px.bar(
                        missing_df,
                        x='sequence',
                        y=['before', 'after'],
                        barmode='group',
                        labels={'sequence': 'Operation Sequence', 'value': 'Missing Values Count'},
                        title='Missing Values Before & After Each Operation',
                        hover_data=['column', 'strategy', 'timestamp']
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Line chart showing cumulative impact
                    missing_df['cumulative_fixed'] = missing_df['difference'].cumsum()
                    fig2 = px.line(
                        missing_df,
                        x='sequence',
                        y='cumulative_fixed',
                        markers=True,
                        labels={'sequence': 'Operation Sequence', 'cumulative_fixed': 'Cumulative Missing Values Fixed'},
                        title='Cumulative Impact of Missing Values Operations',
                        hover_data=['column', 'strategy', 'timestamp']
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Table of operations
                    st.subheader("Missing Values Operations")
                    display_df = missing_df[['timestamp', 'column', 'strategy', 'before', 'after', 'difference']]
                    display_df.columns = ['Timestamp', 'Column', 'Strategy', 'Before', 'After', 'Values Fixed']
                    st.dataframe(display_df, use_container_width=True)
            
            # Outliers tab
            with viz_tabs[1]:
                if not st.session_state.changes_tracking['outliers']:
                    st.info("No outlier operations have been performed yet.")
                else:
                    st.subheader("Outliers Handling")
                    
                    # Create a dataframe from the tracked changes
                    outliers_df = pd.DataFrame(st.session_state.changes_tracking['outliers'])
                    
                    # Add a column for time sequence
                    outliers_df['sequence'] = range(1, len(outliers_df) + 1)
                    
                    # Bar chart showing before and after for each operation
                    fig1 = px.bar(
                        outliers_df,
                        x='sequence',
                        y=['before', 'after'],
                        barmode='group',
                        labels={'sequence': 'Operation Sequence', 'value': 'Outliers Count'},
                        title='Outliers Before & After Each Operation',
                        hover_data=['column', 'strategy', 'timestamp']
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Line chart showing cumulative impact
                    outliers_df['cumulative_fixed'] = outliers_df['difference'].cumsum()
                    fig2 = px.line(
                        outliers_df,
                        x='sequence',
                        y='cumulative_fixed',
                        markers=True,
                        labels={'sequence': 'Operation Sequence', 'cumulative_fixed': 'Cumulative Outliers Fixed'},
                        title='Cumulative Impact of Outlier Operations',
                        hover_data=['column', 'strategy', 'timestamp']
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Table of operations
                    st.subheader("Outlier Operations")
                    display_df = outliers_df[['timestamp', 'column', 'strategy', 'before', 'after', 'difference']]
                    display_df.columns = ['Timestamp', 'Column', 'Strategy', 'Before', 'After', 'Outliers Fixed']
                    st.dataframe(display_df, use_container_width=True)
            
            # Data Consistency tab
            with viz_tabs[2]:
                if not st.session_state.changes_tracking['data_consistency']:
                    st.info("No data consistency operations have been performed yet.")
                else:
                    st.subheader("Data Consistency Operations")
                    
                    # Create a dataframe from the tracked changes
                    consistency_df = pd.DataFrame(st.session_state.changes_tracking['data_consistency'])
                    
                    # Add a column for time sequence
                    consistency_df['sequence'] = range(1, len(consistency_df) + 1)
                    
                    # Bar chart showing before and after for each operation
                    fig1 = px.bar(
                        consistency_df,
                        x='sequence',
                        y=['original_issue_count', 'remaining_issue_count'],
                        barmode='group',
                        labels={'sequence': 'Operation Sequence', 'value': 'Issues Count'},
                        title='Consistency Issues Before & After Each Operation',
                        hover_data=['timestamp', 'issues_fixed', 'rows_modified']
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Line chart showing cumulative impact
                    consistency_df['improvement_pct'] = (consistency_df['original_issue_count'] - consistency_df['remaining_issue_count']) / consistency_df['original_issue_count'] * 100
                    fig2 = px.line(
                        consistency_df,
                        x='sequence',
                        y='improvement_pct',
                        markers=True,
                        labels={'sequence': 'Operation Sequence', 'improvement_pct': 'Improvement (%)'},
                        title='Data Quality Improvement (%)',
                        hover_data=['timestamp', 'issues_fixed', 'rows_modified']
                    )
                    fig2.update_layout(yaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Table of operations
                    st.subheader("Data Consistency Operations")
                    display_df = consistency_df[['timestamp', 'issues_fixed', 'rows_modified', 'original_issue_count', 'remaining_issue_count']]
                    display_df.columns = ['Timestamp', 'Issues Fixed', 'Rows Modified', 'Before', 'After']
                    st.dataframe(display_df, use_container_width=True)
            
            # Combined view tab
            with viz_tabs[3]:
                st.subheader("Combined Data Quality Improvement")
                
                # Prepare data for combined view
                combined_data = []
                
                # Missing values data
                for i, item in enumerate(st.session_state.changes_tracking['missing_values']):
                    combined_data.append({
                        'timestamp': item['timestamp'],
                        'operation_type': 'Missing Values',
                        'sequence': i + 1,
                        'items_fixed': item['difference'],
                        'column': item['column'],
                        'strategy': item['strategy']
                    })
                
                # Outliers data
                for i, item in enumerate(st.session_state.changes_tracking['outliers']):
                    combined_data.append({
                        'timestamp': item['timestamp'],
                        'operation_type': 'Outliers',
                        'sequence': i + 1,
                        'items_fixed': item['difference'],
                        'column': item['column'],
                        'strategy': item['strategy']
                    })
                
                # Consistency data
                for i, item in enumerate(st.session_state.changes_tracking['data_consistency']):
                    combined_data.append({
                        'timestamp': item['timestamp'],
                        'operation_type': 'Data Consistency',
                        'sequence': i + 1,
                        'items_fixed': item['issues_fixed'],
                        'rows_modified': item['rows_modified']
                    })
                
                if combined_data:
                    # Create combined dataframe and sort by timestamp
                    combined_df = pd.DataFrame(combined_data)
                    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                    combined_df = combined_df.sort_values('timestamp')
                    combined_df['global_sequence'] = range(1, len(combined_df) + 1)
                    
                    # Bar chart by operation type
                    fig1 = px.bar(
                        combined_df,
                        x='global_sequence',
                        y='items_fixed',
                        color='operation_type',
                        labels={'global_sequence': 'Operation Sequence', 'items_fixed': 'Items Fixed'},
                        title='Data Quality Improvement by Operation Type',
                        hover_data=['timestamp', 'operation_type']
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Pie chart of operations by type
                    fig2 = px.pie(
                        combined_df,
                        names='operation_type',
                        title='Distribution of Cleaning Operations'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Timeline of all operations
                    st.subheader("Timeline of All Data Cleaning Operations")
                    display_df = combined_df[['timestamp', 'operation_type', 'items_fixed']]
                    display_df.columns = ['Timestamp', 'Operation Type', 'Items Fixed']
                    st.dataframe(display_df.sort_values('Timestamp', ascending=False), use_container_width=True)
                else:
                    st.info("No operations performed on the dataset yet.")

# Show footer
st.markdown("---")
st.markdown("Dataset Cleaner | Created by [Louce (Dendi Rivaldi)](https://github.com/Louce/dataset-cleaner) | Built with Streamlit")

