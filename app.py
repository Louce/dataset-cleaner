import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from data_cleaning import DataCleaner
from data_visualization import DataVisualizer
from utils import get_dataset_info, save_dataframe

# Set page configuration
st.set_page_config(
    page_title="CSV Dataset Cleaner",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("CSV Dataset Cleaner")
st.markdown("""
This application allows you to upload any CSV file, clean the dataset, view data quality metrics,
and export the cleaned dataset. All transformations made to the data are documented.
""")

# Initialize session state variables if they don't exist
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
    st.session_state.current_df = None
    st.session_state.transformations = []
    st.session_state.cleaned = False

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        st.session_state.original_df = pd.read_csv(uploaded_file)
        st.session_state.current_df = st.session_state.original_df.copy()
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Original Dataset", "Data Cleaning", "Data Quality Report", "Statistical Summary", "Export Data"]
)

# Only show content if a file is uploaded
if st.session_state.original_df is not None:
    # Display dataset info in the sidebar
    dataset_info = get_dataset_info(st.session_state.original_df)
    with st.sidebar.expander("Dataset Information"):
        st.write(f"Rows: {dataset_info['rows']}")
        st.write(f"Columns: {dataset_info['columns']}")
        st.write(f"Numeric columns: {', '.join(dataset_info['numeric_columns'])}")
        st.write(f"Categorical columns: {', '.join(dataset_info['categorical_columns'])}")

    # Data cleaner instance
    cleaner = DataCleaner(st.session_state.original_df)
    visualizer = DataVisualizer()

    # Original Dataset page
    if page == "Original Dataset":
        st.header("Original Dataset")
        
        # Display dataset dimensions
        st.write(f"Dataset dimensions: {st.session_state.original_df.shape[0]} rows x {st.session_state.original_df.shape[1]} columns")
        
        # Display the dataset
        st.dataframe(st.session_state.original_df)
        
        # Display column info
        st.subheader("Column Information")
        column_info = pd.DataFrame({
            'Column': st.session_state.original_df.columns,
            'Data Type': st.session_state.original_df.dtypes.astype(str),
            'Missing Values': st.session_state.original_df.isnull().sum(),
            'Missing Percentage': (st.session_state.original_df.isnull().sum() / len(st.session_state.original_df) * 100).round(2),
            'Unique Values': [st.session_state.original_df[col].nunique() for col in st.session_state.original_df.columns]
        })
        
        st.dataframe(column_info)
        
        # Sample data visualization
        st.subheader("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show distribution of a numeric column
            numeric_cols = st.session_state.original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                selected_num_col = st.selectbox("Select a numeric column for histogram", numeric_cols)
                fig = visualizer.plot_histogram(st.session_state.original_df, selected_num_col)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show distribution of a categorical column
            cat_cols = st.session_state.original_df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                selected_cat_col = st.selectbox("Select a categorical column for bar chart", cat_cols)
                fig = visualizer.plot_bar_chart(st.session_state.original_df, selected_cat_col)
                st.plotly_chart(fig, use_container_width=True)

    # Data Cleaning page
    elif page == "Data Cleaning":
        st.header("Data Cleaning")
        
        if st.session_state.cleaned:
            st.success("Dataset has been cleaned! Navigate to other sections to explore the cleaned data.")
            
            # Show transformations
            st.subheader("Transformations Applied")
            for i, transformation in enumerate(st.session_state.transformations, 1):
                st.write(f"{i}. {transformation}")
                
            # Reset button to start over
            if st.button("Reset and Clean Again"):
                st.session_state.current_df = st.session_state.original_df.copy()
                st.session_state.transformations = []
                st.session_state.cleaned = False
                st.rerun()
        else:
            st.write("Click the button below to clean the dataset. The cleaning process includes:")
            st.markdown("""
            - Handling missing values
            - Detecting and handling outliers
            - Standardizing categorical variables
            - Ensuring data consistency
            """)
            
            # Clean data button
            if st.button("Clean Dataset"):
                with st.spinner("Cleaning dataset..."):
                    # Handle missing values
                    df_no_missing, missing_transformations = cleaner.handle_missing_values(st.session_state.current_df)
                    st.session_state.transformations.extend(missing_transformations)
                    
                    # Standardize categorical variables
                    df_standardized, standardize_transformations = cleaner.standardize_categorical_variables(df_no_missing)
                    st.session_state.transformations.extend(standardize_transformations)
                    
                    # Handle outliers
                    df_no_outliers, outlier_transformations = cleaner.handle_outliers(df_standardized)
                    st.session_state.transformations.extend(outlier_transformations)
                    
                    # Final consistency checks
                    df_consistent, consistency_transformations = cleaner.ensure_data_consistency(df_no_outliers)
                    st.session_state.transformations.extend(consistency_transformations)
                    
                    # Update the current dataframe
                    st.session_state.current_df = df_consistent
                    st.session_state.cleaned = True
                    
                st.success("Dataset cleaned successfully!")
                st.rerun()

    # Data Quality Report page
    elif page == "Data Quality Report":
        st.header("Data Quality Report")
        
        # Check if data is cleaned
        if not st.session_state.cleaned:
            st.warning("Please clean the dataset first by going to the 'Data Cleaning' page.")
            st.stop()
        
        # Create tabs for before and after
        tab1, tab2, tab3 = st.tabs(["Before vs After", "Missing Values", "Data Type Changes"])
        
        with tab1:
            st.subheader("Data Quality Metrics: Before vs After Cleaning")
            
            # Create a metrics comparison dataframe
            metrics_comparison = pd.DataFrame({
                'Metric': ['Number of rows', 'Number of columns', 'Missing values', 'Duplicated rows'],
                'Before': [
                    st.session_state.original_df.shape[0],
                    st.session_state.original_df.shape[1],
                    st.session_state.original_df.isnull().sum().sum(),
                    st.session_state.original_df.duplicated().sum()
                ],
                'After': [
                    st.session_state.current_df.shape[0],
                    st.session_state.current_df.shape[1],
                    st.session_state.current_df.isnull().sum().sum(),
                    st.session_state.current_df.duplicated().sum()
                ]
            })
            
            st.dataframe(metrics_comparison)
            
            # Visualize the comparison
            for metric in metrics_comparison['Metric']:
                if metric in ['Number of rows', 'Number of columns', 'Missing values', 'Duplicated rows']:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Before', 'After'],
                        y=[metrics_comparison.loc[metrics_comparison['Metric'] == metric, 'Before'].values[0],
                           metrics_comparison.loc[metrics_comparison['Metric'] == metric, 'After'].values[0]],
                        name=metric
                    ))
                    fig.update_layout(title=f'{metric}: Before vs After', yaxis_title='Count')
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Missing Values: Before vs After Cleaning")
            
            # Calculate missing values for each column before and after
            missing_before = pd.DataFrame({
                'Column': st.session_state.original_df.columns,
                'Missing Count': st.session_state.original_df.isnull().sum(),
                'Missing Percentage': (st.session_state.original_df.isnull().sum() / len(st.session_state.original_df) * 100).round(2)
            })
            
            missing_after = pd.DataFrame({
                'Column': st.session_state.current_df.columns,
                'Missing Count': st.session_state.current_df.isnull().sum(),
                'Missing Percentage': (st.session_state.current_df.isnull().sum() / len(st.session_state.current_df) * 100).round(2)
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Before Cleaning")
                st.dataframe(missing_before)
            
            with col2:
                st.write("After Cleaning")
                st.dataframe(missing_after)
        
        with tab3:
            st.subheader("Data Type Changes")
            
            # Compare data types before and after cleaning
            dtypes_before = pd.DataFrame({
                'Column': st.session_state.original_df.columns,
                'Data Type Before': st.session_state.original_df.dtypes.astype(str)
            })
            
            dtypes_after = pd.DataFrame({
                'Column': st.session_state.current_df.columns,
                'Data Type After': st.session_state.current_df.dtypes.astype(str)
            })
            
            # Merge the two dataframes
            dtypes_comparison = pd.merge(dtypes_before, dtypes_after, on='Column')
            
            # Add a column to indicate if the data type changed
            dtypes_comparison['Changed'] = dtypes_comparison['Data Type Before'] != dtypes_comparison['Data Type After']
            
            # Display the dataframe
            st.dataframe(dtypes_comparison)
            
            # Display only the columns where the data type changed
            st.subheader("Changed Data Types")
            changed_dtypes = dtypes_comparison[dtypes_comparison['Changed']]
            if not changed_dtypes.empty:
                st.dataframe(changed_dtypes)
            else:
                st.write("No data type changes were made during cleaning.")

    # Statistical Summary page
    elif page == "Statistical Summary":
        st.header("Statistical Summary")
        
        # Check if data is cleaned
        if not st.session_state.cleaned:
            st.warning("Please clean the dataset first by going to the 'Data Cleaning' page.")
            st.stop()
        
        # Create tabs for numerical and categorical summaries
        tab1, tab2 = st.tabs(["Numerical Summary", "Categorical Summary"])
        
        with tab1:
            st.subheader("Numerical Columns Summary")
            
            # Get numerical columns
            numerical_cols = st.session_state.current_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numerical_cols:
                # Display numerical statistics
                num_summary = st.session_state.current_df[numerical_cols].describe().T
                num_summary['Range'] = st.session_state.current_df[numerical_cols].max() - st.session_state.current_df[numerical_cols].min()
                num_summary['Median'] = st.session_state.current_df[numerical_cols].median()
                st.dataframe(num_summary)
                
                # Display distributions
                selected_num_col = st.selectbox("Select a numerical column to visualize", numerical_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot histogram
                    fig = visualizer.plot_histogram(st.session_state.current_df, selected_num_col)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Plot box plot
                    fig = visualizer.plot_box_plot(st.session_state.current_df, selected_num_col)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No numerical columns found in the dataset.")
        
        with tab2:
            st.subheader("Categorical Columns Summary")
            
            # Get categorical columns
            categorical_cols = st.session_state.current_df.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols:
                # Select a categorical column to display
                selected_cat_col = st.selectbox("Select a categorical column", categorical_cols)
                
                # Display value counts
                cat_counts = st.session_state.current_df[selected_cat_col].value_counts().reset_index()
                cat_counts.columns = [selected_cat_col, 'Count']
                cat_counts['Percentage'] = (cat_counts['Count'] / cat_counts['Count'].sum() * 100).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Value Counts")
                    st.dataframe(cat_counts)
                
                with col2:
                    # Plot bar chart
                    fig = visualizer.plot_bar_chart(st.session_state.current_df, selected_cat_col)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No categorical columns found in the dataset.")

    # Export Data page
    elif page == "Export Data":
        st.header("Export Cleaned Data")
        
        # Check if data is cleaned
        if not st.session_state.cleaned:
            st.warning("Please clean the dataset first by going to the 'Data Cleaning' page.")
            st.stop()
        
        # Display sample of cleaned data
        st.subheader("Sample of Cleaned Data")
        st.dataframe(st.session_state.current_df.head(10))
        
        # Export options
        st.subheader("Export Options")
        export_format = st.radio("Select export format", ["CSV", "Excel"])
        filename = st.text_input("Enter filename (without extension)", "cleaned_dataset")
        
        if st.button("Export Data"):
            try:
                if export_format == "CSV":
                    # Export to CSV
                    file_path = f"{filename}.csv"
                    success = save_dataframe(st.session_state.current_df, file_path, file_format='csv')
                else:
                    # Export to Excel
                    file_path = f"{filename}.xlsx"
                    success = save_dataframe(st.session_state.current_df, file_path, file_format='excel')
                
                if success:
                    st.success(f"Data exported successfully as {file_path}!")
                    
                    # Create a download link
                    st.download_button(
                        label=f"Download {export_format} file",
                        data=open(file_path, 'rb').read(),
                        file_name=file_path,
                        mime='application/octet-stream'
                    )
                else:
                    st.error(f"Failed to export data as {file_path}.")
            except Exception as e:
                st.error(f"Error exporting data: {e}")
else:
    st.info("Please upload a CSV file to begin.")
