# Sample Datasets

This folder contains sample datasets that can be used to test the CSV Dataset Cleaner functionality.

## Available Datasets

### 1. `sample_data.csv`

A general-purpose dataset with various data quality issues for demonstrating the application's features:

- Contains missing values in different columns
- Includes outliers in numeric columns
- Has data consistency issues (percentage values outside valid range, date order issues)
- Includes duplicate ID values

### Usage

To use these sample datasets:

1. Start the application with `streamlit run app.py`
2. In the sidebar, click "Upload CSV File"
3. Navigate to this folder and select the desired sample dataset
4. Click "Import Data" to load the dataset into the application

## Creating Custom Test Datasets

You can add your own test datasets to this folder. For best results:

1. Include a variety of data types (numeric, text, dates)
2. Incorporate common data quality issues
3. Keep the file size manageable (under 10MB for quick testing)
4. Document the dataset structure and known issues 