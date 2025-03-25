# Sample Datasets

This folder contains sample datasets that can be used to test the Dataset Cleaner functionality.

## Available Datasets

### CSV Examples

Located in the `csv_examples/` directory:

- **sample_data.csv**: A general-purpose dataset with various data quality issues for demonstrating the application's features:
  - Contains missing values in different columns
  - Includes outliers in numeric columns
  - Has data consistency issues (percentage values outside valid range, date order issues)
  - Includes duplicate ID values

### Excel Examples

Located in the `excel_examples/` directory:

- **sample_data.xlsx**: Multi-sheet workbook with various data quality issues across different sheets
- **formatting_example.xlsx**: Example with formatted cells to demonstrate Excel-specific import options

## Usage

To use these sample datasets:

1. Start the application with `streamlit run src/app.py`
2. In the sidebar, click "Upload Data File"
3. Navigate to the appropriate folder (`csv_examples/` or `excel_examples/`) and select the desired sample dataset
4. Configure any format-specific import settings in the "Advanced Import Settings" section
5. Click "Import Data" to load the dataset into the application

## Creating Custom Test Datasets

You can add your own test datasets to this folder. For best results:

1. Include a variety of data types (numeric, text, dates)
2. Incorporate common data quality issues
3. Keep the file size manageable (under 10MB for quick testing)
4. For Excel files, include multiple sheets and formatted cells to test advanced features
5. Document the dataset structure and known issues in this README 