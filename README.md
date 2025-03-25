# Dataset Cleaner

<div align="center">

![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

An interactive data cleaning platform for CSV and Excel datasets with intelligent issue detection and automated cleaning solutions. This application allows you to upload any CSV or Excel file, automatically identify data quality issues, and clean your data through an intuitive interface.

<div align="center">
  <img src="https://github.com/Louce/dataset-cleaner/blob/master/Asset/Screenshot.png" alt="Dataset Cleaner Screenshot" width="80%">
</div>

<div align="center">

## ✨ [Try the Live Demo!](https://csv-dataset-cleaner-6s8m2uemcdhrxrghybne4k.streamlit.app/) ✨
No installation required. Clean your data instantly in your browser.

</div>

> **Note:** The live demo is deployed using Python 3.12 for compatibility with Streamlit Cloud. For local development, use Python 3.13 with `requirements/requirements-py313.txt` for the full experience.

## 🚀 Features

- **Multi-format Support**: Import data from CSV and Excel (.xlsx, .xls) files
- **Dynamic Dataset Handling**: Works with any file structure without hardcoded column references
- **Intelligent Missing Value Detection & Filling**:
  - **🧠 Synergy Strategy**: Automatically determines optimal fill methods using KNN imputation for numeric fields and conditional mode for categorical data
  - **📊 Statistical Methods**: Mean, median, mode for appropriate column types
  - **🔧 Custom Values**: Intuitive UI for entering custom replacement values
- **Advanced Outlier Detection & Handling**: 
  - Multiple detection algorithms (IQR, Z-score)
  - Flexible treatment options (removal, capping, transformation)
- **Data Consistency Verification**:
  - Automatic detection of logical inconsistencies
  - Powerful validation rules
- **Interactive Visualization Dashboard**:
  - Before/after cleaning comparisons
  - Data quality metrics
- **Flexible Export Options**: CSV, Excel, JSON, and Pickle formats with customizable settings

## 📋 Requirements

- Python 3.8+ (fully compatible with Python 3.13)
- Dependencies listed in `requirements/requirements.txt`
- Excel support provided by openpyxl (.xlsx), xlrd (.xls) and xlsxwriter (enhanced formatting)

## 💻 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Louce/dataset-cleaner.git
   cd dataset-cleaner
   ```

2. Create a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   # For local development with Python 3.13 (recommended for local use)
   pip install -r requirements/requirements-py313.txt
   
   # For Python 3.12 (primarily for Streamlit Cloud deployment)
   pip install -r requirements/requirements.txt
   ```

4. Launch the application:
   ```bash
   streamlit run src/app.py
   ```

## 🌐 Live Demo

Try the Dataset Cleaner without installation: [Live Demo](https://dataset-cleaner-6s8m2uemcdhrxrghybne4k.streamlit.app/)

## 📊 Usage Workflow

1. **Upload** your CSV or Excel file with format-specific import settings
2. **Analyze** your dataset with automated profiling
3. **Fix Missing Values** with intelligent strategies
4. **Handle Outliers** with precision
5. **Check Data Consistency** to identify logical errors
6. **Review Changes** with interactive visualizations
7. **Export** your cleaned dataset in your preferred format

## 🚀 Deploy Your Own

To deploy your own version using Streamlit Cloud:

1. Fork this repository to your GitHub account
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app in Streamlit Cloud
4. Enter your GitHub repository URL followed by the path to app.py:
   ```
   https://github.com/Louce/dataset-cleaner/blob/master/src/app.py
   ```
5. In the advanced settings, set the Python version to 3.12
6. Use the default requirements/requirements.txt file path
7. Deploy and share your application!

## 🧰 Advanced Features

### Excel-Specific Features

- **Multi-sheet Support**: Select which sheet to import from Excel workbooks
- **Advanced Import Options**: 
  - Choose header row location
  - Select specific columns to import
  - Convert formatted values to their display format
- **Enhanced Export Options**:
  - Custom sheet naming
  - Header row freezing
  - Automatic filtering
  - Excel table formatting with styles

### Missing Value Handling

Our intelligent system offers multiple strategies for handling missing data:

- **Synergy Strategy**: Uses machine learning techniques to determine the best approach:
  - KNN imputation for numeric columns based on data patterns
  - Conditional mode imputation for categorical data based on similar rows
  - Adaptive fallback methods when primary methods aren't applicable

- **Statistical Methods**: 
  - Mean/median for numeric data
  - Mode for categorical data
  - Zero substitution where appropriate

- **Custom Value Replacement**: 
  - Intuitive interface for entering replacement values
  - Quick selection options for common replacements
  - Preview of changes before applying

### Outlier Detection and Treatment

- **Multiple Detection Algorithms**:
  - IQR (Interquartile Range) method
  - Z-score method with configurable thresholds
  
- **Flexible Treatment Options**:
  - Remove outliers
  - Cap at boundaries
  - Transform values

### Data Consistency Verification

- **Automated Issue Detection**:
  - Percentage range validation
  - Date order verification
  - Duplicate ID detection
  - Format consistency checking

- **Intelligent Fixing Options**:
  - Context-aware correction suggestions
  - Batch fixing of related issues

### Change Tracking and Visualization

- **Comprehensive Change Log**:
  - Track all transformations
  - Compare before and after states
  
- **Interactive Visualizations**:
  - Charts showing impact of cleaning operations
  - Visual data quality assessment

## 🔧 Development

For developers who want to contribute or customize:

1. Install development dependencies:
   ```bash
   pip install -r requirements/requirements-dev.txt
   ```

2. Run tests:
   ```bash
   pytest
   ```

3. Check code quality:
   ```bash
   flake8 .
   black .
   ```

## 📚 Documentation

- [Contributing Guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Changelog](CHANGELOG.md)
- [Security Policy](SECURITY.md)

## 🧪 Python 3.13 Compatibility

This project is fully compatible with Python 3.13. All dependencies have been verified to work with the latest Python version. If you encounter any compatibility issues, please run the included compatibility test scripts:

- Windows: `scripts/run_compatibility_test.bat`
- Linux/macOS: `scripts/run_compatibility_test.sh`

## 📂 Project Structure

```
dataset-cleaner/
├── src/ # Source code directory
│ ├── init.py # Makes src a proper package
│ ├── app.py # Main Streamlit application
│ ├── data_cleaning.py # Core data cleaning logic
│ ├── data_visualization.py # Visualization components
│ └── utils.py # Utility functions
│
├── assets/ # Static assets
│ └── screenshots/ # Application screenshots
│ └── README.md
│
├── scripts/ # Utility scripts
│ ├── run_compatibility_test.sh # Linux/macOS compatibility test
│ └── run_compatibility_test.bat # Windows compatibility test
├── requirements/ # Separated requirements files
│ ├── base.txt # Common dependencies
│ ├── dev.txt # Development dependencies
│ └── py313.txt # Python 3.13 specific dependencies
│
├── .github/ # GitHub templates and workflows
├── LICENSE # License file
├── README.md # Project readme
├── CHANGELOG.md # Version history
├── CONTRIBUTING.md # Contribution guidelines
├── CODE_OF_CONDUCT.md # Code of conduct
├── SECURITY.md # Security policy
└── .gitignore # Git ignore file

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data visualization powered by [Plotly](https://plotly.com/)
- Data processing with [Pandas](https://pandas.pydata.org/)
- Machine learning components with [scikit-learn](https://scikit-learn.org/)
- Excel processing with [openpyxl](https://openpyxl.readthedocs.io/), [xlrd](https://xlrd.readthedocs.io/) and [xlsxwriter](https://xlsxwriter.readthedocs.io/)
