# CSV Dataset Cleaner

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

An interactive data cleaning platform for CSV datasets with intelligent issue detection and automated cleaning solutions. This application allows you to upload any CSV file, automatically identify data quality issues, and clean your data through an intuitive interface.

<div align="center">
  <img src="https://raw.githubusercontent.com/Louce/csv-dataset-cleaner/master/sample_data/Screenshot.png" alt="CSV Dataset Cleaner Screenshot" width="80%">
</div>

## 🚀 Features

- **Dynamic Dataset Handling**: Works with any CSV file structure without hardcoded column references
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
- **Flexible Export Options**: CSV, Excel, JSON, and Pickle formats

## 📋 Requirements

- Python 3.8+ (fully compatible with Python 3.13)
- Dependencies listed in `requirements.txt`

## 💻 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Louce/csv-dataset-cleaner.git
   cd csv-dataset-cleaner
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
   # For Python 3.13 (recommended)
   pip install -r requirements.txt
   
   # For Python 3.12 (alternative)
   pip install -r requirements-py312.txt
   ```

4. Launch the application:
   ```bash
   streamlit run app.py
   ```

## 🌐 Live Demo

To deploy a live demo using Streamlit Cloud:

1. Fork this repository to your GitHub account
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app in Streamlit Cloud
4. Enter your GitHub repository URL followed by the path to app.py:
   ```
   https://github.com/Louce/csv-dataset-cleaner/blob/master/app.py
   ```
5. In the advanced settings, set the Python version to 3.13
   (If Python 3.13 is not yet supported by Streamlit Cloud, use Python 3.12 instead)
6. Change the requirements file path to:
   ```
   # For Python 3.13 (if supported)
   requirements-streamlit.txt
   
   # For Python 3.12 (if 3.13 is not supported)
   requirements-py312.txt
   ```
7. Deploy and share your application!

## 📊 Usage Workflow

1. **Upload** your CSV file with optional import settings
2. **Analyze** your dataset with automated profiling
3. **Fix Missing Values** with intelligent strategies
4. **Handle Outliers** with precision
5. **Check Data Consistency** to identify logical errors
6. **Review Changes** with interactive visualizations
7. **Export** your cleaned dataset

## 🧰 Advanced Features

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
   pip install -r requirements-dev.txt
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

- Windows: `run_compatibility_test.bat`
- Linux/macOS: `run_compatibility_test.sh`

## 📂 Project Structure

```
csv-dataset-cleaner/
├── app.py                    # Main Streamlit application
├── data_cleaning.py          # Core data cleaning logic
├── data_visualization.py     # Visualization components
├── utils.py                  # Utility functions
├── requirements.txt          # Project dependencies
├── requirements-dev.txt      # Development dependencies
├── .github/                  # GitHub templates and workflows
├── sample_data/              # Sample datasets for testing
├── CHANGELOG.md              # Version history
├── CODE_OF_CONDUCT.md        # Community guidelines
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # MIT license
├── README.md                 # This file
└── SECURITY.md               # Security policy
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data visualization powered by [Plotly](https://plotly.com/)
- Data processing with [Pandas](https://pandas.pydata.org/)
- Machine learning components with [scikit-learn](https://scikit-learn.org/)