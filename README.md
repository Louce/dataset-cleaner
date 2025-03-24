# CSV Dataset Cleaner

A powerful and user-friendly web application for cleaning and analyzing CSV datasets. Built with Streamlit, this tool provides an intuitive interface for data cleaning, visualization, and export.

## Features

- 📊 **Interactive Data Visualization**
  - Histograms for numeric columns
  - Bar charts for categorical columns
  - Box plots for outlier detection
  - Before/After comparisons

- 🧹 **Smart Data Cleaning**
  - Automatic missing value handling
  - Outlier detection and treatment
  - Categorical variable standardization
  - Data consistency checks
  - Duplicate removal

- 📈 **Data Analysis**
  - Statistical summaries
  - Data quality metrics
  - Column information
  - Data type analysis

- 💾 **Export Options**
  - CSV export
  - Excel export
  - Custom file naming

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/csv-dataset-cleaner.git
cd csv-dataset-cleaner
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload your CSV file using the file uploader

4. Use the navigation sidebar to:
   - View the original dataset
   - Clean the data
   - View data quality reports
   - Generate statistical summaries
   - Export the cleaned data

## Project Structure

```
csv-dataset-cleaner/
├── app.py                 # Main Streamlit application
├── data_cleaning.py       # Data cleaning logic
├── data_visualization.py  # Visualization functions
├── utils.py              # Utility functions
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Dependencies

- streamlit>=1.32.0
- pandas>=2.2.1
- numpy>=1.26.4
- matplotlib>=3.8.3
- plotly>=5.19.0
- openpyxl>=3.1.2
- scipy>=1.12.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data visualization powered by [Plotly](https://plotly.com/)
- Data processing with [Pandas](https://pandas.pydata.org/)