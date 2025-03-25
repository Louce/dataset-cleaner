from setuptools import setup, find_packages

setup(
    name="dataset-cleaner",
    version="1.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.31.0",
        "pandas>=2.1.4",
        "numpy>=1.26.3",
        "plotly>=5.18.0",
        "scikit-learn>=1.3.2",
        "openpyxl>=3.1.2",
        "xlrd>=2.0.1",
        "xlsxwriter>=3.1.1",
    ],
    entry_points={
        "console_scripts": [
            "dataset-cleaner=src.app:main",
        ],
    },
) 