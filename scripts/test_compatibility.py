"""
Test script to verify compatibility with Python 3.13 and pandas 2.x
"""
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import logging
from datetime import datetime

def print_version_info():
    """Print version information for key packages"""
    print(f"Python version: {sys.version}")
    print(f"pandas version: {pd.__version__}")
    print(f"numpy version: {np.__version__}")
    try:
        import streamlit
        print(f"streamlit version: {streamlit.__version__}")
    except ImportError:
        print("streamlit not installed")
    
    try:
        import plotly
        print(f"plotly version: {plotly.__version__}")
    except ImportError:
        print("plotly not installed")

def test_pandas_compatibility():
    """Test pandas compatibility with newer versions"""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'numeric': [1, 2, None, 4, 5],
        'text': ['a', 'b', None, 'd', 'e'],
        'date': pd.date_range('2023-01-01', periods=5, freq='D'),
        'category': pd.Categorical(['x', 'y', 'z', 'x', None])
    })
    
    print("\nTesting pandas compatibility...")
    
    # Test missing value functions
    missing_count = df.isna().sum()
    print(f"Missing value count: {missing_count}")
    
    # Test data type handling
    print(f"Data types: {df.dtypes}")
    
    # Test categorical handling
    print(f"Categorical unique values: {df['category'].unique()}")
    
    # Test datetime handling
    print(f"Date min: {df['date'].min()}")
    
    # Test type checking functions
    print(f"Is numeric column numeric? {pd.api.types.is_numeric_dtype(df['numeric'])}")
    print(f"Is category column categorical? {pd.api.types.is_categorical_dtype(df['category'])}")
    
    print("Basic pandas compatibility tests passed!")

def test_streamlit_compatibility():
    """Test streamlit compatibility"""
    print("\nPlease run this script with 'streamlit run test_compatibility.py' to test Streamlit compatibility")
    print("This test cannot be run directly from Python")
    
    if 'streamlit.runtime.scriptrunner' in sys.modules:
        st.title("Streamlit Compatibility Test")
        st.write("If you can see this, Streamlit is working correctly!")
        
        # Test basic widgets
        st.write("Testing basic widgets:")
        if st.button("Test Button"):
            st.write("Button clicked!")
        
        option = st.selectbox("Test Selectbox", ["Option 1", "Option 2", "Option 3"])
        st.write(f"Selected: {option}")
        
        # Test DataFrame display
        st.write("Testing DataFrame display:")
        df = pd.DataFrame({
            'numeric': [1, 2, None, 4, 5],
            'text': ['a', 'b', None, 'd', 'e']
        })
        st.dataframe(df)
        
        # Test plotly integration
        st.write("Testing Plotly integration:")
        fig = px.bar(df, x=df.index, y='numeric', title="Test Plot")
        st.plotly_chart(fig)
        
        st.success("All Streamlit compatibility tests passed!")

if __name__ == "__main__":
    print_version_info()
    test_pandas_compatibility()
    test_streamlit_compatibility() 