#!/bin/bash

echo "Running compatibility tests for CSV Dataset Cleaner"
echo
echo "Make sure Python 3.13 and dependencies are installed"
echo

python3 -c "import sys; print(f'Python version: {sys.version}')"

echo
echo "Step 1: Testing basic Python compatibility"
python3 test_compatibility.py

echo
echo "Step 2: Testing Streamlit compatibility"
echo "This will open Streamlit in your browser. Close the browser when done."
echo "Press Ctrl+C to exit the Streamlit server when testing is complete."
sleep 3
streamlit run test_compatibility.py

echo
echo "Tests complete!"
echo "If any issues were encountered, check error messages and update code as needed." 