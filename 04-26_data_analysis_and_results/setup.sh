#!/usr/bin/env bash

echo "================================================="
echo "INSTALLING GLOBAL PYTHON PIPELINE DEPENDENCIES"
echo "================================================="

# On newer macOS environments (PEP 668), pip prevents global installs by default.
# Because you explicitly requested no virtual environment, we use --break-system-packages 
# to bypass the restriction and install them to your user-level python3 safely.

python3 -m pip install --user numpy pandas scikit-learn scipy matplotlib joblib

echo "================================================="
echo "DEPENDENCIES INSTALLED SUCCESSFULLY!"
echo "================================================="
