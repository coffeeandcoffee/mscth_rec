cd "/Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/05-12_data_analysis_and_results/scripts"

# Make sure your environment is active
source ../venv/bin/activate

# Start a fresh run
python3 run.py

# redu last uncompleted step and continute
python3 run.py --resume 20260603_171603

# repeat prev step
python3 run.py --resume 20260603_171603 --restep

# only run next step
python3 run.py --resume 20260603_171603 --step