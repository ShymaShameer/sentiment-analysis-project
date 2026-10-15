#This is the new name of the the ReadMe: README_SHYMA




<<<<<<< HEAD
# sentiment-analysis-project
sentiment-analysis-project

#this is a test
=======
# Sentiment Analysis Pipeline

## Setup

### Option 1: Python venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### Option 2: Conda
conda create -n sentiment-env python=3.11 -y
conda activate sentiment-env
pip install -r requirements.txt

## Train
python src/train.py --data data/train.csv --out models/sentiment.joblib

## Predict
python src/predict.py "I absolutely loved it" "That was awful"



#Adding a line at the end by Julien

#Adding another line 15:05

#Adding another line 15:22

#Adding another line 15:40