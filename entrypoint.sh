#!/bin/bash
set -e

DATA_DIR="/app/data"
DATASET_DIR="$DATA_DIR/LJSpeech-1.1"
DATA_URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

mkdir -p $DATA_DIR
cd $DATA_DIR

if [ ! -d "$DATASET_DIR" ]; then
    echo "Downloading LJSpeech dataset..."
    wget $DATA_URL
    tar -xvjf LJSpeech-1.1.tar.bz2
    rm LJSpeech-1.1.tar.bz2
else
    echo "Dataset already exists. Skipping download."
fi

cd /app

echo "Starting Training..."
python -m src.train.train