# SMLDLProject
# Public Transport Occupancy Prediction Pipeline

This repository contains an end-to-end machine learning pipeline for predicting public transport crowding levels (e.g. EMPTY vs CROWDED) for specific bus routes in Skåne, Sweden.

The system ingests GTFS-Realtime data from the KoDa API, synchronizes it with GTFS-Static schedules, enriches the data with weather and holiday context, and trains a Random Forest classifier to predict hourly occupancy levels.

The pipeline is designed to be reproducible, robust to missing data, and suitable for multi-year longitudinal analysis.

---

## Project Structure

kodaProcessor.py  
Core engine. Handles API authentication, resilient downloading, archive extraction, and trip_id → route_id mapping.


select_route.py  
Used to find the best route for our model. Using the data from a random day and the trip to route mapping we compute entropy to find the route with more diverse data as possible

verify_route.py
We used this file to verify the diversity of the route on one month data

run_year_route_dataset.py  
Downloads one year of GTFS-Realtime data for a given route, maps trips to routes, samples snapshots, and aggregates occupancy into hourly bins.

mergedatasets.py  
Combines multiple yearly CSV files into a single  dataset. 

build_final_dataset.py  
Feature engineering stage. Enriches occupancy data with weather data (Open-Meteo), Swedish holidays, and calendar-based features.

train.py  
Trains a class-weighted Random Forest classifier, evaluates performance using temporal splits, and outputs F1 scores and confusion matrices.

one_day_aggr.py  
Lightweight prototype script for testing the pipeline on a single day of data.

CountProperties.py  
Simple EDA utility to count rows by month, weekday, or other calendar dimensions.

Generated datasets are written to the out/ directory.

---

## Requirements

- Python 3.11
- KoDa API key from Trafiklab
- Internet access (KoDa API and Open-Meteo API)

### Python Dependencies
This project uses `uv` for fast dependecy management. Run the following command to create a virtual environment and sync dependencies from `pyproject.toml`:

```
uv sync
```

Alternatively, using standard pip with `requirements.txt`:
```
pip install -r requirements.txt
```

## Environment Setup

The KoDa API key must be provided as an environment variable.

Windows (PowerShell):

$env:KODA_API_KEY="your_key_here"

macOS / Linux:

export KODA_API_KEY="your_key_here"

---

## Data Pipeline

### Phase 1: Data Extraction and Mapping
Script: run_year_route_dataset.py

This stage downloads and aggregates GTFS-Realtime data for a single route and date range.

Inputs:
- Date range (e.g. 2024-10-01 to 2025-08-31)
- Route ID

Processing steps:
- Downloads GTFS-Static files for each relevant month
- Parses trips.txt to build a trip_id → route_id lookup table
- Downloads daily GTFS-Realtime protobuf archives
- Samples 10 evenly spaced snapshots per hour
- Aggregates occupancy status into a single hourly label using the mode

Output:
out/skane_<start>_<end>_route_<route_id>_hourly.csv

---

### Phase 2: Dataset Merging
Script: mergedatasets.py

Because KoDa data is large, extraction is typically performed in yearly chunks.

Processing steps:
- Merges multiple yearly CSV files
- Deduplicates overlapping rows
- Prioritizes rows with a higher number of observations

Output:
out/merged_<N>years_route_<route_id>_hourly.csv

---

### Phase 3: Feature Engineering
Script: build_final_dataset.py

Prepares the dataset for machine learning.

Features added:
- Weather data from Open-Meteo:
  - Temperature
  - Rain
  - Wind speed
- Calendar context:
  - is_holiday_se
  - is_weekend
  - time_of_day (e.g. morning, rush hour, evening)
- Target encoding:
  - Sparse occupancy labels are collapsed into broader classes
  - Example: STANDING_ROOM_ONLY and FULL are mapped to CROWDED

---

### Phase 4: Model Training
Script: train.py

Model:
- Random Forest Classifier
- Class-weighted to handle label imbalance

Evaluation:
- Temporal train/validation/test split to avoid data leakage
- Macro F1-score and confusion matrix are reported

---

## Route Discovery

Script: select_route.py

This script analyzes a sample of data to identify routes with:
- Sufficient temporal coverage
- High label entropy

It helps avoid training models on routes with sparse or uninformative occupancy data.

---

## Sampling Strategy

GTFS-Realtime data is highly frequent and noisy.

Instead of processing every snapshot, the pipeline:
- Samples 10 evenly spaced observations per hour
- Aggregates them into a single hourly label using the most frequent occupancy state

This reduces noise while keeping the dataset statistically meaningful.

---

## Results
the best results were obtained by using all the time related data, but without the weather data.

## Limitations
- Occupancy labels are inferred from vehicle reports, not direct passenger counts
- Weather data is regional (Skåne-level), not stop-specific
- Models are route-specific and require retraining for new routes
- Data availability depends on KoDa coverage and operator reporting quality

---

## License

This project is provided for research and educational purposes.
Use of KoDa and Open-Meteo data is subject to their respective terms of service.
