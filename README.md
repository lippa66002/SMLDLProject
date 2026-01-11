# Skåne Public Transport Occupancy Prediction

A machine learning pipeline for predicting occupancy levels on Skånetrafiken buses. This end-to-end system ingests GTFS-Realtime data, enriches it with weather and calendar context, trains LightGBM models, and serves predictions through an interactive dashboard.

## Live Dashboard

**[View the Dashboard](https://lippa66002.github.io/SMLDLProject/)** - Updated daily with predictions for all Skåne bus routes.

## Project Structure

```
SMLDLProject/
├── src/                        # All Python source code
│   ├── pipelines/              # Main executable scripts
│   │   ├── feature_pipeline.py         # Daily feature ingestion
│   │   ├── batch_inference_pipeline.py # Dashboard data generation
│   │   ├── train_avg_occupancy.py      # Train avg occupancy model
│   │   ├── train_max_occupancy.py      # Train max occupancy model
│   │   ├── download_traffic_data.py    # Historical data download
│   │   └── upload_features.py          # Backfill Hopsworks
│   │
│   ├── utils/                  # Shared utility modules
│   │   ├── koda_processor.py   # KoDa GTFS-RT API client
│   │   ├── gtfs_schedule.py    # GTFS schedule parsing
│   │   └── transformations/    # Data transformation functions
│   │
│   └── hopsworks_jobs/         # Server-side Spark scripts
│       ├── get_existing_dates.py
│       └── export_past_traffic.py
│
├── tests/                      # Unit tests
├── docs/                       # Dashboard (static HTML/JS)
│   ├── index.html
│   ├── style.css
│   └── dashboard_data.json     # Generated predictions
├── .github/workflows/          # CI/CD automation
└── pyproject.toml              # Project configuration
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Hopsworks account with API key
- KoDa API key (for historical GTFS-RT data)
- GTFS Regional API key (for schedule data)

### Setup

```bash
# Clone repository
git clone https://github.com/lippa66002/SMLDLProject.git
cd SMLDLProject

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt


# Edit .env with your API keys:
#   HOPSWORKS_API_KEY=your_key
#   KODA_API_KEY=your_key
#   GTFS_REGIONAL_API_KEY=your_key
```

## Pipelines

### Daily Feature Pipeline

Fetches and processes traffic data from KoDa, weather from Open-Meteo, and calendar features.

```bash
uv run feature-pipeline
```

Runs automatically via GitHub Actions daily at 6:00 AM UTC.

### Batch Inference Pipeline

Generates predictions for the dashboard (3 past days + today + 3 future days).

```bash
uv run inference-pipeline
```

Runs automatically via GitHub Actions daily at 8:00 AM UTC, after feature pipeline.

### Training Pipelines

Train or retrain the LightGBM models:

```bash
# Average occupancy (regression)
uv run train-avg

# Max occupancy (classification)
uv run train-max
```

### Historical Data Backfill

Download historical traffic data from KoDa:

```bash
# Download specific date range
uv run download-traffic --start 2025-10-01 --end 2025-10-31

# Upload to Hopsworks
uv run upload-features
```

## Data Sources

| Source | API | Description |
|--------|-----|-------------|
| **KoDa** | GTFS-Realtime | VehiclePositions with occupancy status from Skånetrafiken |
| **Open-Meteo** | Archive + Forecast | Hourly weather (temperature, precipitation, wind, clouds) |
| **holidays** | Python package | Swedish public holidays |

## Feature Groups (Hopsworks)

| Name | Version | Primary Key | Description |
|------|---------|-------------|-------------|
| `skane_traffic` | 2 | `route_id, direction_id, date, hour` | Aggregated hourly occupancy counts |
| `skane_weather` | 2 | `date, hour` | Weather + previous day values |
| `sweden_calendar` | 2 | `date` | Holidays, weekday, workday flags |

## Models

| Model | Type | Target | Registry Name |
|-------|------|--------|---------------|
| Avg Occupancy | LightGBM Regressor | `avg_occupancy` (0-5 float) | `occupancy_lgbm_clipped_avg_occupancy` |
| Max Occupancy | LightGBM Classifier | `max_occupancy` (0-5 class) | `occupancy_lgbm_classifier_max_occupancy` |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_transformations.py -v
```

## License

This project was developed as part of the ID2223 course at KTH Royal Institute of Technology.
