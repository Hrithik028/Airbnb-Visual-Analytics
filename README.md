# Airbnb Visual Analytics (Sydney)
## Data Cleaning, Integration, and Insight Generation

## Overview
This repository presents an **end-to-end visual analytics pipeline** built on the Sydney Airbnb dataset, integrating listings, reviews, and neighbourhood boundary data. The project demonstrates practical **data engineering workflows**, **exploratory data analysis**, and **publication-quality visualisations** designed to extract actionable insights from real-world data.

The work is intentionally packaged as a **reproducible project rather than a single notebook**, allowing reviewers to inspect the full pipeline, rerun the analysis, and extend it with minimal setup.

---

## Project Objectives
The key objectives of this project are to:

- Transform raw Airbnb data into **analysis-ready datasets**
- Integrate **tabular, temporal, and geospatial data sources**
- Apply visual analytics techniques to reveal **pricing, demand, and spatial patterns**
- Demonstrate best practices in **data cleaning, validation, and reproducibility**

---

## What This Project Does

### Data Loading
- Ingests multiple raw data sources:
  - `listings.csv`
  - `reviews.csv`
  - `neighbourhoods.geojson`

### Data Cleaning & Preprocessing
- Normalises column names and data types
- Handles missing, inconsistent, and malformed values
- Converts string-encoded numeric fields (e.g. prices, percentages) to numeric formats
- Parses and validates review timestamps
- Performs schema and sanity checks to ensure data consistency

### Data Integration
- Joins Airbnb listings with neighbourhood boundary geometries
- Aggregates listings and reviews to neighbourhood-level summaries
- Derives analysis-ready variables for downstream visualisation

### Visual Analytics
The pipeline generates a set of reproducible, publication-ready figures, including:
- Price distribution by room type (violin + box plot)
- Most expensive neighbourhoods (bar chart)
- Review activity by day of week
- Weekday vs weekend review share
- Relationships between ratings, review volume, and host responsiveness
- Price differences by amenities and host portfolio size
- Spatial distribution of listings (point map)
- Average price by neighbourhood (choropleth map)

All figures are saved automatically to `results/figures/`.

---

'''
## Repository structure
├── notebooks/
│ └── analysis.ipynb # Original exploratory analysis (kept for transparency)
├── src/
│ └── run_all.py # Reproducible pipeline to generate all figures
├── results/
│ └── figures/ # Generated visualisations (PNG)
├── data/
│ └── raw/ # Expected location for raw datasets (not committed)
├── report/
│ └── project_summary.md # Short written summary of findings
├── requirements.txt
└── README.md
'''

---

## Dataset
The project expects the following files to be present in `data/raw/`:

- `listings.csv`
- `reviews.csv`
- `neighbourhoods.geojson`


---

## How to Run the Project (Recommended)

This is the preferred way to run the analysis and reproduce all figures.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the full pipeline
python -m src.run_all --data-dir data/raw --out-dir results/figures
```

## How to run (notebook)
```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

## Key Results & Insights

### Figure 01 — Price Distribution by Room Type
Private rooms exhibit lower and more concentrated price distributions, while entire homes show higher variance and higher median prices.

### Figure 03 — Review Activity by Day of Week
Review activity peaks on weekends, suggesting higher guest turnover and engagement during leisure travel periods.

### Figure 06 — Ratings vs Review Volume
Listings with high responsiveness cluster at higher ratings, though rating alone shows only a weak relationship with review volume.

### Figure 08 — Price by Amenity and Host Segment
Enterprise-scale hosts consistently command higher prices for amenity-rich listings, indicating scale and professionalisation effects.

### Figure 09 — Listing Density
Listings are spatially concentrated in coastal and inner-city neighbourhoods, reflecting demand-driven clustering.

### Figure 10 — Average Price by Neighbourhood
Clear spatial price gradients emerge across Sydney, with premium coastal and inner suburbs commanding higher nightly rates.


