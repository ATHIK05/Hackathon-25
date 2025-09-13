# Quick Commerce Agentic AI

## Overview
This project implements an agentic AI system for quick commerce operations, featuring:
- Data-driven sales forecasting (LightGBM, Prophet)
- Dual-role dashboard (Company & Warehouse)
- Automated action triggers via n8n (optional)
- Visual analytics and actionable insights
- Configurable business rules (config.yaml)

## Setup
1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place your data CSVs in the `data/` folder (sales.csv, inventory.csv).
3. Run the Streamlit app:
   ```
   streamlit run src/app.py
   ```

## Configuration
- Edit `src/config.yaml` to set thresholds for urgent PO, safety stock, etc.

## Project Structure
- `src/data_processing.py`: Data cleaning, aggregation
- `src/feature_engineering.py`: Rolling features, inventory merge
- `src/model.py`: LightGBM/Prophet forecasting
- `src/decision_engine.py`: Business rules, allocation, webhook payloads
- `src/app.py`: Streamlit dashboard
- `src/utils.py`: Helper functions
- `src/config.yaml`: Thresholds and parameters
- `data/`: Place your CSVs here

## Roles
- **Company**: View sales, trigger orders to warehouses
- **Warehouse**: View demand, notify company

## Output
- JSON with keys: metrics, decision, webhook_payload, rationale, visualization_instructions

## Contact
For queries, contact the project maintainer.
