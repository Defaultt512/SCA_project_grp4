# 🥬 AgriColdChain DSS — Farm-to-Fork Cold-Chain Network Design & Stochastic Supply Planner

**Live Web Application:** [https://scagrp4project.streamlit.app/](https://scagrp4project.streamlit.app/)  
**Course:** Supply Chain Analytics (Term 5, PGDM-BDA 2026–27) — Goa Institute of Management  
**Instructor:** Prof. Prasad Patil  
**Group 4 Members:**
- **Anmol Agarwal** (`B2025008`)
- **Harsh Raj** (`B2025077`)
- **Athul Nair** (`B2025011`)
- **Shubham Maheshwari** (`B2025163`)

## Project Overview & What This App Does
India loses an estimated **11.61% of its tomato output** (and ₹1.53 lakh crore across all agricultural produce annually) due to post-harvest handling inefficiencies and a **99% deficit in farmgate packhouse pre-cooling infrastructure** (NABCONS / MoFPI; NCCD).

**AgriColdChain DSS** is an interactive decision-support web application built for a **VP of Supply Chain & Network Planning** at a B2B fresh-produce aggregator operating across the **Maharashtra – Goa – Karnataka horticulture corridor** (8 farming clusters, 6 candidate pre-cooling packhouse hubs, and 4 urban consumption hubs: Mumbai, Pune, Goa, and Bengaluru).

### Core Analytical Modules
1. **Executive Dashboard (Descriptive Analytics — CLO 3):**
   - Tracks 104 weeks (`2024-W45` to `2026-W44`) of cluster supply, urban hub demand, mandi modal prices (`₹/kg`), and the post-harvest spoilage trade-off between uncooled ambient trucks (~11.4% loss) and integrated cold-chain dispatch (~3.2% loss).
2. **Demand & Supply Forecasting (Predictive Analytics — CLO 4):**
   - Fits a **Seasonal Autoregressive (Seasonal-AR / Fourier Harmonic)** time-series model with out-of-sample holdout validation (**2.60% Demand MAPE**, **5.30% Supply MAPE**), 4–16 week multi-step forecasts, 95% confidence bands, and CSV export.
3. **Cold-Chain Network Optimizer (Prescriptive Analytics — CLO 5):**
   - Solves a **465-variable Weighted Goal Programming & Two-Stage Stochastic Mixed-Integer Linear Program (MILP)** across three weather scenarios (*Adverse Heatwave/Rain*, *Normal Monsoon*, *Bumper Glut*).
   - Identifies which candidate packhouses to open (`PH01`–`PH06`), allocates first-mile and line-haul flows across ambient vs. refrigerated vehicles, enforces spot-market traceability caps, and generates automated managerial SLA alerts and downloadable decision tables.
4. **Data Upload & Templates:**
   - Includes a pre-loaded 104-week sample dataset for zero-setup testing, a downloadable `blank_input_template.csv`, and an upload validator that checks column schemas and value ranges.
5. **Context, Strategy & Technology Outlook (CLO 1, 2, 6):**
   - Documents end-to-end material/information/financial flows, strategic fit, systems view (causal loops), mathematical formulation, and emerging technologies (IoT Cold-Chain Telemetry and Digital Twins / Agentic AI).

## How to Run the Application

### Option 1: Instant Web Access (No Installation Required)
Open the permanent Streamlit Community Cloud deployment directly in your browser:  
**[https://scagrp4project.streamlit.app/](https://scagrp4project.streamlit.app/)**

### Option 2: Run Locally from This Repository (`app.py`)
1. Clone this repository and navigate into the project folder:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open `http://localhost:8501` in your browser.

### Option 3: Run via Self-Contained Jupyter Notebook / Google Colab
Open `Agri_ColdChain_Decision_Support_App.ipynb` in **Google Colab** or **Jupyter Notebook** and click **Runtime → Run all**. The notebook automatically installs dependencies, restores the 104-week and 72-arc datasets, runs standalone model validation checks, and launches the web server.

## Repository File Structure
- `app.py` — Main Streamlit decision-support application (with built-in dataset fallback).
- `requirements.txt` — Python package dependencies for Streamlit Cloud and local execution.
- `agri_coldchain_104w_timeseries.csv` — 104-week synthesized corridor supply, demand, weather, spoilage, and price dataset calibrated to NABCONS & Agmarknet 2.0 benchmarks.
- `agri_coldchain_network_routes.csv` — 72-arc origin–destination highway distance, transit time, freight rate, and thermal spoilage matrix.
- `blank_input_template.csv` — Blank 24-column CSV template for custom data uploads.
- `Agri_ColdChain_Decision_Support_App.ipynb` — Self-contained notebook version of the application and model validation suite.
