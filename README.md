# Global Volatility Spillovers to the Indian Stock Market During COVID-19

This repository contains the complete econometric data pipeline and modelling framework used to analyze structural breaks and volatility transmission from global equity markets to the Indian BSE Sensex during the COVID-19 pandemic.

## 📊 Overview
This project mathematically demonstrates how the COVID-19 exogenous shock structurally altered the financial integration of the Indian equity market. Using daily adjusted closing prices from January 2019 to December 2021, the analysis tracks the transmission of returns and volatility from four major indices (S&P 500, DAX, Nikkei 225, SSE Composite) to the Sensex.

**Key Findings:**
- **The Integration Paradox:** Pre-COVID, the Indian market was largely insulated from Asian regional shocks. Post-COVID, transmission channels from the Chinese SSE Composite "awoke," while integration with Western markets (S&P 500, DAX) aggressively tightened.
- **The "Leverage Effect":** The EGARCH models confirmed that negative shocks generated significantly higher volatility than positive shocks of equal magnitude, proving the existence of asymmetric panic in the Indian market.
- **Failure of Symmetry:** Standard linear models (OLS) and symmetric variance models (GARCH) structurally broke down during the pandemic crash, requiring the transition to asymmetric EGARCH modelling to capture the extreme variance.

## 🛠 Methodology Pipeline
1. **Data Preprocessing:** Forward-filling non-trading days to maintain an intact chronological time-series for autoregressive variance updating.
2. **Structural Break Testing:** Using the endogenous Zivot-Andrews Unit Root Test to mathematically identify the March 23, 2020 crash without arbitrarily assigning dates.
3. **Linear Baselines:** Running OLS regressions and conducting Breusch-Godfrey & ARCH-LM diagnostics to prove the existence of conditional heteroskedasticity.
4. **ARIMA Filtering:** Fitting ARMA(p,q) models to filter linear predictability and extract independent residual shocks.
5. **GARCH / EGARCH Modelling:** Estimating symmetric GARCH(1,1) baselines and advanced EGARCH models to capture asymmetric volatility clustering and extract the `h_t` conditional variance series.

## 🚀 Running the Project

### Prerequisites
- Python 3.9+
- `pip install -r requirements.txt`

### Step 1: Collect Data
Run the data script to fetch high-frequency daily data via Yahoo Finance.
```bash
python data_collection.py
```
This generates `adjusted_close_prices.csv` and `log_returns.csv` in the `data/` folder.

### Step 2: Execute Econometric Pipeline
Run the main pipeline to perform all statistical tests, estimate the ARIMA-GARCH-EGARCH models, and generate the heatmaps.
```bash
python run_pipeline.py
```
All statistical summaries (ADF, OLS diagnostics, EGARCH parameters) and visual plots (Time Series, Correlation Heatmaps) will be saved in the `outputs/` directory.

## 📂 Repository Structure
- `data_collection.py` - Fetches and cleans financial time-series data.
- `run_pipeline.py` - Main script executing the econometric analysis.
- `requirements.txt` - Python dependencies (`arch`, `statsmodels`, `pmdarima`, `yfinance`, etc.)
- `data/` - Contains the raw and processed CSV datasets.
- `outputs/` - Generated plots, diagnostics, and model estimation summaries.

## 👤 Author
**Dhruv Singh**
*(Bachelor Thesis Project - 2)*
