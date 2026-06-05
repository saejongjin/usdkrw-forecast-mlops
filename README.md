# USD/KRW Forecast MLOps

## Overview
USD/KRW forecasting preprocessing pipeline.

Uses:
- BTC-USD
- KRW=X
- JPY=X
- CL=F (WTI)
- GC=F (GOLD)
- DX-Y.NYB (DXY)

Processes:
- 1 minute market data collection
- UTC timestamp alignment
- common rolling index generation
- interpolation
- CSV save
- PostgreSQL save

---

## Project Structure

txt src/preprocess.py data/processed/market_data_1m_24h_interpolated.csv 

---
docker environment build : 

docker compose up

---

## Run

Project root:

bash python3 src/preprocess.py 

---

## Output

CSV:

txt data/processed/market_data_1m_24h_interpolated.csv 

PostgreSQL:

txt marketdb.latest_24h_market_data_1m 

---

## DB Connection

txt postgresql://admin:admin@localhost:5432/marketdb 

---

## Notes

- Recent valid rolling 1440 minutes used
- Weekend inactive period excluded:
  - Friday 22:00 UTC ~ Sunday 22:00 UTC
- Interpolation applied for small gaps

Market characteristics:
- BTC: almost 24/7
- KRW/JPY: active during most FX market hours
- WTI/GOLD: short maintenance periods exist
- DXY: intermittent gaps may occur