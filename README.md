# Consumer Discretionary Quant Strategy

## Overview
Quant strategy using:
- Momentum ranking (3M + 6M returns)
- Market regime filter (XLY 200-day MA)

## Metrics
- Evaluates Sharpe, Omega, drawdown
- Benchmarked against XLY

## How to Run
```bash
pip install pandas numpy matplotlib yfinance
python main.py