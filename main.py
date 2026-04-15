import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

REBALANCE_EVERY_N_DAYS = 21
TOP_N = 5

TICKERS = [
    "AMZN", "TSLA", "HD", "MCD", "BKNG", "TJX", "LOW", "NKE",
    "SBUX", "CMG", "ROST", "ORLY", "MAR", "YUM", "HLT", "EBAY",
    "AZO", "DHI", "LEN", "RCL", "CCL", "LVS", "ULTA", "DPZ"
]

BENCHMARK = "XLY"
RISK_FREE_RATE = 0.0


def download_prices(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False
    )["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    return data.dropna(axis=1, how="all")


def compute_stock_signals(price_df):
    signal_dict = {}

    for ticker in price_df.columns:
        s = price_df[ticker].dropna().copy()

        df = pd.DataFrame(index=s.index)
        df["close"] = s
        df["ma50"] = s.rolling(50).mean()
        df["ret_63"] = s.pct_change(63)   # about 3 months
        df["ret_126"] = s.pct_change(126) # about 6 months

        signal_dict[ticker] = df

    return signal_dict


def build_rank_table(signal_dict):
    all_dates = sorted(set().union(*[df.index for df in signal_dict.values()]))
    rank_table = pd.DataFrame(index=all_dates)

    for ticker, df in signal_dict.items():
        temp = df.copy()

        # only consider stocks above 50-day MA
        valid_trend = temp["close"] > temp["ma50"]

        # combine 3-month and 6-month momentum
        temp["score"] = np.where(
            valid_trend,
            0.5 * temp["ret_63"] + 0.5 * temp["ret_126"],
            np.nan
        )

        rank_table[ticker] = temp["score"]

    return rank_table.sort_index()


def generate_rebalance_dates(price_index, step):
    return price_index[::step]


def choose_portfolio(scores_on_date, top_n):
    valid = scores_on_date.dropna().sort_values(ascending=False)
    selected = valid.head(top_n)

    if len(selected) == 0:
        return {}

    weight = 1.0 / len(selected)
    return {ticker: weight for ticker in selected.index}


def run_backtest(price_df, rank_table, benchmark_df, rebalance_every_n_days=21, top_n=5):
    daily_returns = price_df.pct_change().fillna(0)
    portfolio_returns = pd.Series(0.0, index=price_df.index)
    rebalance_dates = generate_rebalance_dates(price_df.index, rebalance_every_n_days)

    current_weights = {}
    rebalance_log = []

    for date in price_df.index:
        xly_price = benchmark_df.loc[date, "XLY"]
        xly_ma200 = benchmark_df.loc[date, "ma200"]

        # allow trading before MA200 exists
        market_good = pd.isna(xly_ma200) or (xly_price > xly_ma200)

        if date in rebalance_dates:
            if market_good:
                scores_today = rank_table.loc[date].reindex(price_df.columns)
                current_weights = choose_portfolio(scores_today, top_n)
            else:
                current_weights = {}

            rebalance_log.append({
                "date": date,
                "market_good": market_good,
                "holdings": list(current_weights.keys())
            })

        if current_weights:
            day_ret = 0.0
            for ticker, weight in current_weights.items():
                if ticker in daily_returns.columns and pd.notna(daily_returns.loc[date, ticker]):
                    day_ret += weight * daily_returns.loc[date, ticker]
            portfolio_returns.loc[date] = day_ret
        else:
            portfolio_returns.loc[date] = 0.0

    return portfolio_returns, rebalance_log


def compute_omega_ratio(returns, threshold=0.0):
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()

    if losses == 0:
        return np.inf

    return gains / losses


@dataclass
class PerformanceMetrics:
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    omega_ratio: float
    max_drawdown: float


def compute_metrics(daily_returns, risk_free_rate=0.0):
    cumulative_curve = (1 + daily_returns).cumprod()
    total_return = cumulative_curve.iloc[-1] - 1

    n_days = len(daily_returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    annualized_volatility = daily_returns.std() * np.sqrt(252)

    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf

    if daily_returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = excess_returns.mean() / daily_returns.std() * np.sqrt(252)

    omega = compute_omega_ratio(daily_returns)

    rolling_max = cumulative_curve.cummax()
    drawdown = (cumulative_curve / rolling_max) - 1
    max_drawdown = drawdown.min()

    return PerformanceMetrics(
        cumulative_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe,
        omega_ratio=omega,
        max_drawdown=max_drawdown
    )


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Cumulative Return:   {metrics.cumulative_return:.2%}")
    print(f"Annualized Return:   {metrics.annualized_return:.2%}")
    print(f"Annualized Vol:      {metrics.annualized_volatility:.2%}")
    print(f"Sharpe Ratio:        {metrics.sharpe_ratio:.3f}")
    print(f"Omega Ratio:         {metrics.omega_ratio:.3f}")
    print(f"Max Drawdown:        {metrics.max_drawdown:.2%}")


def main():
    price_df = download_prices(TICKERS, START_DATE, END_DATE)
    benchmark_df = download_prices([BENCHMARK], START_DATE, END_DATE)
    benchmark_df = benchmark_df.rename(columns={BENCHMARK: "XLY"})
    benchmark_df["ma200"] = benchmark_df["XLY"].rolling(200).mean()

    benchmark_df = benchmark_df.reindex(price_df.index).ffill()

    signal_dict = compute_stock_signals(price_df)
    rank_table = build_rank_table(signal_dict).reindex(price_df.index)

    strategy_returns, rebalance_log = run_backtest(
        price_df,
        rank_table,
        benchmark_df,
        rebalance_every_n_days=REBALANCE_EVERY_N_DAYS,
        top_n=TOP_N
    )

    benchmark_returns = benchmark_df["XLY"].pct_change().fillna(0)

    strategy_metrics = compute_metrics(strategy_returns, risk_free_rate=RISK_FREE_RATE)
    benchmark_metrics = compute_metrics(benchmark_returns, risk_free_rate=RISK_FREE_RATE)

    print_metrics("Strategy", strategy_metrics)
    print_metrics("Benchmark (XLY)", benchmark_metrics)

    strategy_curve = (1 + strategy_returns).cumprod()
    benchmark_curve = (1 + benchmark_returns).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(strategy_curve.index, strategy_curve.values, label="Strategy")
    plt.plot(benchmark_curve.index, benchmark_curve.values, label="XLY")
    plt.title("Trend-Following Consumer Discretionary Strategy vs XLY")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    invested_days = (strategy_returns != 0).sum()
    total_days = len(strategy_returns)
    print(f"\nInvested {100 * invested_days / total_days:.2f}% of trading days.")

    print("\nSample Rebalance Decisions:")
    for entry in rebalance_log[:10]:
        print(f"{entry['date'].date()} | Market Good: {entry['market_good']} | Holdings: {entry['holdings']}")


if __name__ == "__main__":
    main()