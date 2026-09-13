"""Backtest a simple SPY volatility-aware exposure strategy against buy-and-hold."""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


TICKER = "SPY"
START_DATE = "2023-01-01"
END_DATE = "2026-06-05"
SHORT_WINDOW = 20
LONG_WINDOW = 60
TRADING_DAYS = 252

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def annualized_return(returns: pd.Series) -> float:
    return (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series) -> float:
    volatility = annualized_volatility(returns)
    return np.nan if volatility == 0 else annualized_return(returns) / volatility


def max_drawdown(cumulative_returns: pd.Series) -> float:
    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    return drawdown.min()


def download_and_prepare_data() -> pd.DataFrame:
    """Download prices and construct volatility, signal, and return columns."""
    spy = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=False)

    if spy.empty:
        raise RuntimeError("No SPY data was downloaded.")

    # yfinance may return a two-level column index even for one ticker.
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy = spy.dropna(subset=["Close"]).copy()
    spy["Daily_Return"] = spy["Close"].pct_change()
    spy["Volatility_20D"] = spy["Daily_Return"].rolling(SHORT_WINDOW).std()
    spy["Volatility_60D_Avg"] = spy["Volatility_20D"].rolling(LONG_WINDOW).mean()
    spy = spy.dropna().copy()

    spy["Signal"] = np.where(
        spy["Volatility_20D"] <= spy["Volatility_60D_Avg"], 1, 0
    )
    spy["Buy_Hold_Return"] = spy["Daily_Return"]

    # Yesterday's signal is used to prevent look-ahead bias.
    spy["Strategy_Return"] = spy["Signal"].shift(1) * spy["Daily_Return"]
    spy = spy.dropna().copy()
    spy["Buy_Hold_Cumulative"] = (1 + spy["Buy_Hold_Return"]).cumprod()
    spy["Strategy_Cumulative"] = (1 + spy["Strategy_Return"]).cumprod()
    return spy


def calculate_metrics(spy: pd.DataFrame) -> pd.DataFrame:
    """Calculate comparable return and risk metrics for both strategies."""
    return pd.DataFrame(
        {
            "Buy-and-Hold": [
                annualized_return(spy["Buy_Hold_Return"]),
                annualized_volatility(spy["Buy_Hold_Return"]),
                sharpe_ratio(spy["Buy_Hold_Return"]),
                max_drawdown(spy["Buy_Hold_Cumulative"]),
            ],
            "Volatility-Aware Strategy": [+                annualized_return(spy["Strategy_Return"]),
                annualized_volatility(spy["Strategy_Return"]),
                sharpe_ratio(spy["Strategy_Return"]),
                max_drawdown(spy["Strategy_Cumulative"]),
            ],
        },
        index=[
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
        ],
    )


def save_figures(spy: pd.DataFrame) -> None:
    """Save the cumulative-return and high-volatility-period charts."""
    plt.figure(figsize=(12, 6))
    plt.plot(spy.index, spy["Buy_Hold_Cumulative"], label="Buy-and-Hold")
    plt.plot(spy.index, spy["Strategy_Cumulative"], label="Volatility-Aware Strategy")
    plt.title("SPY Buy-and-Hold vs Volatility-Aware Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cumulative_returns_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(spy.index, spy["Close"], label="SPY Close Price")
    plt.fill_between(
        spy.index,
        spy["Close"].min(),
        spy["Close"].max(),
        where=spy["Signal"] == 0,
        alpha=0.2,
        label="High-volatility period (Signal = 0)",
    )
    plt.title("SPY Price with High-Volatility Periods")
    plt.xlabel("Date")
    plt.ylabel("SPY Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "spy_price_high_volatility_periods.png", dpi=300)
    plt.close()


def save_ai_interpretation(formatted_metrics: pd.DataFrame) -> None:
    """Optionally request a short interpretation when explicitly enabled."""
    if os.getenv("GENERATE_AI_INTERPRETATION") != "1":
        return

    from openai import OpenAI

    prompt = f"""You are helping write a concise quantitative finance report.

Compare Buy-and-Hold with a Volatility-Aware SPY Strategy using these metrics:
{formatted_metrics.to_string()}

In two short, beginner-friendly paragraphs, compare return, volatility, Sharpe
ratio, and maximum drawdown. Explain the risk-control trade-off. Do not give
financial advice.
"""
    response = OpenAI().responses.create(model="gpt-5", input=prompt)
    (RESULTS_DIR / "ai_interpretation.txt").write_text(
        response.output_text, encoding="utf-8"
    )


def main() -> None:
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    spy = download_and_prepare_data()
    metrics = calculate_metrics(spy)
    formatted = metrics.copy().astype(object)

    for row in ("Annualized Return", "Annualized Volatility", "Max Drawdown"):
        formatted.loc[row] = metrics.loc[row].map(lambda value: f"{value:.2%}")
    formatted.loc["Sharpe Ratio"] = metrics.loc["Sharpe Ratio"].map(
        lambda value: f"{value:.2f}"
    )

    spy.to_csv(DATA_DIR / "spy_backtest_data.csv")
    metrics.to_csv(RESULTS_DIR / "performance_metrics.csv")
    formatted.to_csv(RESULTS_DIR / "formatted_performance_metrics.csv")
    save_figures(spy)
    save_ai_interpretation(formatted)

    print("Backtest complete.\n")
    print(formatted)


if __name__ == "__main__":
    main()
