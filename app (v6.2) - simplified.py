from __future__ import annotations

import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from jugaad_data.nse import index_df as jugaad_index_df, index_tri_raw as jugaad_index_tri_raw
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier


# Page setup & styling
st.set_page_config(page_title="Factor Investing in India", page_icon="💰", layout="wide")

st._config.set_option(f'theme.backgroundColor' ,"white" )
st._config.set_option(f'theme.base' ,"light" )
st._config.set_option(f'theme.primaryColor' ,"#FFBF00" )
st._config.set_option(f'theme.textColor' ,"#392D00")
st._config.set_option(f'theme.font' ,"serif")

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TRADING_DAYS = 252

# Index universe for each factor.
FACTOR_INDEX_OPTIONS: dict[str, tuple[str, ...]] = {
    "Alpha": ("Nifty Alpha 50", "Nifty100 Alpha 30", "Nifty200 Alpha 30"),
    "Momentum": ("Nifty Midcap150 Momentum 50", "Nifty500 Momentum 50"),
    "Quality": ("Nifty500 Quality 50", "Nifty Midcap150 Quality 50", "Nifty Smallcap250 Quality 50"),
    "Value": ("Nifty50 Value 20", "Nifty200 Value 30", "Nifty500 Value 50"),
    "Volatility": ("Nifty Low Volatility 50",),
}


def stream_text(text: str):
    """Yield words with a small delay, for st.write_stream-style typing effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


# Data fetching 
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_index_data_pr(index_name: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetch historical Price Return (PR) data for an index. Tries
    jugaad-data first (actively maintained, chunks & parallelises long date
    ranges); falls back to a direct niftyindices.com call if that fails."""
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    try:
        df = jugaad_index_df(symbol=index_name, from_date=start, to_date=end)
        df["HistoricalDate"] = pd.to_datetime(df["HistoricalDate"])
        df["CLOSE"] = df["CLOSE"].astype(float)
        df = df.set_index("HistoricalDate").sort_index()
        return df
    except Exception as jugaad_error:
        st.warning(
            f"Could not fetch data for {index_name}. "
            f"jugaad-data error: {jugaad_error}."
        )
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_index_data_tr(index_name: str, start_date: str, end_date: str) -> pd.DataFrame | None:    
    try:
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        records = jugaad_index_tri_raw(name=index_name, index_name=index_name, from_date=start, to_date=end)
        df = pd.DataFrame.from_records(records)
        df["Date"] = pd.to_datetime(df["Date"])
        df["TotalReturnsIndex"] = df["TotalReturnsIndex"].astype(float)
        df = df.rename(columns={"Date": "HistoricalDate", "TotalReturnsIndex": "CLOSE"})
        df = df.set_index("HistoricalDate").sort_index()
        return df
    except Exception as jugaad_error:
        st.warning(
            f"Could not fetch data for {index_name}. "
            f"jugaad-data error: {jugaad_error}."
        )
        return None


# Metrics & transforms
def calculate_metrics(df: pd.DataFrame, rf_rate: float) -> tuple[float, float, float]:
    """CAGR, annualised volatility, and Sharpe ratio for a CLOSE price series."""
    daily_returns = df["CLOSE"].pct_change().dropna()

    start_value, end_value = df["CLOSE"].iloc[0], df["CLOSE"].iloc[-1]
    years = len(df) / TRADING_DAYS
    cagr = (end_value / start_value) ** (1 / years) - 1

    volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe_ratio = (daily_returns.mean() * TRADING_DAYS - rf_rate) / volatility

    return cagr, volatility, sharpe_ratio


def calculate_rolling_returns(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    df = df.copy()
    cagr = (df["CLOSE"] / df["CLOSE"].shift(window_days)) ** (TRADING_DAYS / window_days) - 1
    df[f"Rolling_{window_days // TRADING_DAYS}_Year_CAGR"] = cagr
    return df


def calculate_yearly_ranks(data_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_yearly_returns = []
    for index_name, df in data_frames.items():
        df = df.copy()
        df["Year"] = df.index.year
        yearly_returns = (
            df.groupby("Year")["CLOSE"]
            .apply(lambda x: (x.iloc[-1] / x.iloc[0]) - 1)
            .reset_index(name="Yearly_Return")
        )
        yearly_returns["INDEX_NAME"] = index_name
        all_yearly_returns.append(yearly_returns)

    combined_returns = pd.concat(all_yearly_returns, ignore_index=True)
    combined_returns["Rank"] = combined_returns.groupby("Year")["Yearly_Return"].rank(
        ascending=False, method="dense"
    )
    return combined_returns.sort_values(by=["Year", "Rank"])


def find_best_and_worst_indices(combined_returns: pd.DataFrame):
    rank_counts = combined_returns.groupby("INDEX_NAME")["Rank"].agg(
        first_count=lambda x: (x == 1).sum(),
        last_count=lambda x: (x == x.max()).sum(),
    ).reset_index()

    avg_ranks = (
        combined_returns.groupby("INDEX_NAME")["Rank"].mean().reset_index()
        .rename(columns={"Rank": "Average_Rank"})
    )

    summary = pd.merge(rank_counts, avg_ranks, on="INDEX_NAME")
    best_index = summary.sort_values(by=["first_count", "Average_Rank"], ascending=[False, True]).iloc[0]
    worst_index = summary.sort_values(by=["last_count", "Average_Rank"], ascending=[False, False]).iloc[0]

    return summary, best_index, worst_index


# Plotting
def plot_normalized_prices(df: pd.DataFrame, title: str = "Normalized Index Prices") -> go.Figure:
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode="lines", name=column))
        # Label the final value directly on the chart instead of adding a
        # second legend entry per index.
        fig.add_annotation(
            x=df.index[-1], y=df[column].iloc[-1],
            text=f"{df[column].iloc[-1]:.1f}",
            showarrow=False, xanchor="left", font=dict(size=11),
        )
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Normalized Price (Base 100)",
        template="plotly_white", legend_title="Index", hovermode="x unified",
    )
    return fig


def plot_yearly_ranks_plotly(combined_returns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for index_name in combined_returns["INDEX_NAME"].unique():
        df = combined_returns[combined_returns["INDEX_NAME"] == index_name]
        fig.add_trace(go.Scatter(
            x=df["Year"], y=df["Rank"], mode="lines+markers+text", name=index_name,
            text=[f"{ret:.1%}" for ret in df["Yearly_Return"]], textposition="top center",
            hovertemplate=(
                f"<b>Index:</b> {index_name}<br><b>Year:</b> %{{x}}<br>"
                "<b>Rank:</b> %{y}<br><b>Return:</b> %{text}<extra></extra>"
            ),
        ))
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Rank",
        yaxis=dict(autorange="reversed"), legend_title="Index Name",
        template="plotly_white", height=600,
    )
    return fig


def build_efficient_frontier_points(mu, S, weight_bounds, n_points: int = 40):
    """Trace the efficient frontier by solving efficient_return at a range
    of target returns. Infeasible targets are skipped."""
    risks, rets = [], []
    for target in np.linspace(mu.min(), mu.max(), n_points):
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
            ef.efficient_return(target_return=target)
            r, v, _ = ef.portfolio_performance()
            risks.append(v)
            rets.append(r)
        except Exception:
            continue
    return risks, rets

def plot_efficient_frontier_plotly(mu, S, weight_bounds, rf_rate: float) -> go.Figure:
    """Interactive Plotly efficient frontier: frontier curve, a cloud of
    random portfolios shaded by Sharpe ratio, and the max-Sharpe / min-vol
    tangency points."""
    ef_frontier_risks, ef_frontier_rets = build_efficient_frontier_points(mu, S, weight_bounds)

    ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    ef_max_sharpe.max_sharpe(risk_free_rate=rf_rate)
    ret_ms, std_ms, _ = ef_max_sharpe.portfolio_performance(risk_free_rate=rf_rate)

    ef_min_vol = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    ef_min_vol.min_volatility()
    ret_mv, std_mv, _ = ef_min_vol.portfolio_performance(risk_free_rate=rf_rate)

    # Random portfolios for context (Dirichlet weights sum to 1).
    n_assets = len(mu)
    n_samples = 4000
    w = np.random.dirichlet(np.ones(n_assets), n_samples)
    rand_rets = w @ mu.values
    rand_stds = np.sqrt(np.einsum("ij,jk,ik->i", w, S.values, w))
    rand_sharpes = (rand_rets - rf_rate) / rand_stds

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rand_stds, y=rand_rets, mode="markers", name="Random portfolios",
        marker=dict(size=5, color=rand_sharpes, colorscale="Plasma", showscale=True,
                    colorbar=dict(title="Sharpe")),
        hovertemplate="Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ef_frontier_risks, y=ef_frontier_rets, mode="lines", name="Efficient frontier",
        line=dict(color="#392D00", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[std_ms], y=[ret_ms], mode="markers", name="Max Sharpe",
        marker=dict(symbol="star", size=16, color="#D62728"),
    ))
    fig.add_trace(go.Scatter(
        x=[std_mv], y=[ret_mv], mode="markers", name="Min Volatility",
        marker=dict(symbol="star", size=16, color="#2CA02C"),
    ))
    fig.update_layout(
        title="Efficient Frontier with Random Portfolios",
        xaxis_title="Annual Volatility (Risk)", yaxis_title="Expected Annual Return",
        xaxis_tickformat=".1%", yaxis_tickformat=".1%",
        template="plotly_white", height=560, legend=dict(orientation="h", y=-0.15),
    )
    return fig

# Main app
def main():
    st.markdown(
        """
        <div class="hero">
            <h1>💰 Factor Investing in India</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    #Sidebar inputs
    return_type = st.sidebar.selectbox(
        "**Return type**",
        ["Total Returns Index Values", "Price Returns Index Values"],
        label_visibility="visible",
    )

    st.sidebar.subheader("Select factors to analyse")
    factors_to_analyse = st.sidebar.pills(
        label="Factors to analyse",
        options=list(FACTOR_INDEX_OPTIONS.keys()),
        selection_mode="multi",
        label_visibility="collapsed",
        default=["Alpha", "Momentum", "Quality"],
    )

    st.sidebar.subheader("Select an index for each factor")
    selected_indices = [
        st.sidebar.selectbox(f"Index for **{factor}**:", FACTOR_INDEX_OPTIONS[factor])
        for factor in factors_to_analyse
    ]

    allow_short_position = st.sidebar.radio("**Allow short positions**", ["No", "Yes"])
    allow_short = allow_short_position == "Yes"
    weight_bounds = (-1, 1) if allow_short else (0, 1)
    rf_rate = st.sidebar.number_input("**Risk-free rate**", value=6.00, help="Value in %") / 100

    st.sidebar.divider()
    start_date = st.sidebar.date_input(
        "**Start date**", value=pd.to_datetime("today") - dt.timedelta(days=1825),
        format="DD/MM/YYYY", min_value=dt.datetime(2005, 1, 1),
    )
    end_date = st.sidebar.date_input(
        "**End date**", format="DD/MM/YYYY", min_value=start_date, max_value=pd.to_datetime("today"),
    )
    start_date, end_date = str(start_date), str(end_date)

    if not st.sidebar.button("Compute", type="primary"):
        st.info("Choose your factors and indices from the sidebar, then click **Compute**.", icon="⬅️")
        return

    if not selected_indices:
        st.error("Select at least one factor to analyse.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["**Overview**", "**Rolling Returns**", "**Yearly Returns**",
         "**Rankings**", "**Efficient Frontier**", "_Download_"]
    )

    #Fetch data 
    data_frames: dict[str, pd.DataFrame] = {}
    for index in selected_indices:
        st.toast(f"Fetching data for {index}...")
        fetcher = fetch_index_data_tr if return_type == "Total Returns Index Values" else fetch_index_data_pr
        df = fetcher(index, start_date, end_date)
        if df is not None and not df.empty:
            data_frames[index] = df

    if not data_frames:
        st.error("No data available for the selected indices in this date range.")
        return

    # Single CLOSE-price frame, aligned on common dates, for the optimiser.
    close_df = pd.concat(
        {name: df["CLOSE"] for name, df in data_frames.items()}, axis=1
    ).dropna()
    close_df.columns = [f"CLOSE_{name}" for name in close_df.columns]

    # ---------------- Tab 1: Overview ----------------
    with tab1:
        results = []
        for index_name, df in data_frames.items():
            cagr, volatility, sharpe_ratio = calculate_metrics(df, rf_rate)
            results.append({"Index Name": index_name, "CAGR": cagr, "Volatility": volatility,
                             "Sharpe Ratio": sharpe_ratio})
        results_df = pd.DataFrame(results)

        st.subheader("Metrics")
        st.dataframe(
            results_df, width='stretch', hide_index=True,
            column_config={
                "CAGR": st.column_config.NumberColumn("CAGR", format="percent"),
                "Volatility": st.column_config.NumberColumn("Volatility", format="percent"),
                "Sharpe Ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f"),
            },
        )

        st.subheader("Normalised index prices (base 100)")
        base_100_data = pd.DataFrame({
            index: (df["CLOSE"] / df["CLOSE"].iloc[0]) * 100
            for index, df in data_frames.items()
        })
        with st.container(border=True):
            st.plotly_chart(plot_normalized_prices(base_100_data), width='stretch')

    # ---------------- Tab 2: Rolling returns ----------------
    with tab2:
        rolling_windows = [2 * TRADING_DAYS, 5 * TRADING_DAYS, 10 * TRADING_DAYS]
        average_rolling_returns = {}

        for index, df in data_frames.items():
            avg_returns = {}
            for window in rolling_windows:
                df = calculate_rolling_returns(df, window)
                column_name = f"Rolling_{window // TRADING_DAYS}_Year_CAGR"
                avg_returns[f"{window // TRADING_DAYS}_Year_Average_CAGR"] = df[column_name].mean()
            average_rolling_returns[index] = avg_returns

            fig_rr = go.Figure()
            for window in rolling_windows:
                years = window // TRADING_DAYS
                fig_rr.add_trace(go.Scatter(
                    x=df.index, y=df[f"Rolling_{years}_Year_CAGR"],
                    mode="lines", name=f"{years}-Year Rolling CAGR",
                ))
            fig_rr.update_layout(
                title=f"Rolling Returns for {index}", xaxis_title="Date", yaxis_title="Rolling CAGR",
                legend_title="Rolling Period", template="plotly_white", yaxis_tickformat=".0%",
            )
            with st.container(border=True):
                st.plotly_chart(fig_rr, width='stretch')

        average_rolling_returns_df = pd.DataFrame(average_rolling_returns).T * 100
        average_rolling_returns_df = (
            average_rolling_returns_df.style.format("{:.2f}%")
            .highlight_min(axis=0, color="#FFB3A7").highlight_max(axis=0, color="#B7E4C7")
        )
        st.dataframe(average_rolling_returns_df, column_config={"": "Index"})

    # ---------------- Tab 3: Yearly returns ----------------
    with tab3:
        yearly_ranks = calculate_yearly_ranks(data_frames)
        returns_table = yearly_ranks.pivot(index="Year", columns="INDEX_NAME", values="Yearly_Return")
        returns_table.index = returns_table.index.map(str)

        st.subheader("Yearly returns")
        st.write("The table below shows the yearly returns for each index.")
        st.dataframe(
            returns_table.style.highlight_min(axis=1, color="#FFB3A7")
            .highlight_max(axis=1, color="#B7E4C7").format("{:.1%}"),
            width='stretch',
        )

    # ---------------- Tab 4: Rankings ----------------
    with tab4:
        with st.container(border=True):
            yearly_ranks = calculate_yearly_ranks(data_frames)
            ranking_table = yearly_ranks.pivot(index="Year", columns="INDEX_NAME", values="Rank")
            ranking_table.index = ranking_table.index.map(str)

            summary, best_index, worst_index = find_best_and_worst_indices(yearly_ranks)
            summary = summary.sort_values(by="Average_Rank", ascending=True)

            st.subheader("Ranking summary")
            st.dataframe(summary, hide_index=True, width='stretch')

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Best ranking index")
                st.write(f"Index Name: {best_index['INDEX_NAME']}")
                st.write(f"Times Ranked 1st: {best_index['first_count']}")
                st.write(f"Average Rank: {best_index['Average_Rank']:.2f}")
            with col2:
                st.subheader("❌ Worst ranking index")
                st.write(f"Index Name: {worst_index['INDEX_NAME']}")
                st.write(f"Times Ranked Last: {worst_index['last_count']}")
                st.write(f"Average Rank: {worst_index['Average_Rank']:.2f}")

        st.subheader("Yearly ranks")
        st.write("The table below shows the yearly ranks for each index (1 = best).")
        st.dataframe(
            ranking_table.style.highlight_min(axis=1, color="#B7E4C7")
            .highlight_max(axis=1, color="#FFB3A7").format("{:.0f}"),
            width='stretch',
        )

        st.subheader("Yearly Ranks of Indices with Returns")
        st.plotly_chart(plot_yearly_ranks_plotly(yearly_ranks), width='stretch')

    # ---------------- Tab 5: Efficient frontier ----------------
    with tab5:
        if len(factors_to_analyse) > 2:
            mu = expected_returns.mean_historical_return(close_df)
            S = risk_models.sample_cov(close_df)

            ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
            max_sharpe_weights = ef_max_sharpe.max_sharpe(risk_free_rate=rf_rate)

            with st.container(border=True):
                perf = ef_max_sharpe.portfolio_performance(risk_free_rate=rf_rate)
                st.subheader("Weights for max Sharpe ratio")
                st.write(
                    "Also called the tangency portfolio: the portfolio for which the "
                    "capital market line is tangent to the efficient frontier."
                )
                st.dataframe(
                    pd.Series(max_sharpe_weights, name="Weight"),
                    column_config={"value": st.column_config.NumberColumn("Weight", format="percent")},
                    width='stretch',
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Expected annual return", f"{perf[0]:.2%}")
                c2.metric("Annual volatility", f"{perf[1]:.2%}")
                c3.metric("Sharpe ratio", f"{perf[2]:.2f}")

            ef_min_vol = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
            min_vol_weights = ef_min_vol.min_volatility()

            with st.container(border=True):
                perf = ef_min_vol.portfolio_performance(risk_free_rate=rf_rate)
                st.subheader("Weights for min volatility portfolio")
                st.dataframe(
                    pd.Series(min_vol_weights, name="Weight"),
                    column_config={"value": st.column_config.NumberColumn("Weight", format="percent")},
                    width='stretch',
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Expected annual return", f"{perf[0]:.2%}")
                c2.metric("Annual volatility", f"{perf[1]:.2%}")
                c3.metric("Sharpe ratio", f"{perf[2]:.2f}")

            with st.container(border=True):
                st.subheader("Efficient frontier")
                st.plotly_chart(
                    plot_efficient_frontier_plotly(mu, S, weight_bounds, rf_rate),
                    width='stretch',
                )
        else:
            st.info("Select at least 3 indices for the efficient frontier to be displayed.", icon="ℹ️")

    # ---------------- Tab 6: Download ----------------
    with tab6:
        st.subheader("Data used")
        st.dataframe(close_df, width='stretch')

if __name__ == "__main__":
    main()
