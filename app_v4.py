import streamlit as st, numpy as np, pandas as pd
from nsepython import index_history, index_total_returns
import plotly.graph_objects as go
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import matplotlib.pyplot as plt

# Fetch historical PR data for an index
def fetch_index_data_PR(index_name, start_date, end_date):
    try:
        data = index_history(index_name, start_date, end_date)
        df = pd.DataFrame(data)
        df["HistoricalDate"] = pd.to_datetime(df["HistoricalDate"])
        df['CLOSE'] = df['CLOSE'].astype(float)
        df.set_index("HistoricalDate", inplace=True)
        df.sort_values(by=['HistoricalDate'], inplace=True)
        return df
    except Exception as e:
        st.warning(f"Could not fetch data for {index_name}. Error: {e}")
        return None

# Fetch historical TR data for an index
def fetch_index_data_TR(index_name, start_date, end_date):
    try:
        data = index_total_returns(index_name, start_date, end_date)
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"])
        df['TotalReturnsIndex'] = df['TotalReturnsIndex'].astype(float)
        df.sort_values(by=['Date'], inplace=True)
        df.set_index("Date", inplace=True)
        return df
    except Exception as e:
        st.warning(f"Could not fetch data for {index_name}. Error: {e}")
        return None

# Function to calculate metrics
def calculate_metrics(df, rf_rate):
    # Calculate daily returns
    df["Daily Returns"] = df["CLOSE"].pct_change()
    
    # CAGR
    start_value = df["CLOSE"].iloc[0]
    end_value = df["CLOSE"].iloc[-1]
    total_days = len(df)
    years = total_days / 252
    cagr = (end_value / start_value) ** (1 / years) - 1
    
    # Volatility
    volatility = df["Daily Returns"].std() * np.sqrt(252)
    
    # Sharpe Ratio
    average_daily_return = df["Daily Returns"].mean()
    sharpe_ratio = (average_daily_return * 252 - rf_rate) / volatility
    
    return cagr, volatility, sharpe_ratio

# Function to plot normalized index prices
def plot_normalized_prices(df, title='Normalized Index Prices'):
    fig = go.Figure()

    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))

    # Add the present value of 100 near the end of each stock's graph
    for column in df.columns:
        last_value = df[column].iloc[-1]
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[last_value],
                                 mode='markers+text',
                                 marker=dict(color='red', size=8),
                                 text=f'{last_value:.2f}',
                                 textposition='bottom right',
                                 name=f'{column} PV'))

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Normalized Price')
    return fig

# Function to calculate rolling returns
def calculate_rolling_returns(df, window_days):
    df = df.copy()
    cagr = (
        (df["CLOSE"] / df["CLOSE"].shift(window_days)) ** (252 / window_days) - 1
    )
    # Align CAGR with the end date of the rolling window
    df[f"Rolling_{window_days // 252}_Year_CAGR"] = cagr
    return df

# Function to calculate yearly returns and rank indices
def calculate_yearly_ranks(data_frames):
    all_yearly_returns = []
    
    for index_name, df in data_frames.items():
        df = df.copy()
        df["Year"] = df.index.year
        yearly_returns = (
            df.groupby("Year")["CLOSE"]
            .apply(lambda x: (x.iloc[-1] / x.iloc[0]) - 1)
            .reset_index()
            .rename(columns={"CLOSE": "Yearly_Return"})
        )
        yearly_returns["INDEX_NAME"] = index_name
        all_yearly_returns.append(yearly_returns)
    
    combined_returns = pd.concat(all_yearly_returns)
    combined_returns["Rank"] = combined_returns.groupby("Year")["Yearly_Return"].rank(ascending=False, method="dense")
    combined_returns = combined_returns.sort_values(by=["Year", "Rank"])
    
    return combined_returns

# Function to calculate best & worst ranking indices
def find_best_and_worst_indices(combined_returns):
    # Count how many times each index was ranked 1st and last
    rank_counts = combined_returns.groupby("INDEX_NAME")["Rank"].agg(
        first_count=lambda x: (x == 1).sum(),
        last_count=lambda x: (x == x.max()).sum()
    ).reset_index()

    # Calculate the average rank of each index
    avg_ranks = combined_returns.groupby("INDEX_NAME")["Rank"].mean().reset_index()
    avg_ranks.rename(columns={"Rank": "Average_Rank"}, inplace=True)

    # Merge the results
    summary = pd.merge(rank_counts, avg_ranks, on="INDEX_NAME")

    # Sort indices for best and worst ranking
    best_index = summary.sort_values(by=["first_count", "Average_Rank"], ascending=[False, True]).iloc[0]
    worst_index = summary.sort_values(by=["last_count", "Average_Rank"], ascending=[False, False]).iloc[0]

    return summary, best_index, worst_index

# Function to plot yrarly ranks
def plot_yearly_ranks_plotly(combined_returns):
    fig = go.Figure()

    # Add lines and markers for each index
    for index_name in combined_returns["INDEX_NAME"].unique():
        df = combined_returns[combined_returns["INDEX_NAME"] == index_name]
        fig.add_trace(go.Scatter(
            x=df["Year"],
            y=df["Rank"],
            mode="lines+markers+text",
            name=index_name,
            text=[f"{ret:.1%}" for ret in df["Yearly_Return"]],
            textposition="top center",
            hovertemplate=(
                f"<b>Index:</b> {index_name}<br>"
                "<b>Year:</b> %{x}<br>"
                "<b>Rank:</b> %{y}<br>"
                "<b>Return:</b> %{text}<extra></extra>"
            )
        ))

    # Customize layout
    fig.update_layout(
        title="Yearly Ranks of Indices with Returns",
        xaxis_title="Year",
        yaxis_title="Rank",
        yaxis=dict(autorange="reversed"),  # Invert y-axis so Rank 1 is at the top
        legend_title="Index Name",
        template="plotly_white",
        height=600,
        width=900,
    )
    
    return fig

# Main Streamlit app
def main():

    # Page Config
    st.set_page_config(page_title='Factor Investing in India', page_icon="💰", layout='wide')

    st.title('Factor Investing in India')

    # Inputs
    type = st.sidebar.selectbox("**TR/PR**", ["Total Returns Index Values", "Price Returns Index Values"], label_visibility='hidden')

    selected_indices = []

    st.sidebar.subheader("**Select Factors to Analyse:**")
    options = ["Alpha", "Momentum", "Quality", "Value", "Volatility"]
    factors_to_analyse = st.sidebar.pills(label="Factors to Analyse", options=options, selection_mode="multi", label_visibility="collapsed", default=["Alpha", "Momentum", "Quality"])

    st.sidebar.subheader("**Select Index for each Factor:**")
    if "Alpha" in factors_to_analyse:
        selected_indices.append(st.sidebar.selectbox(f"Select index for factor **Alpha**:", ("Nifty Alpha 50", "Nifty100 Alpha 30", "Nifty200 Alpha 30")))

    if "Momentum" in factors_to_analyse:
        selected_indices.append(st.sidebar.selectbox(f"Select index for factor **Momentum**:", ("Nifty Midcap150 Momentum 50", "Nifty500 Momentum 50")))
        
    if "Quality" in factors_to_analyse:
        selected_indices.append(st.sidebar.selectbox(f"Select index for factor **Quality**:", ("Nifty500 Quality 50", "Nifty Midcap150 Quality 50", "Nifty Smallcap250 Quality 50")))
    
    if "Value" in factors_to_analyse:
        selected_indices.append(st.sidebar.selectbox(f"Select index for factor **Value**:", ("Nifty50 Value 20", "Nifty200 Value 30", "Nifty500 Value 50")))
    
    if "Volatility" in factors_to_analyse:
        selected_indices.append(st.sidebar.selectbox(f"Select index for factor **Volatility**:", ("Nifty Low Volatility 50")))
    
    allow_short_position = st.sidebar.radio("**Allow Short Positions**", ["Yes", "No"])
    rf_rate = st.sidebar.number_input("**Risk Free Rate**", value=2.00, help="Value in %")/100

    st.sidebar.divider()
    
    start_date = str(st.sidebar.date_input('**Start Date**', value=pd.to_datetime('today') - dt.timedelta(days=365), format="DD/MM/YYYY", min_value=dt.datetime(2005, 1, 1)))
    end_date = str(st.sidebar.date_input('**End Date**', format="DD/MM/YYYY"))
    
    #Compute
    if st.sidebar.button('Compute', type="primary"):

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['**Overview**', '**Rolling Returns**', '**Yearly Returns**', '**Rankings**', '**Efficient Frontier**', '_Download_'])

        #Fetch data and ensure availability
        data_frames = {}
        for index in selected_indices:
            st.toast(f"Fetching data for {index}...")
            if type == "Total Returns Index Values":
                df = fetch_index_data_TR(index, start_date, end_date)
                df = df.rename(columns={'Date': 'HistoricalDate', 'TotalReturnsIndex': 'CLOSE'})
            else:
                df = fetch_index_data_PR(index, start_date, end_date)
            if df is not None:
                data_frames[index] = df
        
        if not data_frames:
            st.error("No data available for the selected indices.")
            return
        
        #Get CLOSE prices in a single dataframe
        close_df = pd.DataFrame()

        for index_name, df in data_frames.items():
            close_column = df[['CLOSE']].rename(columns={'CLOSE': f'CLOSE_{index_name}'})
            
            if close_df.empty:
                close_df = close_column
            else:
                close_df = close_df.merge(close_column, left_index=True, right_index=True)

        close_df.reset_index(inplace=True)
        if type == "Price Returns Index Values":
            close_df.set_index("HistoricalDate", inplace=True)
        else:
            close_df.set_index("Date", inplace=True)

        with tab1:
            # Calculate metrics for all indices
            results = []
            for index_name, df in data_frames.items():
                cagr, volatility, sharpe_ratio = calculate_metrics(df, rf_rate)
                results.append({
                    "Index Name": index_name,
                    "CAGR (%)": cagr * 100,
                    "Volatility (%)": volatility * 100,
                    "Sharpe Ratio": sharpe_ratio
                })

            # Convert results to DataFrame
            results_df = pd.DataFrame(results)

            # Display the table
            st.subheader("Metrics Table")
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            st.subheader("Normalised Index Prices to Base 100")

            # Normalize prices to a base of 100
            base_100_data = pd.DataFrame()
            
            for index, df in data_frames.items():
                df["Base100"] = (df["CLOSE"] / df["CLOSE"].iloc[0]) * 100
                base_100_data[index] = df["Base100"]

            normalised_plot = plot_normalized_prices(base_100_data)

            container = st.container(border=True)
            container.plotly_chart(normalised_plot, use_container_width=True)

        with tab2:
            #Rolling returns
            rolling_windows = [2 * 252, 5 * 252, 10 * 252]

            # Dictionary to store average rolling returns
            average_rolling_returns = {}
            
            for index, df in data_frames.items():
                st.subheader(f"Rolling Returns for {index}")

                # Calculate rolling returns for each window and store the results
                avg_returns = {}
                # Calculate rolling returns
                for window in rolling_windows:
                    df = calculate_rolling_returns(df, window)
                    column_name = f"Rolling_{window // 252}_Year_CAGR"
                    avg_returns[f"{window // 252}_Year_Average_CAGR"] = df[column_name].mean()

                # Store average rolling returns for the index
                average_rolling_returns[index] = avg_returns
    
                # Plot rolling returns
                fig_RR = go.Figure()
                for window in rolling_windows:
                    years = window // 252
                    fig_RR.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[f"Rolling_{years}_Year_CAGR"],
                            mode="lines",
                            name=f"{years}-Year Rolling CAGR",
                        )
                    )
                fig_RR.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Rolling CAGR",
                    legend_title="Rolling Period",
                    template="plotly",
                )
                st.plotly_chart(fig_RR, use_container_width=True)
            
            # Convert the results into a DataFrame for better visualization
            average_rolling_returns_df = pd.DataFrame(average_rolling_returns).T
            average_rolling_returns_df = average_rolling_returns_df * 100
            average_rolling_returns_df = average_rolling_returns_df.style.format("{:.2f}%").highlight_min(axis=0, color="orangered").highlight_max(axis=0, color="lawngreen")

            st.dataframe(average_rolling_returns_df)

        with tab3:
            #Yearly returns and ranking
            yearly_ranks = calculate_yearly_ranks(data_frames)
            
            returns_table = yearly_ranks.pivot(index="Year", columns="INDEX_NAME", values="Yearly_Return")
            returns_table.index = returns_table.index.map(str)

            st.subheader("Yearly Returns")
            st.write("The table below shows the yearly returns for each index.")

            st.dataframe(returns_table.style.highlight_min(axis=1, color="orangered").highlight_max(axis=1, color="lawngreen").
                            format("{:.1%}"), use_container_width=True)

        with tab4:
            #Yearly returns and ranking
            yearly_ranks = calculate_yearly_ranks(data_frames)

            ranking_table = yearly_ranks.pivot(index="Year", columns="INDEX_NAME", values="Rank")
            ranking_table.index = ranking_table.index.map(str)

            summary, best_index, worst_index = find_best_and_worst_indices(yearly_ranks)

            # Display results
            st.subheader("Ranking Summary")
            summary = summary.sort_values(by='Average_Rank', ascending=True)
            st.dataframe(summary, hide_index=True)

            st.subheader("✅ Best Ranking Index")
            st.write(f"Index Name: {best_index['INDEX_NAME']}")
            st.write(f"Times Ranked 1st: {best_index['first_count']}")
            st.write(f"Average Rank: {best_index['Average_Rank']:.2f}")

            st.subheader("❌ Worst Ranking Index")
            st.write(f"Index Name: {worst_index['INDEX_NAME']}")
            st.write(f"Times Ranked Last: {worst_index['last_count']}")
            st.write(f"Average Rank: {worst_index['Average_Rank']:.2f}")

            # Visualization
            st.subheader("Ranking Visualization")
            st.bar_chart(summary.set_index("INDEX_NAME")[["first_count", "last_count"]])
                
            st.subheader("Yearly Ranks")
            st.write("The table below shows the yearly ranks for each index.")

            container = st.container(border=True)

            with container:
                st.dataframe(ranking_table.style.highlight_min(axis=1, color="lawngreen").highlight_max(axis=1, color="orangered").
                                format("{:.0f}"), use_container_width=True)
            
            st.subheader("Rankings Trend Over the Years")
            line_chart_rank = plot_yearly_ranks_plotly(yearly_ranks)
            st.plotly_chart(line_chart_rank, use_container_width=True)
   
        with tab5:
            if len(factors_to_analyse) > 2:
                # Calculate expected returns and sample covariance
                mu = expected_returns.mean_historical_return(close_df)
                S = risk_models.sample_cov(close_df)

                if allow_short_position == "Yes":
                    ef = EfficientFrontier(mu, S, weight_bounds=(-1,1))
                else:
                    ef = EfficientFrontier(mu, S)

                max_sharpe_weights = ef.max_sharpe()

                with st.container(border=True):
                    # ef.portfolio_performance(verbose=True)
                    porfolio_performance = ef.portfolio_performance(verbose=False, risk_free_rate=rf_rate)
                    ear = str(round(porfolio_performance[0]*100, 2)) + "%"
                    ann_vol = str(round(porfolio_performance[1]*100, 2)) + "%"
                    sharpe_ratio = round(porfolio_performance[2], 2)

                    st.subheader("Weights for Max Sharpe Ratio")
                    st.write("The result is also referred to as the tangency portfolio, as it is the portfolio for which the capital market line is tangent to the efficient frontier.")
                    st.dataframe(max_sharpe_weights, column_config={"": "Index", 
                                                        "value": st.column_config.NumberColumn("Weights", help="In decimals")},
                                use_container_width=True)
                    
                    st.subheader("Portfolio Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected annual return", value=ear)
                    col2.metric("Annual volatility", value=ann_vol)
                    col3.metric("Sharpe Ratio", value=sharpe_ratio)


                # Calculate expected returns and sample covariance
                mu = expected_returns.mean_historical_return(close_df)
                S = risk_models.sample_cov(close_df)

                if allow_short_position == "Yes":
                    ef = EfficientFrontier(mu, S, weight_bounds=(-1,1))
                else:
                    ef = EfficientFrontier(mu, S)

                min_vol = ef.min_volatility()

                with st.container(border=True):
                    # ef.portfolio_performance(verbose=True)
                    porfolio_performance = ef.portfolio_performance(verbose=False, risk_free_rate=rf_rate)
                    ear = str(round(porfolio_performance[0]*100, 2)) + "%"
                    ann_vol = str(round(porfolio_performance[1]*100, 2)) + "%"
                    sharpe_ratio = round(porfolio_performance[2], 2)

                    st.subheader("Weights for Min Volatility Ratio")
                    st.dataframe(min_vol, column_config={"": "Index", 
                                                        "value": st.column_config.NumberColumn("Weights", help="In decimals")},
                                use_container_width=True)
                    
                    st.subheader("Portfolio Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected annual return", value=ear)
                    col2.metric("Annual volatility", value=ann_vol)
                    col3.metric("Sharpe Ratio", value=sharpe_ratio)

                if allow_short_position == "Yes":
                    ef = EfficientFrontier(mu, S, weight_bounds=(-1,1))
                else:
                    ef = EfficientFrontier(mu, S)

                with st.container(border=True):
                    fig, ax = plt.subplots()
                    ef_max_sharpe = ef.deepcopy()
                    ef_min_vol = ef.deepcopy()
                    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

                    # Find the tangency portfolio
                    ef_max_sharpe.max_sharpe()
                    ef_min_vol.min_volatility()
                    ret_tangent_mvol, std_tangent_mvol, _ = ef_min_vol.portfolio_performance()
                    ret_tangent_msharpe, std_tangent_msharpe, _ = ef_max_sharpe.portfolio_performance()
                    ax.scatter(std_tangent_msharpe, ret_tangent_msharpe, marker="*", s=100, c="r", label="Max Sharpe")
                    ax.scatter(ret_tangent_mvol, std_tangent_mvol, marker="*", s=100, c="g", label="Min Volatility")

                    # Generate random portfolios
                    n_samples = 10000
                    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
                    rets = w.dot(ef.expected_returns)
                    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
                    sharpes = rets / stds
                    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

                    # Output
                    ax.set_title("Efficient Frontier with random portfolios")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info('Select atleast 3 indices for Efficient Frontier to be displayed.', icon="ℹ️")

        with tab6:
            st.subheader('Data used :')
            st.dataframe(close_df)

if __name__ == '__main__':
    main()