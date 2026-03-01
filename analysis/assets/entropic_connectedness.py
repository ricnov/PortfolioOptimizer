import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RenyiConnectedness:
    """
    Implements the class of systemic risk indicators based on Rényi entropy
    as proposed by Mishel Qyrana (2024).

    Reference:
    Qyrana, M. (2024). Forecasting stock market crashes through entropy-based
    proper measures of connectedness.

    Implements the alpha-Renyi entropy-based measure of connectedness mu_alpha(C).

    Parameters based on user specification:
    n: number of assets
    C: asset correlation matrix (n x n)
    lambda_i: eigenvalues of C
    p_i = lambda_i / n
    alpha: positive real number, alpha != 1

    Formula:
    mu_alpha(C) = 1 / (1 + (1 / (1 - alpha)) * log(sum(p_i ^ alpha)))
    """

    def __init__(self, alpha=2, window_size=13):
        """
        Args:
            alpha (float): The order of Renyi entropy (alpha != 1). Default is 2.
            window_size (int): The rolling window size for correlation calculation.
        """
        if alpha == 1:
            raise ValueError("Alpha cannot be 1 for this specific implementation (singularity in 1/(1-alpha)).")
        if alpha < 0:
            raise ValueError("Alpha must be >= 0.")

        self.alpha = alpha
        self.window_size = window_size

    def calculate_measure(self, log_returns):
        """
        Applies the rolling window calculation over the log returns data.
        """
        results = {}

        # We need at least 'window_size' data points
        if len(log_returns) < self.window_size:
            raise ValueError("Data length is smaller than the window size.")

        # Loop through the data
        for i in range(self.window_size, len(log_returns) + 1):
            # 1. Define the window
            window = log_returns.iloc[i - self.window_size : i]
            date = window.index[-1]

            # 2. Compute Correlation Matrix C (n x n)
            # The correlation matrix of the returns in the window
            C = window.corr().values

            # Handle cases where correlation cannot be computed (NaNs)
            if np.isnan(C).any():
                results[date] = np.nan
                continue

            # n is the number of assets
            n = C.shape[0]

            # 3. Compute Eigenvalues (lambda)
            # numpy.linalg.eigh returns eigenvalues in ascending order
            eigenvalues, _ = np.linalg.eigh(C)

            # Filter numerical noise (ensure non-negative) and sort just in case
            eigenvalues = np.sort(np.maximum(eigenvalues, 0))

            # 4. Compute Probabilities p_i
            # p_i = lambda_i / n
            p = eigenvalues / n

            # 5. Compute alpha-Renyi Entropy part
            # Term: sum(p_i ^ alpha)
            sum_p_alpha = np.sum(p ** self.alpha)

            # H_alpha = (1 / (1 - alpha)) * log(sum_p_alpha)
            # We use numpy's natural log (log)
            entropy = (1 / (1 - self.alpha)) * np.log(sum_p_alpha)

            # 6. Compute Connectedness Measure mu_alpha(C)
            # mu = 1 / (1 + entropy)
            mu = 1 / (1 + entropy)

            results[date] = mu

        return pd.Series(results).dropna()

# --- Helper to fetch data ---
def get_data():
    """
    Fetches the standard S&P 500 sector ETFs to represent the 'n' assets.
    """
    tickers = [
        "XLE", "XLB", "XLI", "XLY", "XLP",
        "XLV", "XLF", "XLK", "XLU"
        # Note: We use 9 legacy sectors to ensure long history availability
    ]
    print(f"Fetching data for n={len(tickers)} assets...")
    data = yf.download(tickers, start="2020-01-01", end="2026-02-11", progress=False, auto_adjust=False)['Adj Close']

    # Calculate Log Returns
    log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

# --- Execution ---
if __name__ == "__main__":
    # 1. Load Data
    returns_df = get_data()

    # 2. Instantiate the exact algorithm
    # Using alpha=2 (Collision entropy) as implied by common literature,
    # but the formula supports any alpha != 1
    model = RenyiConnectedness(alpha=2, window_size=20)

    # 3. Compute the indicator
    print("Calculating Renyi Connectedness Measure...")
    indicator = model.calculate_measure(returns_df)

    # 4. Fetch Benchmark for Plotting (S&P 500)
    sp500 = yf.download("^GSPC", start=indicator.index[0], end=indicator.index[-1], progress=False, auto_adjust=False)['Adj Close']

    # 5. Plotting
    sp500 = yf.download("^GSPC", start=indicator.index[0], end=indicator.index[-1], progress=False, auto_adjust=False)['Adj Close']
    sp500 = sp500['^GSPC']

    # 1. Create the Subplot Structure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'Renyi Entropy-Based Measure of Connectedness μ<sub>α</sub>(C)', 'S&P 500 Index')
    )

    # 2. Add Top Panel: Indicator (Firebrick color)
    fig.add_trace(
        go.Scatter(
            x=indicator.index,
            y=indicator.values,
            mode='lines',
            name=f'Connectedness μ<sub>α</sub> (α={model.alpha})',
            line=dict(color='firebrick', width=1.5)
        ),
        row=1, col=1
    )

    # 3. Add Bottom Panel: Market (Navy color)
    fig.add_trace(
        go.Scatter(
            x=sp500.index,
            y=sp500.values,
            mode='lines',
            name='S&P 500 Index',
            line=dict(color='navy', width=1.5)
        ),
        row=2, col=1
    )

    # 4. Update Layout to match the style
    fig.update_layout(
        height=1200,  # Similar to figsize=(12, 8)
        #width=1200,
        showlegend=True,
        plot_bgcolor='white', # Clean background
        hovermode="x unified" # Shows values for both plots when hovering
    )

    # 5. Update Axes (Grids and Labels)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    # specific Y-axis labels
    fig.update_yaxes(title_text="Connectedness", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.show()

    print("Calculation complete.")
