import pandas as pd
import numpy as np


class ARIMASimulator:
    """
    Generates synthetic time series data as a DataFrame with known ARIMA properties.
    """

    def __init__(self, start_date='2020-01-01', n_samples=500, seed=42):
        """
        Initializes the simulator.

        Args:
            start_date (str): The starting date for the time series.
            n_samples (int): The number of data points to generate.
            seed (int): Random seed for reproducibility.
        """
        self.start_date = start_date
        self.n_samples = n_samples
        np.random.seed(seed)

    def _create_dataframe(self, values):
        """Helper function to create a date-indexed DataFrame."""
        dates = pd.date_range(start=self.start_date,
                              periods=len(values), freq='D')
        return pd.DataFrame({'date': dates, 'value': values})

    def generate_arima_data(self, d=0, p_coeffs=[], q_coeffs=[], drift=0.0):
        """
        Generates a time series with specified ARIMA(p, d, q) properties.

        Args:
            d (int): The order of integration (number of differencing steps).
            p_coeffs (list): A list of AR coefficients (phi). The length is the 'p' order.
            q_coeffs (list): A list of MA coefficients (theta). The length is the 'q' order.
            drift (float): A constant drift term to add to the integrated series.

        Returns:
            pd.DataFrame: A DataFrame with 'date' and 'value' columns.
        """
        p = len(p_coeffs)
        q = len(q_coeffs)
        burn_in = 100  # Generate extra points to let the process stabilize
        total_samples = self.n_samples + burn_in

        # Start with random noise (the "shocks" or epsilon)
        noise = np.random.randn(total_samples)
        series = np.zeros(total_samples)

        # --- Generate the stationary ARMA(p,q) component ---
        for t in range(max(p, q), total_samples):
            ar_term = 0
            # Add AR terms (memory of past values)
            if p > 0:
                ar_term = np.sum([p_coeffs[i] * series[t-i-1]
                                 for i in range(p)])

            ma_term = 0
            # Add MA terms (memory of past errors/shocks)
            if q > 0:
                ma_term = np.sum([q_coeffs[i] * noise[t-i-1]
                                 for i in range(q)])

            series[t] = ar_term + ma_term + noise[t]

        # --- Apply integration 'd' times to make it non-stationary ---
        if d > 0:
            # Drop burn-in period before integrating
            stationary_component = series[burn_in:]
            integrated_series = stationary_component
            for _ in range(d):
                integrated_series = np.cumsum(integrated_series)

            # Add drift to the final integrated series
            if drift != 0:
                integrated_series += np.arange(self.n_samples) * drift

            final_series = integrated_series
        else:
            # If d=0, the series is just the stationary ARMA part
            final_series = series[burn_in:]

        return self._create_dataframe(final_series)
