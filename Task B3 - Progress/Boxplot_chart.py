import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def plot_boxplot_chart(data, n_days=10, column='Close', title='Boxplot of Stock Prices', save_as=None):

    if n_days < 1:
        raise ValueError("n_days must be greater than or equal to 1.")

    # Initialize list to hold rolling window data
    boxplot_data = []

    # Collecting the data for the boxplot: 
    for i in range(len(data) - n_days + 1):
        window_data = data[column].iloc[i:i+n_days].values
        boxplot_data.append(window_data)

    # Plotting chart
    plt.figure(figsize=(12, 6))
    plt.boxplot(boxplot_data, patch_artist=True, showfliers=True)
    plt.title(title)
    plt.xlabel('Rolling Window Number')
    plt.ylabel(f'{column} Price')
    plt.grid(True)

    plt.xticks(ticks=range(1, len(boxplot_data)+1, max(1, len(boxplot_data)//10)), 
               labels=range(n_days, len(data)+1, max(1, len(boxplot_data)//10)))

    if save_as:
        plt.savefig(save_as)
    else:
        plt.show()


if __name__ == "__main__":
    # Downloading stock data form META
    data = yf.download('META', start='2022-01-01', end='2023-01-01')

    data.index = pd.to_datetime(data.index)

    plot_boxplot_chart(data, n_days=10, column='Close', title='10-Day Rolling Boxplot of META Stock Prices')
