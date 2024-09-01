import yfinance as yf
import mplfinance as mpf
import pandas as pd

def plot_candlestick_chart(data, title='Candlestick Chart', n_days=1, save_as=None):

    if n_days < 1:
        raise ValueError("n_days must be greater than or equal to 1.")

    # Resample the data if n_days > 1
    if n_days > 1:
        data_resampled = data.resample(f'{n_days}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    else:
        data_resampled = data

    #Ploting the Chart
    if save_as:
        mpf.plot(
            data_resampled,
            type='candle',
            title=title,
            style='charles',
            savefig=save_as
        )
    else:
        mpf.plot(
            data_resampled,
            type='candle',
            title=title,
            style='charles',
        )
        mpf.show()


if __name__ == "__main__":
    # Download stock data
    data = yf.download('META', start='2022-01-01', end='2023-01-01')

    # Convert the index to datetime and ensure the data is properly formatted
    data.index = pd.to_datetime(data.index)

    # Plotting chart grouping every 5 days
    plot_candlestick_chart(data, title='META. Candlestick Chart', n_days=5)
