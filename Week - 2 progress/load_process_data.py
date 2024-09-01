import pandas as pd
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(ticker, start_date, end_date, nan_method='drop', split_method='ratio',
                          train_size=0.8, test_size=0.2, split_date=None, random_state=None,
                          save_data=False, local_path=None, scale_features=False, save_scaler=False):
    
    # Load data locally if available
    if local_path and os.path.exists(local_path):
        data = pd.read_csv(local_path, index_col='Date', parse_dates=True)
    else:
        # Download the data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Save the data locally
        if save_data and local_path:
            data.to_csv(local_path)
    
    # Handle NaN values
    if nan_method == 'drop':
        data = data.dropna()
    elif nan_method == 'fill':
        data = data.ffill().bfill()
    
    # Ensure data is not empty
    if data.empty:
        raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}. Please check the date range and ticker symbol.")

    # Split the data
    if split_method == 'ratio':
        train_data, test_data = train_test_split(data, train_size=train_size, test_size=test_size, random_state=random_state, shuffle=True)
    elif split_method == 'date' and split_date:
        train_data = data[:split_date]
        test_data = data[split_date:]

    # Scale features if required
    scalers = {}
    if scale_features:
        for column in train_data.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_data[column] = scaler.fit_transform(train_data[[column]])
            if not test_data.empty:
                test_data[column] = scaler.transform(test_data[[column]])
            if save_scaler:
                scalers[column] = scaler
    
    return train_data, test_data, scalers