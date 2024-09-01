import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import os
import tensorflow as tf
import load_process_data as data_function
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM



if __name__ == "__main__":

    # Setting up the date ranges
    start_date = '2012-01-01'
    end_date = '2020-01-01'

    test_start = '2019-01-01'
    test_end = '2020-01-01'

    # Function Parameters
    company = 'META'
    local_path = f'{company}_Data.csv'
    scale_features = True
    save_scaler = True

    # Load and process data
    train_data, test_data, scalers = data_function.load_and_process_data(
        ticker=company,
        start_date=start_date,
        end_date=end_date,
        nan_method='fill',
        split_method='date',
        split_date='2020-01-01',
        save_data=True,
        local_path=local_path,
        scale_features=scale_features,
        save_scaler=save_scaler
    )

    # Prepare the test data with the new date range
    test_data = yf.download(company, start=test_start, end=test_end)

    if test_data.empty:
        raise ValueError("Test data is empty. Adjust the test date range or verify data availability.")

    scaler_data = scalers['Close'].transform(train_data['Close'].values.reshape(-1, 1))
    prediction_days = 10

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaler_data)):
        x_train.append(scaler_data[x-prediction_days:x, 0])
        y_train.append(scaler_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Corrected shape

    # Building the model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    '''Testing the model Accuracy on Existing Data'''

    # Prepare test data
    scaler_test_data = scalers['Close'].transform(test_data['Close'].values.reshape(-1, 1))

    # Ensure the correct slicing and dimension consistency
    model_input = np.concatenate((scaler_data[-prediction_days:], scaler_test_data))

    if len(model_input) <= prediction_days:
        print("Length of scaler_data:", len(scaler_data))
        print("Length of test_data:", len(scaler_test_data))
        print("Shape of model_input:", model_input.shape)
        raise ValueError("Not enough data to create test samples. Adjust 'prediction_days' or check the data input.")

    x_test = []
    for x in range(prediction_days, len(model_input)):
        x_test.append(model_input[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scalers['Close'].inverse_transform(predicted_prices)
    actual_prices = test_data['Close'].values

    '''Plotting the test predictions'''
    plt.plot(actual_prices, color="blue", label=f"Actual {company} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Prices")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f"{company} Share Price")
    plt.legend()
    plt.show()