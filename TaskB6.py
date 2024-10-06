import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
import tensorflow as tf
import statsmodels.api as sm
import math

# Parameters setup
N_STEPS = 60
LOOKUP_STEP = 15
SCALE = True
SHUFFLE = True
SPLIT_BY_DATE = False
TEST_SIZE = 0.2
FEATURE_COLUMNS = ["Adj Close", "Volume", "Open", "High", "Low"]

N_LAYERS = 2
CELL = GRU
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = False

LOSS = tf.keras.losses.Huber()
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 50

ticker = "META"
MODEL_NAME = f"{datetime.now().strftime('%Y-%m-%d')}_{ticker}-n_steps-{N_STEPS}-lookup-{LOOKUP_STEP}-layers-{N_LAYERS}"

# Load and process data
def load_and_process_data(ticker, start_date, end_date, handle_nan='fill', split_ratio=0.8, split_by_date=True, scale=True, save_local=True, local_file='data.csv'):
    directory = os.path.dirname(local_file)
    if save_local and not os.path.exists(directory):
        os.makedirs(directory)

    if save_local and local_file and os.path.exists(local_file):
        df = pd.read_csv(local_file, index_col='Date', parse_dates=True)
    else:
        df = yf.download(ticker, start=start_date, end=end_date)
        if save_local:
            df.to_csv(local_file)

    if handle_nan == 'drop':
        df.dropna(inplace=True)
    elif handle_nan == 'fill':
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    column_scaler = {}
    if scale:
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
        column_scaler = {col: scaler for col in df.columns}
        
        adjclose_scaler = MinMaxScaler()
        df['Adj Close'] = adjclose_scaler.fit_transform(df[['Adj Close']])
        column_scaler['Adj Close'] = adjclose_scaler

    if split_by_date:
        train_df = df[:int(len(df) * split_ratio)]
        test_df = df[int(len(df) * split_ratio):]
    else:
        train_df = df.sample(frac=split_ratio, random_state=42)
        test_df = df.drop(train_df.index)

    return {
        'train': train_df,
        'test': test_df,
        'column_scaler': column_scaler,
        'test_df': test_df,
    }, column_scaler

# Create LSTM model
def create_model(n_steps, n_features, loss='huber', units=256, cell=LSTM, n_layers=2, dropout=0.4, optimizer='adam', bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=True), input_shape=(n_steps, n_features)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(n_steps, n_features)))
        elif i == n_layers - 1:
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))
    
    model.add(Dense(1, activation='linear'))
    model.compile(loss=loss, optimizer=optimizer)
    return model

# Create multivariate sequences
def create_multivariate_sequences(data, feature_columns, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[feature_columns].iloc[i:i + n_steps].values)
        y.append(data['Adj Close'].iloc[i + n_steps])
    return np.array(X), np.array(y)

# Multistep prediction for LSTM
def multistep_predict(model, data, n_steps, k):
    last_sequence = data['test'][FEATURE_COLUMNS].values[-n_steps:]
    last_sequence = last_sequence.reshape((1, n_steps, len(FEATURE_COLUMNS)))
    predictions = []
    
    for step in range(k):
        prediction = model.predict(last_sequence)
        prediction = prediction.reshape((1, 1, 1))

        if SCALE:
            prediction = data['column_scaler']['Adj Close'].inverse_transform(prediction.reshape(-1, 1)).reshape(1, 1, 1)
        
        predictions.append(prediction[0][0][0])

        new_sequence = np.copy(last_sequence)
        new_sequence[:, -1, 0] = prediction[0, 0, 0]
        last_sequence = np.append(new_sequence[:, 1:, :], new_sequence[:, -1:, :], axis=1)

    return predictions

def train_arima_model(train_data, order=(5,1,0), seasonal_order=(0,0,0,0)):
    arima_model = sm.tsa.statespace.SARIMAX(train_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    arima_result = arima_model.fit(disp=False)
    return arima_result

def arima_predict(arima_model, start, end):
    predictions = arima_model.predict(start=start, end=end, dynamic=True)
    return predictions

def ensemble_predictions(lstm_preds, arima_preds, weight_lstm=0.5, weight_arima=0.5):
    combined_preds = (weight_lstm * lstm_preds) + (weight_arima * arima_preds)
    return combined_preds

def calculate_rmse(true_values, predicted_values):
    return math.sqrt(mean_squared_error(true_values, predicted_values))

def get_final_df(model, data, n_steps, look_ahead=LOOKUP_STEP):
    X_test = data["test"][FEATURE_COLUMNS].values
    X_test = np.array([X_test[i:i + n_steps] for i in range(len(X_test) - n_steps)])

    X_test = X_test.reshape((X_test.shape[0], n_steps, len(FEATURE_COLUMNS)))
    y_test = data["test"]['Adj Close'].values[n_steps:]  
    y_pred = model.predict(X_test)

    if SCALE:
        y_test = data["column_scaler"]["Adj Close"].inverse_transform(np.expand_dims(y_test, axis=0)).flatten()
        y_pred = data["column_scaler"]["Adj Close"].inverse_transform(y_pred).flatten()

    test_df = data["test_df"].copy()
    test_df = test_df.iloc[-len(y_test):].copy()

    test_df[f"adjclose_{look_ahead}"] = y_pred
    test_df[f"true_adjclose_{look_ahead}"] = y_test
    return test_df

def plot_graph(test_df):
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.plot(test_df[f'ensemble_adjclose_{LOOKUP_STEP}'], c='g')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "LSTM Predicted", "Ensemble Predicted"])
    plt.show()

data, scalers = load_and_process_data(ticker=ticker, start_date='2010-01-01', end_date='2020-01-01',
                                      handle_nan='fill', split_ratio=0.8, split_by_date=True, 
                                      scale=True, save_local=True, local_file=f'data/{ticker}_data.csv')

X_train, y_train = create_multivariate_sequences(data['train'], FEATURE_COLUMNS, N_STEPS)
X_test, y_test = create_multivariate_sequences(data['test'], FEATURE_COLUMNS, N_STEPS)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(FEATURE_COLUMNS)))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(FEATURE_COLUMNS)))

model = create_model(
    n_steps=N_STEPS,
    n_features=len(FEATURE_COLUMNS),
    loss=LOSS,
    units=UNITS,
    cell=CELL,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    optimizer=OPTIMIZER,
    bidirectional=BIDIRECTIONAL
)

checkpoint_path = os.path.join("results", MODEL_NAME + ".keras")
tensorboard_log_dir = os.path.join("logs", MODEL_NAME)

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

tensorboard = TensorBoard(
    log_dir=tensorboard_log_dir
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, tensorboard],
    verbose=1
)

train_adjclose = data['train']['Adj Close']
arima_model = train_arima_model(train_adjclose)

lstm_preds = multistep_predict(model, data, N_STEPS, LOOKUP_STEP)
arima_preds = arima_predict(arima_model, start=len(train_adjclose), end=len(train_adjclose) + LOOKUP_STEP - 1)

# Combine ARIMA and LSTM predictions
combined_preds = ensemble_predictions(np.array(lstm_preds), np.array(arima_preds), weight_lstm=0.6, weight_arima=0.4)

# Get final DataFrame
final_df = get_final_df(model, data, N_STEPS, LOOKUP_STEP)

# Ensure the length of combined predictions matches the length of the relevant test data
final_df[f'ensemble_adjclose_{LOOKUP_STEP}'] = np.nan

final_df.iloc[-LOOKUP_STEP:, final_df.columns.get_loc(f'ensemble_adjclose_{LOOKUP_STEP}')] = combined_preds

rmse_df = final_df[[f'true_adjclose_{LOOKUP_STEP}', f'ensemble_adjclose_{LOOKUP_STEP}']].dropna()

# Calculate RMSE for ensemble
rmse = calculate_rmse(rmse_df[f'true_adjclose_{LOOKUP_STEP}'], rmse_df[f'ensemble_adjclose_{LOOKUP_STEP}'])
print(f'Ensemble RMSE: {rmse}')

# Plot graph
plot_graph(final_df)
