import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
import tensorflow as tf

# Parameters setup
N_STEPS = 60
LOOKUP_STEP = 15
SCALE = True
SHUFFLE = True
SPLIT_BY_DATE = False
TEST_SIZE = 0.2
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

N_LAYERS = 2
CELL = GRU
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = False

LOSS = tf.keras.losses.Huber()
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 120

ticker = "AAPL"
MODEL_NAME = f"{datetime.now().strftime('%Y-%m-%d')}_{ticker}-n_steps-{N_STEPS}-lookup-{LOOKUP_STEP}-layers-{N_LAYERS}"

# Load and process data
def load_and_process_data(ticker, start_date, end_date, handle_nan='fill', split_ratio=0.8, split_by_date=True, scale=True, save_local=True, local_file='data.csv'):
    # Ensure the directory exists
    directory = os.path.dirname(local_file)
    if save_local and not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist
        print(f"Directory '{directory}' created.")

    # Load the data from Yahoo Finance or a local file
    if save_local and local_file and os.path.exists(local_file):
        df = pd.read_csv(local_file, index_col='Date', parse_dates=True)
        print(f"Data loaded from local file: {local_file}")
    else:
        df = yf.download(ticker, start=start_date, end=end_date)
        if save_local:
            df.to_csv(local_file)
            print(f"Data downloaded for ticker: {ticker}")
            print(f"Data saved locally at: {local_file}")

    # Handle NaN values
    if handle_nan == 'drop':
        df.dropna(inplace=True)
        print("NaN values dropped from dataset.")
    elif handle_nan == 'fill':
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        print("NaN values filled using forward and backward fill methods.")

    # Scale the features
    column_scaler = {}
    if scale:
        # Create separate scalers for all features and the target ('Adj Close')
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
        column_scaler = {col: scaler for col in df.columns}
        
        # Scale 'Adj Close' separately
        adjclose_scaler = MinMaxScaler()
        df['Adj Close'] = adjclose_scaler.fit_transform(df[['Adj Close']])
        column_scaler['Adj Close'] = adjclose_scaler
        
        print("Feature columns scaled using MinMaxScaler.")
    else:
        column_scaler = {}

    # Split data into train/test sets
    if split_by_date:
        train_df = df[:int(len(df) * split_ratio)]
        test_df = df[int(len(df) * split_ratio):]
    else:
        train_df = df.sample(frac=split_ratio, random_state=42)
        test_df = df.drop(train_df.index)
    
    print(f"Data split by {'date' if split_by_date else 'random sampling'} with train size: {train_df.shape}, test size: {test_df.shape}")

    return {
        'train': train_df,
        'test': test_df,
        'column_scaler': column_scaler,
        'test_df': test_df,
    }, column_scaler

# Create model
def create_model(n_steps, n_features, loss='huber', units=256, cell=GRU, n_layers=2, dropout=0.4, optimizer='adam', bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # First layer
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=True), input_shape=(n_steps, n_features)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(n_steps, n_features)))
        elif i == n_layers - 1:
            # Last layer
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # Hidden layers
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))
    
    model.add(Dense(1, activation='linear'))
    model.compile(loss=loss, optimizer=optimizer)
    print("Model created and compiled.")
    return model

# Create sequences for training
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Iterative prediction logic
def iterative_predict(model, data, n_steps, future_steps):
    # Retrieve the last 'n_steps' from the test set (ensure the correct column is used)
    last_sequence = data['test']['Adj Close'].values[-n_steps:]  # Use the correct column name

    # Reshape to match the model's expected input shape: (1, N_STEPS, 1)
    last_sequence = last_sequence.reshape((1, n_steps, 1))

    predictions = []
    
    for step in range(future_steps):
        # Predict the next step
        prediction = model.predict(last_sequence)

        # Reshape the prediction to match the shape needed for appending (1, 1)
        prediction = prediction.reshape((1, 1, 1))

        # If scaling, apply inverse transform using the 'Adj Close' scaler only
        if SCALE:
            prediction = data['column_scaler']['Adj Close'].inverse_transform(prediction.reshape(-1, 1)).reshape(1, 1, 1)

        # Store the predicted price
        predictions.append(prediction[0][0][0])

        # Update the sequence with the new prediction for the next step
        # Remove the first element and append the prediction at the end
        last_sequence = np.append(last_sequence[:, 1:, :], prediction, axis=1)
    
    return predictions

def get_final_df(model, data, n_steps, look_ahead=LOOKUP_STEP):
    # Prepare the test data
    X_test = data["test"]['Adj Close'].values
    X_test = np.array([X_test[i:i + n_steps] for i in range(len(X_test) - n_steps)])

    # Reshape X_test to be in the form (batch_size, n_steps, n_features)
    X_test = X_test.reshape((X_test.shape[0], n_steps, 1))

    y_test = data["test"]['Adj Close'].values[n_steps:]  # True future prices

    # Perform prediction on the test set
    y_pred = model.predict(X_test)

    # If scaling is enabled, inverse transform the predicted and true values
    if SCALE:
        y_test = data["column_scaler"]["Adj Close"].inverse_transform(np.expand_dims(y_test, axis=0)).flatten()
        y_pred = data["column_scaler"]["Adj Close"].inverse_transform(y_pred).flatten()

    # Create a copy of the test DataFrame
    test_df = data["test_df"].copy()

    # Ensure the test DataFrame has enough rows to match the predicted values
    test_df = test_df.iloc[-len(y_test):].copy()

    # Add predicted future prices and true future prices to the DataFrame
    test_df[f"adjclose_{look_ahead}"] = y_pred
    test_df[f"true_adjclose_{look_ahead}"] = y_test

    return test_df


# Plotting function
def plot_graph(test_df):
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

# Load and process data
data, scalers = load_and_process_data(ticker=ticker, start_date='2010-01-01', end_date='2020-01-01',
                                      handle_nan='fill', split_ratio=0.8, split_by_date=True, 
                                      scale=True, save_local=True, local_file=f'data/{ticker}_data.csv')

# Print the columns to inspect the available data
print("Columns in the DataFrame:", data['train'].columns)

train_data = data['train']['Adj Close'].values
test_data = data['test']['Adj Close'].values

X_train, y_train = create_sequences(train_data, N_STEPS)
X_test, y_test = create_sequences(test_data, N_STEPS)

# Reshape data for model input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Create model
model = create_model(
    n_steps=N_STEPS,
    n_features=1,
    loss=LOSS,
    units=UNITS,
    cell=CELL,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    optimizer=OPTIMIZER,
    bidirectional=BIDIRECTIONAL
)

# Callbacks
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

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, tensorboard],
    verbose=1
)

# Predict future prices using iterative prediction
future_steps = LOOKUP_STEP  # You can adjust the number of steps to predict
predicted_prices = iterative_predict(model, data, N_STEPS, future_steps)

final_df = get_final_df(model, data, N_STEPS, LOOKUP_STEP)

# Plot true vs. predicted prices
plot_graph(final_df)
