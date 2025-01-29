# Import libraries needed
# Imports stock market data
import yfinance as yf

# Used for analyzing data from the stock market
import pandas as pd

# Creates the graphing visuals
import matplotlib.pyplot as plt

# TensorFlow for stock prediction using neural networks
import tensorflow as tf

# Numpy for handling numerical operations
import numpy as np

# Streamlit for creating the web app interface
import streamlit as st

# Create a function to retrieve the stock's data
def fetch_stock_data(ticker, start_date, end_date):
    # Download the stock data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    # Return the data
    return stock_data

# Create a function to prepare the data for training
def prepare_data(stock_data):
    # Normalize the stock data (using only the 'Close' price)
    stock_data = stock_data[['Close']].values
    stock_data = stock_data.astype('float32')
    
    # Split the data into training and test sets (80% train, 20% test)
    train_size = int(len(stock_data) * 0.8)
    train_data, test_data = stock_data[:train_size], stock_data[train_size:]
    
    # Prepare X and Y for training
    def create_dataset(data, window_size=30):
        X, Y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size, 0])
            Y.append(data[i+window_size, 0])
        return np.array(X), np.array(Y)
    
    X_train, Y_train = create_dataset(train_data)
    X_test, Y_test = create_dataset(test_data)

    # Debugging the dataset sizes
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)

    # Ensure the data is correctly shaped for LSTM
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Insufficient data to create train/test sets")

    # Reshape the input for the LSTM model
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, Y_train, X_test, Y_test, test_data

# Create the LSTM model for stock price prediction
def build_lstm_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(50, return_sequences=False))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create a function to visualize the stock prices
def plot_stock_data(stock_data, ticker, predicted_prices, future_dates):
    # Ensure the lengths of future_dates and predicted_prices match
    future_dates = future_dates[:len(predicted_prices)]

    # Create a graph plot with x and y axes
    plt.figure(figsize=(14, 7))

    # Plot the actual closing prices
    plt.plot(stock_data['Close'], label=f'{ticker} Closing Prices', color='green')

    # Plot the predicted prices
    plt.plot(future_dates, predicted_prices, label='Predicted Prices', color='red', linestyle='--')

    # Add labels and title
    plt.title(f'{ticker} Stock Prices with Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')

    # Include a legend and a grid
    plt.legend()
    plt.grid()

    # Display the plot in the Streamlit app
    st.pyplot(plt)

# Console output function to print predictions
def print_predictions(ticker, predicted_prices, future_dates):
    print(f"Predictions for {ticker}:")
    for date, price in zip(future_dates, predicted_prices):
        print(f"Date: {date.strftime('%Y-%m-%d')}, Predicted Price: ${price:.2f}")

# Streamlit app interface
st.title("Stock Price Prediction with LSTM Neural Network")

# Get user input to enter the stock ticker, start date, and end date
ticker = st.text_input("Enter the stock ticker (e.g., GOOGL for Google, AAPL for Apple):")
start_date = st.date_input("Enter the start date:")
end_date = st.date_input("Enter the end date:")

# If the button is clicked, execute the prediction and plot the results
if st.button("Predict Stock Prices"):

    # Get the stock data based on the user's input
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Check if the stock data exists
    if stock_data.empty:
        st.error("Invalid stock ticker or no data available for the selected dates.")
    else:
        # Prepare the data for the LSTM model
        try:
            X_train, Y_train, X_test, Y_test, test_data = prepare_data(stock_data)
            
            # Build and train the LSTM model
            model = build_lstm_model((X_train.shape[1], 1))
            model.fit(X_train, Y_train, batch_size=1, epochs=1, verbose=1)
            
            # Predict future stock prices
            predictions = model.predict(X_test)
            
            # Scale back the predictions to match the original stock prices
            predicted_prices = predictions.flatten()

            # Create predicted dates based on the latest date in stock data
            future_dates = pd.date_range(start=stock_data.index[-1], periods=len(predicted_prices) + 1, freq='D')[1:]
            
            # Print predictions to console
            print_predictions(ticker, predicted_prices, future_dates)
            
            # Plot the stock data with predicted prices
            plot_stock_data(stock_data, ticker, predicted_prices, future_dates)
        
        except ValueError as e:
            st.error(f"Error: {e}")
