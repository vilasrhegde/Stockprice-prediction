import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import streamlit as st
from datetime import date
import plotly.express as px
# from forex_python.converter import CurrencyRates
import requests
import time




st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://vilasrhegde.github.io/',
        'Report a bug': "https://www.linkedin.com/in/vilasrhegde/",
        'About': "# Made by Vilas Hegde."
    }
)




# Collect historical stock prices data
def stockPred():
    st.sidebar.markdown('# Select a stock')
    option = st.sidebar.selectbox(
        '',
        ('IBM','INFY','TCS','SBI','NVDA','TSLA','GOOGL','AAPL','AMZN','MSFT','DIS', 'ORCL','LVMUY','KO','INTC','UNH','BABA','MA','V','JPM','NSRGY','TSM','JNJ','META','WMT','TCEHY'))

    if not option:
        st.error('Please select one stock.')
        st.stop()




    start_date = st.sidebar.date_input(
        "Start date",
        value=date(2018,1,1))
    end_date = st.sidebar.date_input(
        "End date",
        date.today(),
        disabled=False
        )
    @st.cache_data
    def get_data(option,start_date,end_date):
        data = yf.download(option, start=start_date, end=end_date)
        return data
    df=get_data(option,start_date,end_date)
    st.write(df.describe())
    fig = px.line(df,y=df.columns[0:4:3],color_discrete_sequence=['red', 'blue'],title=option)
    
    st.plotly_chart(fig)
    # Preprocess data
    df = df[['Close']]
    # st.table(df.tail())


    # get converted dollar rate dynamically
    # c = CurrencyRates()
    # todays_USD = c.get_rate('USD', 'INR')
    url = 'https://api.exchangerate-api.com/v4/latest/USD'
    data= requests.get(url).json()
    currencies = data['rates']
    # converter = RealTimeCurrencyConverter(url)
    # st.json(data['rates'])
    st.write('1 USD =',data['rates']['INR'],'INR')

    

    # currency = st.sidebar.selectbox("Select alternate currency",
    #                         data['rates'].keys())

    todays_USD = data['rates']['INR']
    last_close = round(float(df['Close'][-1]* todays_USD),2) 
    st.write(f'## Last closing price `₹{last_close}`',f'({df["Close"][-1].round(2)}$)')
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)

    # Create sequences and labels
    seq_length = 20
    X = []
    y = []
    for i in range(seq_length, len(df)):
        X.append(df[i-seq_length:i])
        y.append(df[i])
    X = np.array(X)
    y = np.array(y)

    # build the model
    model = Sequential()
    model.add(SimpleRNN(32, input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Split data into training and testing sets
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    @st.cache_data
    def get_result(X_train, y_train):
        model.fit(X_train, y_train, epochs=100, batch_size=32)
        # Evaluate model
        mse = model.evaluate(X_test, y_test)
        st.write('Mean squared error:', round(mse,4))

        last_sequence = df[-seq_length:]
        last_sequence = np.reshape(last_sequence, (1, seq_length, 1))
        next_price = model.predict(last_sequence)
        next_price = scaler.inverse_transform(next_price)
        predicted = round(float(next_price)* todays_USD,2)

        next_price = next_price.round(2)
        # next_price =  np.around(next_price[0][0], decimals=3)
        result = f'## Predicted stock price:\n # `₹{str(predicted)}`  '
        st.success(result)

        diff = round(  predicted - last_close , 2)
        st.sidebar.metric(label='Prediction in ₹', value=predicted, delta=diff)


    # Create and train RNN model
    # if st.sidebar.button('Predict',use_container_width=True,type='primary'):
    #     # Predict stock prices
    #     get_result(X_train, y_train)
    get_result(X_train, y_train)
    
    


import streamlit as st
import bcrypt
import sqlite3

def create_user_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                 username TEXT NOT NULL, 
                 password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def signup():
    if  len(st.session_state['username'])>0:
        st.write("You are already logged in as:", st.session_state['username'])
        stockPred()
        return

    st.write("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    if st.button("Signup"):
        if password == confirm_password:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            conn.close()
            st.success("You have successfully created an account!")
            st.info("Please login to proceed.")
        else:
            st.warning("Passwords do not match.")

def login():
    # session = get(username='')
    if len(st.session_state['username'])>0:
        st.write("### Hey",':blue[', st.session_state.username.capitalize()+' 👋',']')
        stockPred()
        return

    st.write("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        result = c.fetchone()
        conn.close()
        if result:
            hashed_password = result[2]
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                st.session_state['username'] = username
                st.success("You have successfully logged in!")
                stockPred()
            else:
                st.warning("Incorrect Password")
                st.stop()
        else:
            st.markdown("## :red[Username not found, Please Sign Up!]")
            st.stop()

def logout():
    
    if len(st.session_state['username'])<=0:
        st.warning("You have not logged in yet!")
        return

    st.session_state['username'] = ''
    st.success("You have successfully logged out!")




def main():

    create_user_table()
    st.title("Stock Price Prediction")

    # menu = ["Login", "Signup", "Logout"]
    # choice = "Signup"
    # choice = st.sidebar.selectbox("Select an option", menu)
    # if choice == "Signup":
    #     signup()
    # elif choice == "Login" :
    #     if st.session_state['username']!='':
    #         stockPred()
    #     else:
    #         login()
    # elif choice == "Logout":
    #     logout()
    # else:

    stockPred()

if __name__ == '__main__':
    st.session_state['username']=''
    main()

