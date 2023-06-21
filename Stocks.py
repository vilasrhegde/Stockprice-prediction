from matplotlib.pyplot import title
import streamlit as st
from datetime import date
import yfinance as yf
# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = '2015-01-01'
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Predict the Stock on Web')

stocks = ('AAPL','GOOG','MSFT','GME')
selected_stock = st.selectbox('Select dataset for predictions',stocks)

n_years = st.slider('Years of Predictions: ',1,4)
period = n_years * 365


#load the stock data

@st.cache
def load_data(ticker):
    data =yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True) #sets put date in first column
    return data
data_load_state = st.text('Load data...')
data = load_data(selected_stock)    
data_load_state .text('Loading data...Done!')

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'] , y=data['Open'], name='Stock_open' ))
    fig.add_trace(go.Scatter(x=data['Date'] , y=data['Close'], name='Stock_close' ))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

st.subheader('Data Info')
st.write(data.describe())

#Forecasting: we need fbprophet from now on

# df_train = data[['Date','Close']]
# df_train = df_train.rename(columns={"Date": "ds", "Close": "y"}) #fb needs data like this names

# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=period)
# forecast = m.predict(future)

# st.subheader('Forecast Data: ')
# st.write(forecast.tail())

# st.write('Forecast Data')
# fig1=plot_plotly(m,forecast)
# st.plotly_chart(fig1)

# st.write('Forecast Component')
# fig2 = m.plot_component(forecast)
# st.write(fig2)