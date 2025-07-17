print("Testing python on the remote server ")

# Replace these values with your actual API credentials
client_id = 'V9GQM61IVI-100'
secret_key = '3OH8C9ELBB'
redirect_uri ='https://127.0.0.1:5000/login'

strategy_name='renko_option_buying'

index_name='NIFTY50'
exchange='NSE'
ticker=f"{exchange}:{index_name}-INDEX"
# ticker='MCX:CRUDEOILM25MARFUT'
#ticker='MCX:NATURALGAS25FEBFUT'
strike_count=10
strike_diff=100
account_type='PAPER'

if exchange=='NSE':
    time_zone="Asia/Kolkata"

start_hour,start_min=9,30
end_hour,end_min=15,15
quantity=60

trail_point=3
don_chain_lower_period=20
don_chain_upper_period=20

brick_size=0 #spot/1000
buffer=0  #half of brick size
take_profit_point=30
stop_loss_point=30



# Import the required module from the fyers_apiv3 package
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pendulum as dt
import pandas_ta as ta
import mplfinance as mpf
import numpy as np
import pickle
import time
import webbrowser
import os
import sys
import certifi
import pytz

# @title

#for windows ssl error
os.environ['SSL_CERT_FILE'] = certifi.where()


#disable fyersApi and Fyers Request logs
import logging


#logging to file
logging.basicConfig(level=logging.INFO, filename=f'{strategy_name}_{dt.now(time_zone).date()}.log',filemode='a',format="%(asctime)s - %(message)s")


# Check if access.txt exists, then read the file and get the access token
if os.path.exists(f'access-{dt.now(time_zone).date()}.txt'):
    print('access token exists')
    with open(f'access-{dt.now(time_zone).date()}.txt', 'r') as f:
        access_token = f.read()

else:
    # Define response type and state for the session
    response_type = "code"
    state = "sample_state"
    try:
        # Create a session model with the provided credentials
        session = fyersModel.SessionModel(
            client_id=client_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type=response_type
        )

        # Generate the auth code using the session model
        response = session.generate_authcode()

        # Print the auth code received in the response
        print(response)

        # Open the auth code URL in a new browser window
        webbrowser.open(response, new=1)
        newurl = input("Enter the url: ")
        auth_code = newurl[newurl.index('auth_code=')+10:newurl.index('&state')]

        # Define grant type for the session
        grant_type = "authorization_code"
        session = fyersModel.SessionModel(
            client_id=client_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type=response_type,
            grant_type=grant_type
        )

        # Set the authorization code in the session object
        session.set_token(auth_code)

        # Generate the access token using the authorization code
        response = session.generate_token()

        # Save the access token to access.txt
        access_token = response["access_token"]
        with open(f'access-{dt.now(time_zone).date()}.txt', 'w') as k:
            k.write(access_token)
    except Exception as e:
        # Print the exception and response for debugging
        print(e, response)
        print('unable to get access token')
        sys.exit()

# Print the access token
print('access token:', access_token)


# Get the current time
current_time=dt.now(time_zone)
start_time=dt.datetime(current_time.year,current_time.month,current_time.day,start_hour,start_min,tz=time_zone)
end_time=dt.datetime(current_time.year,current_time.month,current_time.day,end_hour,end_min,tz=time_zone)
print('start time:', start_time)
print('end time:', end_time)


# Initialize FyersModel instances for synchronous and asynchronous operations
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path=None)


# Define the data for the option chain request
data = {
    "symbol": ticker,
    "strikecount": strike_count,
    "timestamp": ""
}

# Get the expiry data from the option chain
response = fyers.optionchain(data=data)['data']
expiry = response['expiryData'][0]['date']
print("current_expiry selected", expiry)
expiry_e = response['expiryData'][0]['expiry']
print(expiry_e)
# Define the data for the option chain request with expiry
data = {
    "symbol": ticker,
    "strikecount": strike_count,
    "timestamp": expiry_e
}

# Get the option chain data
response = fyers.optionchain(data=data)['data']
option_chain = pd.DataFrame(response['optionsChain'])
print(option_chain)
symbols = option_chain['symbol'].to_list()


#get strike_diff
strikes=option_chain[option_chain['option_type'].isin(['CE','PE'])]['strike_price'].to_list()
strikes=list(set(strikes))
strikes.sort()
strike_diff=strikes[-1]-strikes[-2]
print('strike diff:',strike_diff)


# Get the current spot price
spot_price = option_chain['ltp'].iloc[0]
print('current spot price is', spot_price)





f = dt.now(time_zone).date() - dt.duration( days=5)
p = dt.now(time_zone).date()

data = {
    "symbol": ticker,
    "resolution": "1",
    "date_format": "1",
    "range_from": f.strftime('%Y-%m-%d'),
    "range_to": p.strftime('%Y-%m-%d'),
    "cont_flag": "1"
}


# Fetch historical data
response2 =fyers.history(data=data)
hist_data = pd.DataFrame(response2['candles'])
hist_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
ist = pytz.timezone('Asia/Kolkata')
hist_data['date'] = pd.to_datetime(hist_data['date'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist)
data1=hist_data[hist_data['date'].dt.date<dt.now(time_zone).date()]




#filter data for current date
hist_data=hist_data[hist_data['date'].dt.date==dt.now(time_zone).date()]

#get today open price
open_price=hist_data['open'].iloc[0]

print('open price:',open_price)

if brick_size==0:
    brick_size=round(open_price/1000,2)

if buffer==0:
    buffer=brick_size/2

print('brick size:',brick_size)
print('buffer:',buffer)


def fetchOHLC(ticker,interval,duration):
    """extracts historical data and outputs in the form of dataframe"""
    instrument = ticker
    data = {"symbol":instrument,"resolution":interval,"date_format":"1","range_from":dt.now(time_zone).date()-dt.duration( days=duration),"range_to":dt.now(time_zone).date(),"cont_flag":"1",'oi_flag':"1"}
    sdata=fyers.history(data)
    # print(sdata)
    sdata=pd.DataFrame(sdata['candles'])
    # print(sdata)
    column_no=len(sdata.columns)
    # print(column_no)
    if column_no==6:

        sdata.columns=['date','open','high','low','close','volume']
    else:
        sdata.columns=['date','open','high','low','close','volume','oi']
    sdata['date']=pd.to_datetime(sdata['date'], unit='s')
    sdata.date=(sdata.date.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata'))
    sdata['date'] = sdata['date'].dt.tz_localize(None)
    sdata=sdata.set_index('date')
    # sys.exit()
    # sdata['atr']=ta.atr(sdata['high'],sdata['low'],sdata['close'],length=atr_length)

    # sdata=sdata.reset_index()[sdata.reset_index()['date'].dt.date<dt.datetime.now().date()].set_index('date')
    # print(sdata)

    return sdata


def candle_renko_refresh(ticker,data):
    print('candle_renko_refresh')


    calculated_values = {}
    matplotlib.use('Agg')  # Use a non-interactive backend
    mpf.plot(data, type='renko', renko_params=dict(brick_size=brick_size),return_calculated_values=calculated_values,returnfig=True,style='yahoo')
    plt.close()
    renko_df = pd.DataFrame(calculated_values)

    def count_bricks(sign_list):
        list1=[]
        pos_count=0
        neg_count=0
        for k in range(len(sign_list)):
            i=sign_list[k]
            if i>0:
                if sign_list[k-1]<0:
                    pos_count=1
                    list1.append(pos_count)
                else:
                    pos_count+=1
                    list1.append(pos_count)

            elif i<0:
                if sign_list[k-1]>0:
                    neg_count=-1
                    list1.append(neg_count)
                else:
                    neg_count-=1
                    list1.append(neg_count)
            else:
                list1.append(0)
        return list1


    renko_df.drop(columns=['renko_volumes','minx','maxx','miny','maxy'],inplace=True,axis=1)

    renko_df['pos_count']=count_bricks(renko_df['renko_bricks'].diff().tolist())



    high=[0]
    low=[0]
    open=[0]
    close=[0]
    #go through renko_bricks and pos_count columns
    for renko,pos in zip(renko_df['renko_bricks'],renko_df['pos_count']):


        if pos<0:
            open.append(renko)
            high.append(renko)
            close.append(renko-brick_size)
            low.append(renko-brick_size)

        elif pos>0:
            close.append(renko)
            high.append(renko)
            open.append(renko-brick_size)
            low.append(renko-brick_size)

    renko_df['OPEN'] = open
    renko_df['HIGH'] = high
    renko_df['LOW'] = low
    renko_df['CLOSE'] = close

    # calculate donchain channel
    renko_df.set_index('renko_dates',inplace=True)
    # print(renko_df)
    a = ta.donchian(renko_df["HIGH"], renko_df["LOW"], lower_length=don_chain_lower_period, upper_length=don_chain_upper_period)
    # print(a)
    renko_df['dc_lower'] = a[f'DCL_{don_chain_lower_period}_{don_chain_upper_period}']
    renko_df['dc_upper'] = a[f'DCU_{don_chain_lower_period}_{don_chain_upper_period}']
    d = ta.ao(renko_df['HIGH'], renko_df['LOW'], fast=5, slow=34)
    renko_df['awesome'] = d
    #filter data for current date
    renko_df=renko_df[renko_df.index.date==dt.now(time_zone).date()]
    return renko_df



# Function to get the OTM option based on spot price and side (CE/PE)
def get_otm_option(spot_price, side, points=100):
    if side == 'CE':
        otm_strike = (round(spot_price / strike_diff) * strike_diff) + points
    else:
        otm_strike = (round(spot_price / strike_diff) * strike_diff) - points
    otm_option = option_chain[(option_chain['strike_price'] == otm_strike) & (option_chain['option_type'] == side)]['symbol'].squeeze()
    return otm_option, otm_strike



call_option, call_buy_strike = get_otm_option(spot_price, 'CE', 0)
put_option, put_buy_strike = get_otm_option(spot_price, 'PE', 0)
print('call option:', call_option)
print('put option:', put_option)

# Log the start of the strategy
logging.info('started')


# Function to store data using pickle
def store(data, account_type):
    pickle.dump(data, open(f'data-{dt.now(time_zone).date()}-{account_type}.pickle', 'wb'))

# Function to load data using pickle
def load(account_type):
    return pickle.load(open(f'data-{dt.now(time_zone).date()}-{account_type}.pickle', 'rb'))

# Function to place a limit order
def take_limit_position(ticker, action, quantity):
    # limit_price=int(limit_price)
    n = {"symbols":ticker}
    current_price=fyers.quotes(n)['d'][0].get('v')
    price=round((current_price.get('ask')+current_price.get('bid'))/2,2)

    try:
        data = {
            "symbol": ticker,
            "qty": quantity,
            "type": 1,
            "side": action,
            "productType": "INTRADAY",
            "limitPrice": int(price),
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0
        }
        response3 = fyers.place_order(data=data)
        logging.info(response3)
        print(response3)
    except Exception as e:
        logging.info(e)
        print(e)
        print('unable to place order for some reason')

# Load or initialize paper trading information
if account_type == 'PAPER':
    try:
        paper_info = load(account_type)

    except:
        column_names = ['time', 'ticker', 'price', 'action', 'stop_price', 'take_profit', 'spot_price', 'quantity']
        filled_df = pd.DataFrame(columns=column_names)
        filled_df.set_index('time', inplace=True)
        paper_info={}
        paper_info.update({ticker:{"call_name":call_option,'put_name':put_option,'call_flag':0,'put_flag':0, 'brick_size' : brick_size,'no_of_trades':0,'buffer':buffer,'stop_price':0,'take_profit':0,'initial_quantity':0,'current_quantity':0,'entry_price':0,'call_entry_price':0,'put_entry_price':0}})
        paper_info.update({'filled_df':filled_df})

else:
    try:
        real_info = load(account_type)
    except:
        column_names = ['time', 'ticker', 'price', 'action', 'stop_price', 'take_profit', 'spot_price', 'quantity']
        filled_df = pd.DataFrame(columns=column_names)
        filled_df.set_index('time', inplace=True)
        real_info={}
        real_info.update({ticker:{"call_name":call_option,'put_name':put_option,'call_flag':0,'put_flag':0, 'brick_size' : brick_size,'no_of_trades':0,'buffer':buffer,'stop_price':0,'take_profit':0,'initial_quantity':0,'current_quantity':0,'entry_price':0}})
        real_info.update({'filled_df':filled_df})


def paper_order(spot_price,renko_df):

    global paper_info
    global brick_size,buffer,ticker,quantity


    renko_low=round(renko_df.iloc[-1].loc['LOW'].squeeze(),2)
    renko_high=round(renko_df.iloc[-1].loc['HIGH'].squeeze(),2)
    renko_close=round(renko_df.iloc[-1].loc['CLOSE'].squeeze(),2)

    chain_low=round(renko_df.iloc[-1].loc['dc_lower'].squeeze(),2)
    chain_high=round(renko_df.iloc[-1].loc['dc_upper'].squeeze(),2)
    awesome=round(renko_df.iloc[-1].loc['awesome'].squeeze(),2)

    call_entry_price=paper_info.get(ticker).get('call_entry_price')
    put_entry_price=paper_info.get(ticker).get('put_entry_price')

    # Get the current time
    ct = dt.now(time_zone)
    print(ct)
    print('spot price',spot_price)
    print('brick size',brick_size)
    print('buffer',buffer)
    # print('renko_low',renko_low)
    # print('renko_high',renko_high)
    # print('renko_close',renko_close)
    print('chain_low',chain_low)
    print('chain_high',chain_high)
    print('awesome',awesome)
    print('no of bricks',renko_df['pos_count'].iloc[-1])


    # Check if the current time is greater than the start time
    if ct > start_time<end_time:


        # Get option names
        call_name = paper_info.get(ticker).get('call_name')
        put_name = paper_info.get(ticker).get('put_name')

        # Get trade flags
        call_flag = paper_info.get(ticker).get('call_flag')
        put_flag = paper_info.get(ticker).get('put_flag')

        #spot stop prices
        spot_stop_price=paper_info.get(ticker).get('stop_price')
        spot_profit_price=paper_info.get(ticker).get('take_profit')

        #quantity
        initial_quantity=paper_info.get(ticker).get('initial_quantity')
        current_quantity=paper_info.get(ticker).get('current_quantity')




        if (chain_high <= renko_high) and (call_flag == 0 and put_flag==0) and (spot_price>renko_high+buffer) and (awesome>0):
            call_name=get_otm_option(spot_price, 'CE', 0)[0]
            n = {"symbols":call_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            call_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            spot_profit_price=spot_price+take_profit_point
            spot_stop_price=spot_price-stop_loss_point
            a = [call_name, call_price, 'BUY', spot_stop_price, spot_profit_price, spot_price, quantity]
            paper_info['filled_df'].loc[dt.now(time_zone)] = a
            paper_info.get(ticker).update({'call_name':call_name,'call_flag':1,'initial_quantity':quantity,
                                           'current_quantity':quantity,'stop_price':spot_stop_price,
                                           'take_profit':spot_profit_price,'entry_price':spot_price,'call_entry_price':call_price})
            logging.info(f'Call buy condition satisfied: {call_name} at {call_price}')

        # Check call close and trail condition
        elif call_flag == 1:
            n = {"symbols":call_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            call_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            print('current price:',call_price)
            print('call pnl:',(call_price-paper_info.get(ticker).get('call_entry_price'))*paper_info.get(ticker).get('current_quantity'))

            #half quantity close at profit
            #closing current position

            if (spot_price < spot_stop_price) :
                paper_info.get(ticker).update({'current_quantity':0,'initial_quantity':0,'stop_price':0,'take_profit':0,'call_flag':0,'call_entry_price':0})
                a = [call_name, call_price, 'SELL', 0, 0, spot_price, 0]
                paper_info['filled_df'].loc[dt.now(time_zone)] = a  # Update dataframe
                logging.info(f'Call sell condition satisfied: {call_name} at {call_price}')

            #half quantity close at profit
            elif (spot_price > spot_profit_price) and (initial_quantity==current_quantity):
                paper_info.get(ticker).update({'current_quantity':initial_quantity//2,'stop_price':spot_profit_price-stop_loss_point,'take_profit':0,})
                a = [call_name, call_price, 'SELL', spot_profit_price-stop_loss_point, 0, spot_price, initial_quantity//2]
                paper_info['filled_df'].loc[dt.now(time_zone)] = a  # Update dataframe
                logging.info(f'Call sell condition satisfied to close half quantity: {call_name} at {call_price}')

            #trail stop loss
            elif (initial_quantity!=current_quantity):
                #trail stop loss
                if (spot_price>(spot_stop_price+stop_loss_point+trail_point)):
                    new_stop_price=spot_price-stop_loss_point
                    paper_info.get(ticker).update({'stop_price':new_stop_price})
                    a = [call_name, call_price, 'TRAIL', new_stop_price, 0, spot_price, current_quantity]
                    paper_info['filled_df'].loc[dt.now(time_zone)] = a
                    logging.info(f'updating stop price , current_stop is {spot_stop_price} new stop is {new_stop_price} current price is {spot_price}')


        # Check put buy condition
        if (chain_low >= renko_low) and (call_flag == 0 and put_flag==0) and (spot_price<renko_low-buffer) and (awesome<0):
            put_name=get_otm_option(spot_price, 'PE', 0)[0]
            n = {"symbols":put_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            put_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            spot_profit_price=spot_price-take_profit_point
            spot_stop_price=spot_price+stop_loss_point
            a = [put_name, put_price, 'BUY', spot_stop_price, spot_profit_price, spot_price, quantity]
            paper_info['filled_df'].loc[dt.now(time_zone)] = a
            paper_info.get(ticker).update({'put_name':put_name,'put_flag':1,'initial_quantity':quantity,
                                           'current_quantity':quantity,'stop_price':spot_stop_price,
                                           'take_profit':spot_profit_price,'entry_price':spot_price,'put_entry_price':put_price})
            logging.info(f'Put buy condition satisfied: {put_name} at {put_price}')

        # Check put close and trail condition
        elif put_flag == 1:
            n = {"symbols":put_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            put_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            print('put pnl:',(put_price-paper_info.get(ticker).get('put_entry_price'))*paper_info.get(ticker).get('current_quantity'))
            #closing current position
            if (spot_price > spot_stop_price):
                paper_info.get(ticker).update({'current_quantity':0,'initial_quantity':0,'stop_price':0,'take_profit':0,'put_flag':0,'put_entry_price':0})
                a = [put_name, put_price, 'SELL', 0, 0, spot_price, 0]
                paper_info['filled_df'].loc[dt.now(time_zone)] = a
                logging.info(f'Put sell condition satisfied: {put_name} at {put_price}')

            #half quantity close at profit
            elif (spot_price < spot_profit_price) and (initial_quantity==current_quantity):
                paper_info.get(ticker).update({'current_quantity':initial_quantity//2,'stop_price':spot_profit_price+stop_loss_point,'take_profit':0,})
                a = [put_name, put_price, 'SELL', spot_stop_price, 0, spot_price, initial_quantity//2]
                paper_info['filled_df'].loc[dt.now(time_zone)] = a
                logging.info(f'Put sell condition satisfied to close half quantity: {put_name} at {put_price}')

            #trail stop loss
            elif (initial_quantity!=current_quantity):
                #trail stop loss
                if (spot_price<(spot_stop_price-stop_loss_point-trail_point)):
                    new_stop_price=spot_price+stop_loss_point
                    paper_info.get(ticker).update({'stop_price':new_stop_price})
                    a = [put_name, put_price, 'TRAIL', new_stop_price, 0, spot_price, current_quantity]
                    paper_info['filled_df'].loc[dt.now(time_zone)] = a
                    logging.info(f'updating stop price , current_stop is {spot_stop_price} new stop is {new_stop_price} current price is {spot_price}')


        if not paper_info['filled_df'].empty:
                paper_info['filled_df'].to_csv(f'trades_{strategy_name}_{dt.now(time_zone).date()}.csv')

        # Store paper_info using pickle
        print('storing paper info')
        store(paper_info, account_type)

    elif ct>end_time:

        # Check call close and trail condition
        if call_flag == 1:
            n = {"symbols":call_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            call_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            paper_info.get(ticker).update({'current_quantity':0,'initial_quantity':0,'stop_price':0,'take_profit':0,'call_flag':3})
            a = [call_name, call_price, 'SELL', 0, 0, spot_price, 0]
            paper_info['filled_df'].loc[dt.now(time_zone)] = a  # Update dataframe
            logging.info(f'Call sell condition satisfied end time reached: {call_name} at {call_price}')

        if put_flag==1:
            n = {"symbols":put_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            put_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            paper_info.get(ticker).update({'current_quantity':0,'initial_quantity':0,'stop_price':0,'take_profit':0,'put_flag':3})
            a = [put_name, put_price, 'SELL', 0, 0, spot_price, 0]
            paper_info['filled_df'].loc[dt.now(time_zone)] = a
            logging.info(f'Put sell condition satisfied end time reached: {put_name} at {put_price}')


def real_order(spot_price,renko_df):

    global real_info
    global brick_size,buffer,ticker,quantity
    print(renko_df)
    renko_low=round(renko_df.iloc[-1].loc['LOW'].squeeze(),2)
    renko_high=round(renko_df.iloc[-1].loc['HIGH'].squeeze(),2)
    renko_close=round(renko_df.iloc[-1].loc['CLOSE'].squeeze(),2)

    chain_low=round(renko_df.iloc[-1].loc['dc_lower'].squeeze(),2)
    chain_high=round(renko_df.iloc[-1].loc['dc_upper'].squeeze(),2)
    awesome=round(renko_df.iloc[-1].loc['awesome'].squeeze(),2)

    call_entry_price=real_info.get(ticker).get('call_entry_price')
    put_entry_price=real_info.get(ticker).get('put_entry_price')

    # Get the current time
    ct = dt.now(time_zone)
    print(ct)
    print('spot price',spot_price)
    print('brick size',brick_size)
    print('buffer',buffer)
    # print('renko_low',renko_low)
    # print('renko_high',renko_high)
    # print('renko_close',renko_close)
    print('chain_low',chain_low)
    print('chain_high',chain_high)
    print('awesome',awesome)
    print('no of bricks',renko_df['pos_count'].iloc[-1])

    # Check if the current time is greater than the start time
    if ct > start_time<end_time:


        # Get option names
        call_name = real_info.get(ticker).get('call_name')
        put_name = real_info.get(ticker).get('put_name')

        # Get trade flags
        call_flag = real_info.get(ticker).get('call_flag')
        put_flag = real_info.get(ticker).get('put_flag')

        #spot stop prices
        spot_stop_price=real_info.get(ticker).get('stop_price')
        spot_profit_price=real_info.get(ticker).get('take_profit')

        #quantity
        initial_quantity=real_info.get(ticker).get('initial_quantity')
        current_quantity=real_info.get(ticker).get('current_quantity')



        #check call buy condition
        if (chain_high <= renko_high) and (call_flag == 0 and put_flag==0) and (spot_price>renko_high+buffer) and (awesome>0):
            call_name=get_otm_option(spot_price, 'CE', 0)[0]
            n = {"symbols":call_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            call_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            spot_profit_price=spot_price+take_profit_point
            spot_stop_price=spot_price-stop_loss_point
            a = [call_name, call_price, 'BUY', spot_stop_price, spot_profit_price, spot_price, quantity]
            real_info['filled_df'].loc[dt.now(time_zone)] = a
            real_info.get(ticker).update({'call_name':call_name,'call_flag':1,'initial_quantity':quantity,
                                           'current_quantity':quantity,'stop_price':spot_stop_price,
                                           'take_profit':spot_profit_price,'entry_price':spot_price,'call_entry_price':call_price})
            logging.info(f'Call buy condition satisfied: {call_name} at {call_price}')
            take_limit_position(call_name, 1, quantity, call_price)

        # Check call close and trail condition
        elif call_flag == 1:
            n = {"symbols":call_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            call_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            print('current price:',call_price)
            print('call pnl:',(call_price-real_info.get(ticker).get('call_entry_price'))*real_info.get(ticker).get('current_quantity'))



            if (spot_price < spot_stop_price) :
                real_info.get(ticker).update({'current_quantity':0,'initial_quantity':0,'stop_price':0,'take_profit':0,'call_flag':0,'call_entry_price':0})
                a = [call_name, call_price, 'SELL', 0, 0, spot_price, 0]
                real_info['filled_df'].loc[dt.now(time_zone)] = a  # Update dataframe
                logging.info(f'Call sell condition satisfied: {call_name} at {call_price}')
                data = {"id": call_name + "-INTRADAY"}
                response = fyers.exit_positions(data=data)


            #half quantity close at profit
            elif (spot_price > spot_profit_price) and (initial_quantity==current_quantity):
                real_info.get(ticker).update({'current_quantity':initial_quantity//2,'stop_price':spot_profit_price-stop_loss_point,'take_profit':0,})
                a = [call_name, call_price, 'SELL', spot_profit_price-stop_loss_point, 0, spot_price, initial_quantity//2]
                real_info['filled_df'].loc[dt.now(time_zone)] = a  # Update dataframe
                logging.info(f'Call sell condition satisfied to close half quantity: {call_name} at {call_price}')
                take_limit_position(call_name, -1, initial_quantity//2, call_price)

            #trail stop loss
            elif (initial_quantity!=current_quantity):
                #trail stop loss
                if (spot_price>(spot_stop_price+stop_loss_point+trail_point)):
                    new_stop_price=spot_price-stop_loss_point
                    real_info.get(ticker).update({'stop_price':new_stop_price})
                    a = [call_name, call_price, 'TRAIL', new_stop_price, 0, spot_price, current_quantity]
                    real_info['filled_df'].loc[dt.now(time_zone)] = a
                    logging.info(f'updating stop price , current_stop is {spot_stop_price} new stop is {new_stop_price} current price is {spot_price}')


        # Check put buy condition
        if (chain_low >= renko_low) and (call_flag == 0 and put_flag==0) and (spot_price<renko_low-buffer) and (awesome<0):
            put_name=get_otm_option(spot_price, 'PE', 0)[0]
            n = {"symbols":put_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            put_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            spot_profit_price=spot_price-take_profit_point
            spot_stop_price=spot_price+stop_loss_point
            a = [put_name, put_price, 'BUY', spot_stop_price, spot_profit_price, spot_price, quantity]
            real_info['filled_df'].loc[dt.now(time_zone)] = a
            real_info.get(ticker).update({'put_name':put_name,'put_flag':1,'initial_quantity':quantity,
                                           'current_quantity':quantity,'stop_price':spot_stop_price,
                                           'take_profit':spot_profit_price,'entry_price':spot_price,'put_entry_price':put_price})
            logging.info(f'Put buy condition satisfied: {put_name} at {put_price}')
            take_limit_position(put_name, 1, quantity, put_price)

        # Check put close and trail condition
        elif put_flag == 1:
            n = {"symbols":put_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            put_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            print('call pnl:',(call_price-real_info.get(ticker).get('put_entry_price'))*real_info.get(ticker).get('current_quantity'))




            #closing current position
            if (spot_price > spot_stop_price):
                real_info.get(ticker).update({'current_quantity':0,'initial_quantity':0,'stop_price':0,'take_profit':0,'put_flag':0,'put_entry_price':0})
                a = [put_name, put_price, 'SELL', 0, 0, spot_price, 0]
                real_info['filled_df'].loc[dt.now(time_zone)] = a
                logging.info(f'Put sell condition satisfied: {put_name} at {put_price}')
                data = {"id": put_name + "-INTRADAY"}
                response = fyers.exit_positions(data=data)


            #half quantity close at profit
            elif (spot_price < spot_profit_price) and (initial_quantity==current_quantity):
                real_info.get(ticker).update({'current_quantity':initial_quantity//2,'stop_price':spot_profit_price+stop_loss_point,'take_profit':0,})
                a = [put_name, put_price, 'SELL', spot_stop_price, 0, spot_price, initial_quantity//2]
                real_info['filled_df'].loc[dt.now(time_zone)] = a
                logging.info(f'Put sell condition satisfied to close half quantity: {put_name} at {put_price}')
                take_limit_position(put_name, -1, initial_quantity//2, put_price)

            #trail stop loss
            elif (initial_quantity!=current_quantity):
                #trail stop loss
                if (spot_price<(spot_stop_price-stop_loss_point-trail_point)):
                    new_stop_price=spot_price+stop_loss_point
                    real_info.get(ticker).update({'stop_price':new_stop_price})
                    a = [put_name, put_price, 'TRAIL', new_stop_price, 0, spot_price, current_quantity]
                    real_info['filled_df'].loc[dt.now(time_zone)] = a
                    logging.info(f'updating stop price , current_stop is {spot_stop_price} new stop is {new_stop_price} current price is {spot_price}')




    elif ct>end_time:

        # Check call close and trail condition
        if call_flag == 1:
            n = {"symbols":call_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            call_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            real_info.get(ticker).update({'current_quantity':0,'initial_quantity':0,'stop_price':0,'take_profit':0,'call_flag':3,'call_entry_price':0})
            a = [call_name, call_price, 'SELL', 0, 0, spot_price, 0]
            real_info['filled_df'].loc[dt.now(time_zone)] = a  # Update dataframe
            logging.info(f'Call sell condition satisfied end time reached: {call_name} at {call_price}')
            data = {"id": call_name + "-INTRADAY"}
            response = fyers.exit_positions(data=data)

        if put_flag==1:
            n = {"symbols":put_name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            put_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)
            real_info.get(ticker).update({'current_quantity':0,'initial_quantity':0,'stop_price':0,'take_profit':0,'put_flag':3,'put_entry_price':0})
            a = [put_name, put_price, 'SELL', 0, 0, spot_price, 0]
            real_info['filled_df'].loc[dt.now(time_zone)] = a
            logging.info(f'Put sell condition satisfied end time reached: {put_name} at {put_price}')
            data = {"id": put_name + "-INTRADAY"}
            response = fyers.exit_positions(data=data)

    if not real_info['filled_df'].empty:
        real_info['filled_df'].to_csv(f'trades_{strategy_name}_{dt.now(time_zone).date()}.csv')

        # Store real_info using pickle

    print('storing live info')
    store(real_info, account_type)
    #check real time pnl




def chase_order(ord_df):
    # Check if the order dataframe is not empty
    if not ord_df.empty:
        # Filter orders with status 6 (open orders)
        ord_df = ord_df[ord_df['status'] == 6]
        # Iterate through each order in the dataframe
        for i, o1 in ord_df.iterrows():
            # Get the symbol name from the order
            name = o1['symbol']
            # Get the current price of the symbol from the dataframe
            n = {"symbols":name}
            current_price=fyers.quotes(n)['d'][0].get('v')
            current_price=round((current_price.get('ask')+current_price.get('bid'))/2,2)

            try:
                # Check if the order type is limit order (type 1)
                if o1['type'] == 1:
                    # Get the order details
                    name = o1['symbol']
                    id1 = o1['id']
                    lmt_price = o1['limitPrice']
                    qty = o1['qty']
                    # Determine the new limit price based on the current price
                    if current_price > lmt_price:
                        new_lmt_price = round(lmt_price + 0.1, 2)
                    else:
                        new_lmt_price = round(lmt_price - 0.1, 2)
                    # Print the order details and new limit price
                    print(name, lmt_price, qty, new_lmt_price)
                    # Modify the order with the new limit price
                    data = {
                        "id": id1,
                        "type": 1,
                        "limitPrice": new_lmt_price,
                        "qty": qty
                    }
                    # Send the modify order request to Fyers
                    response = fyers.modify_order(data=data)
                    # Print the response from Fyers
                    print(response)
            except:
                # Print an error message if there is an exception
                print('error in chasing order')


pnl=0

def main_strategy_code():


    while True:
        ct = dt.now(time_zone)  # Get the current time
        # print(ct)

        # close program 2 min after end time
        if ct > end_time + dt.duration( minutes=2):
            logging.info('closing program')
            sys.exit()

        # Get current PnL and chase order every 5 seconds
        if ct.second in range(0, 59, 5):
            try:
                # Fetch order book information asynchronously
                order_response = fyers.orderbook()

                # Convert order book response to DataFrame if it exists
                if order_response['orderBook']:
                    order_df = pd.DataFrame(order_response['orderBook'])
                else:
                    order_df = pd.DataFrame()

                # Chase the order based on the order DataFrame
                chase_order(order_df)

                # Fetch positions asynchronously
                pos1 = fyers.positions()

                # Get the total PnL from the positions
                pnl = int(pos1.get('overall').get('pl_total'))

            except:
                # Print error message if unable to fetch PnL or chase order
                print('unable to fetch pnl or chase order')

            # Print the current PnL
            # print("current_pnl", pnl)

        # Refresh the Renko chart every 12 seconds
        if ct.second in range(1,60,12) :
            try:
                d=fetchOHLC(ticker,'1',6)
                data=d
            except:
                print('unable to fetch data')

            print(data)
            closing_price=data['close'].iloc[-1]
            renko_df=candle_renko_refresh(ticker,data)
            # renko_df.to_csv('data.csv')
            if account_type=='PAPER':
                paper_order(closing_price,renko_df)
            else:
                real_order(closing_price,renko_df)

        time.sleep(1)

main_strategy_code()