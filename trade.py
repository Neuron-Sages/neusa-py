from binance.spot import Spot
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from binance.lib.utils import config_logging
from binance.error import ClientError
from pyts.image import GramianAngularField
import logging
import json
import os
import pywt
import pywt.data
import schedule
from datetime import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
from configparser import ConfigParser
from fastai.vision.all import *

SYMBOL = 'BTCUSDT'
INTERVAL = '1m'
INPUT_SIZE = 30
RAW_INPUT_SIZE = 100
THRESHOLD = 0.9
TRADING_PERCENT = 0.1
INITIAL_USD_BALANCE = 1000
STOP_PROFIT = 0.004
STOP_LOSS = 0.004
ORDER_LIFE = 5

#config_logging(logging, logging.DEBUG)
config_file = "config.ini"
#firebase_app = firebase_admin.initialize_app()

wavelet_type = 'sym15'
w = pywt.Wavelet(wavelet_type)
def denoise(data):
    if len(data) > 0:
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        coeffs = pywt.wavedec(data, wavelet_type, level=maxlev)
        coeffs[-1] = np.zeros_like(coeffs[-1])
        datarec = pywt.waverec(coeffs, wavelet_type)
        return datarec
    else:
        return data
    

ind_list = ['qstick', 't3', 'cti', 'mad', 'ha',
            'squeeze', 'aroon', 'bbands', 'kc', 'vwap', 'stoch']
ind_columns = ['qstick', 't3', 'cti', 'mad', 'HA_low', 'SQZ_20_2.0_20_1.5',
               'AROONU_14', 'BBU_5_2.0', 'KCBe_20_2', 'vwap', 'STOCHd_14_3_3']
    
def get_input_image(data):
    np_data = np.array(data)
    df = pd.DataFrame(data=np_data[:, 0:6],
                      index=np_data[:, 0],
                      columns=np_data[0, 0:6])
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['open'] = denoise(pd.to_numeric(df['open']))
    df['high'] = denoise(pd.to_numeric(df['high']))
    df['low'] = denoise(pd.to_numeric(df['low']))
    df['close'] = denoise(pd.to_numeric(df['close']))
    df['volume'] = denoise(pd.to_numeric(df['volume']))
    df['timestamp'].apply(lambda x: pd.to_datetime(x))
    df.set_index(pd.DatetimeIndex(
        df["timestamp"]), inplace=True)
    for indi in ind_list:
        indi_fn = getattr(df.ta, indi)
        data = indi_fn()
        if isinstance(data, pd.Series):
            df[indi] = data
        else:
            for col_name in data.columns.to_numpy().tolist():
                df[col_name] = data[col_name]

    gaf_transformer = GramianAngularField(
        method='difference', image_size=INPUT_SIZE)
    all_input_cols = ['open', 'high', 'low', 'close', 'volume'] + ind_columns
    for col_name in all_input_cols:
        df['gaf_' + col_name] = gaf_transformer.fit_transform(
            df[col_name][-INPUT_SIZE:].reshape(1, -1)).squeeze()
    inputs_list = [df['gaf_' + col_name] for col_name in all_input_cols]
    rows_list = [inputs_list[i:i + 4] for i in range(0, len(inputs_list), 4)]
    image_rows = [np.concatenate(row) for row in rows_list]
    image = np.concatenate(image_rows, axis=1)
    images_path = './images/'
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    matplotlib.image.imsave(images_path + 'input' + '.png', image)
    files = get_image_files(images_path)
    return files[0]

def get_params():
    config = ConfigParser()
    config_file_path = os.path.join(
        os.path.abspath(''), config_file
    )
    config.read(config_file_path)
    return config["keys"]["api_key"], config["keys"]["api_secret"], config["keys"]["url"], config["keys"]["ws_url"]


api_key, api_secret, url, ws_url = get_params()

model = load_learner(fname='model.pkl', cpu=True)

def predict(input):
    prediction, t_index, t_confidence_array = model.predict(get_input_image(input))
    index = t_index.detach().cpu().numpy()
    confidence_array = t_confidence_array.detach().cpu().numpy()
    confidence = confidence_array[index]
    return prediction, confidence

def get_stops(prediction, price):
    stop_loss_shift = int(price / 100.0 * STOP_LOSS)
    stop_profit_shift = int(price / 100.0 * STOP_PROFIT)
    if prediction == 'buy':
        return price + stop_profit_shift, price - stop_loss_shift
    else:
        return price - stop_profit_shift, price + stop_loss_shift
    
def get_new_order_params(prediction, price):
    stop_profit, stop_loss = get_stops(prediction, price)
    quantity = round(TRADING_PERCENT * INITIAL_USD_BALANCE / price, 4)
    return {
        'binance_order': {
            'symbol': SYMBOL,
            'side': prediction.upper(),
            'type': 'MARKET',
            'quantity': quantity
        },
        'metadata': {
            'stop_loss': stop_loss,
            'stop_profit': stop_profit,
            'price': price,
            'prediction': prediction
        }
    }


start_time_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
total_profit_orders_count = 0
total_loss_orders_count = 0
opened_order = None

def run_algorithm():
    global opened_order
    global total_loss_orders_count
    global total_profit_orders_count
    global start_time_string
    input = rest_client.klines(SYMBOL, INTERVAL, limit=RAW_INPUT_SIZE)
    kline = input[-1]
    price = float(kline[4])
    prediction, confidence = predict(input)
    if confidence >= THRESHOLD and prediction != 'wait':
        new_order_params = get_new_order_params(prediction, price)
        now = datetime.now()
        now_string = now.strftime("%d/%m/%Y %H:%M:%S")
        if opened_order == None:
            opened_order = new_order_params
            print("I have opened an order now:",
                opened_order['metadata']['prediction'])
        else:
            old_metadata = opened_order['metadata']
            old_side = old_metadata['prediction']
            old_price = old_metadata['price']
            new_metadata = new_order_params['metadata']
            new_side = new_metadata['prediction']
            new_price = new_metadata['price']
            if old_side != new_side and new_side != 'wait':
                side = old_side
                price = new_price
                if side == 'buy':
                    if price > old_price:
                        print('Closing buy order with profit')
                        total_profit_orders_count = 1 + total_profit_orders_count
                        opened_order = None
                    if price <= old_price:
                        print('Closing buy order with loss')
                        total_loss_orders_count = 1 + total_loss_orders_count
                        opened_order = None
                if side == 'sell':
                    if price < old_price:
                        print('Closing sell order with profit')
                        total_profit_orders_count = 1 + total_profit_orders_count
                        opened_order = None
                    if price >= old_price:
                        print('Closing sell order with loss')
                        total_loss_orders_count = 1 + total_loss_orders_count
                        opened_order = None
                opened_order = new_order_params
                with open('log.txt', 'w') as f:
                    print('From ', start_time_string, 'to', now_string, 'total_profit_orders_count=', total_profit_orders_count, 'total_loss_orders_count=', total_loss_orders_count, file=f)
        #response = rest_client.new_order(**new_order_params['binance_order'])
        #print(response)
        #status = response['status']
        #if status == 'FILLED':
        #    opened_order = new_order_params

rest_client = Spot(api_key=api_key, api_secret=api_secret, base_url=url)
schedule.every().minute.do(run_algorithm)
while True:
    schedule.run_pending()
    time.sleep(1)

#python3 trade.py
