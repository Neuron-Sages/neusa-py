from binance.spot import Spot
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from binance.lib.utils import config_logging
from binance.error import ClientError
from pyts.image import GramianAngularField
import logging
import json
import os
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
ORDER_LIFE = 15

config_file = "config.test.ini"
model = load_learner(fname='model3.pkl', cpu=True)

def get_params():
    config = ConfigParser()
    config_file_path = os.path.join(
        os.path.abspath(''), config_file
    )
    config.read(config_file_path)
    return config["keys"]["api_key"], config["keys"]["api_secret"], config["keys"]["url"]

api_key, api_secret, url = get_params()
rest_client = Spot(api_key=api_key, api_secret=api_secret, base_url=url)

ind_list = ['qstick', 't3', 'cti', 'mad', 'ha',
            'squeeze', 'aroon', 'bbands', 'kc', 'vwap', 'stoch']
ind_columns = ['QS_10', 'T3_10_0.7', 'CTI_12', 'MAD_30', 'HA_low', 'SQZ_20_2.0_20_1.5',
               'AROONU_14', 'BBU_5_2.0', 'KCBe_20_2', 'VWAP_D', 'STOCHd_14_3_3']
    
def get_input_image(data):
    np_data = np.array(data)
    df = pd.DataFrame(data=np_data[:, 0:6],
                      index=np_data[:, 0],
                      columns=np_data[0, 0:6])
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['timestamp'].apply(lambda x: pd.to_datetime(x, unit='ms'))
    df.set_index(pd.DatetimeIndex(
        df["timestamp"]), inplace=True)
    for indi in ind_list:
        indi_fn = getattr(df.ta, indi)
        indi_fn(append=True)

    for col_name in ind_columns:
        df[col_name].replace([np.inf, -np.inf], np.nan, inplace=True)
        df[col_name].fillna(value=0, inplace=True)

    gaf_transformer = GramianAngularField(
        method='difference', image_size=INPUT_SIZE)
    all_input_cols = ['open', 'high', 'low', 'close', 'volume'] + ind_columns
    inputs_list = [gaf_transformer.fit_transform(
        df[col_name][-INPUT_SIZE].reshape(1, -1)).squeeze() for col_name in all_input_cols]
    rows_list = [inputs_list[i:i + 4] for i in range(0, len(inputs_list), 4)]
    image_rows = [np.concatenate(row) for row in rows_list]
    image = np.concatenate(image_rows, axis=1)
    images_path = './images/'
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    matplotlib.image.imsave(images_path + 'input' + '.png', image)
    files = get_image_files(images_path)
    return files[0]

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
    
def change_side(side):
    if side == 'sell':
        return 'buy'
    else:
        return 'sell'
    
def get_new_order_params(prediction, price):
    stop_profit, stop_loss = get_stops(prediction, price)
    quantity = round(TRADING_PERCENT * INITIAL_USD_BALANCE / price, 4)
    best_params = rest_client.book_ticker(symbol=SYMBOL)
    logging.debug("best_params: %s", best_params)
    best_price = best_params['price']
    return  [{
            'symbol': SYMBOL,
            'side': prediction.upper(),
            'type': 'MARKET',
            'quantity': quantity,
            'price': best_price
        },
        {
            'symbol': SYMBOL,
            'side': change_side(prediction).upper(),
            'type': 'LIMIT',
            'quantity': quantity,
            'price': stop_profit
        }]

def run_algorithm():
    opened_orders = rest_client.get_open_orders(symbol=SYMBOL)
    logging.debug('opened_orders: %s', opened_orders)
    if len(opened_orders) == 0:
        input = rest_client.klines(SYMBOL, INTERVAL, limit=RAW_INPUT_SIZE)
        kline = input[-1]
        price = float(kline[4])
        prediction, confidence = predict(input)
        if confidence >= THRESHOLD and prediction != 'wait':
            new_order_params = get_new_order_params(prediction, price)
            response1 = rest_client.new_order(new_order_params[0])
            response2 = rest_client.new_order(new_order_params[1])
            logging.debug("response1: %s", response1)
            logging.debug("response2: %s", response2)

schedule.every().minute.do(run_algorithm)
while True:
    schedule.run_pending()
    time.sleep(1)

#python3 trade.py
