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
#import firebase_admin
from configparser import ConfigParser
from fastai.vision.all import *

SYMBOL = 'BTCUSDT'
INTERVAL = '1m'
INPUT_SIZE = 30
RAW_INPUT_SIZE = 100
THRESHOLD = 0.97
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
    
def get_input_image(data):
    np_data = np.array(data)
    df = pd.DataFrame(data=np_data[:, 1:6],
                      index=np_data[:, 0],
                      columns=np_data[0, 1:6])
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    df_denoised_input = denoise(df.close)
    gaf_transformer = GramianAngularField(
        method='difference', image_size=INPUT_SIZE)
    df_gaf_input = gaf_transformer.fit_transform(
        df_denoised_input[-INPUT_SIZE:].reshape(1, -1))
    images_path = './images/'
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    data = df_gaf_input.squeeze()
    matplotlib.image.imsave(images_path + 'input' + '.png', data)
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
    print(prediction, confidence)
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

opened_order = {}

def run_algorithm():
    input = rest_client.klines(SYMBOL, INTERVAL, limit=RAW_INPUT_SIZE)
    kline = input[-1]
    price = float(kline[4])
    prediction, confidence = predict(input)
    if confidence >= THRESHOLD and prediction != 'wait':
        new_order_params = get_new_order_params(prediction, price)
        print('I would open a new order with params:', new_order_params)
        #response = rest_client.new_order(**new_order_params['binance_order'])
        #print(response)
        #status = response['status']
        #if status == 'FILLED':
        #    opened_order = new_order_params

def message_handler(_, message):
    parsed = json.loads(message)
    if 'e' in parsed:  # kline subscription message
        k = parsed['k']
        if k['x'] == True:  # kline is closed
            if opened_order == {}:
                run_algorithm()

rest_client = Spot(api_key=api_key, api_secret=api_secret, base_url=url)
ws_client = SpotWebsocketStreamClient(on_message=message_handler, stream_url=ws_url  # , on_close=on_close
                                      )
def get_balance():
    print(rest_client.account())

ws_client.kline(symbol=SYMBOL, interval=INTERVAL)
