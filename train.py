from fastbook import *
import fastbook
import datetime as dt
import os
import warnings
import numpy as np
import pandas as pd
import pywt
import pywt.data
import re
import gc
import pandas_ta as ta
from pyts.image import GramianAngularField
from PIL import Image as im

fastbook.setup_book()

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
    

df = pd.read_csv("https://drive.google.com/file/d/1zzvusYoCLq1Kt7OLf8cLNB6rPDedjDVT/view?usp=sharing", header=0,
                 names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = df['timestamp'].apply(lambda x: pd.to_datetime(x))


def calc_label(dataset, position):
    price = dataset['close'].to_numpy()[position]
    stop_loss = STOP_LOSS * price
    stop_profit = STOP_PROFIT * price
    label = "wait"
    bearish_stop_loss = price + stop_loss
    bearish_stop_profit = price - stop_profit
    bullish_stop_loss = price - stop_loss
    bullish_stop_profit = price + stop_profit
    for i in range(1, ORDER_LIFE+1):
        max_price = dataset['high'].to_numpy()[position+i]
        low_price = dataset['low'].to_numpy()[position+i]
        if low_price > bullish_stop_loss:
            if max_price >= bullish_stop_profit:
                label = "buy"
                break
        else:
            break
    for i in range(1, ORDER_LIFE+1):
        max_price = dataset['high'].to_numpy()[position+i]
        low_price = dataset['low'].to_numpy()[position+i]
        if max_price < bearish_stop_loss:
            if low_price <= bearish_stop_profit:
                label = "sell"
                break
        else:
            break
    return label


L = len(df['close'])
df_labels = [None]*L
for i in range(0, L-ORDER_LIFE):
    df_labels[i] = calc_label(df, i)

df['label'] = df_labels

df = df.iloc[:-ORDER_LIFE].reset_index(drop=True).copy()

DS_LENGTH = len(df['close'])
input_open = [None]*DS_LENGTH
input_high = [None]*DS_LENGTH
input_low = [None]*DS_LENGTH
input_close = [None]*DS_LENGTH
input_volume = [None]*DS_LENGTH
input_timestamp = [None]*DS_LENGTH
for i in range(0, DS_LENGTH):
    if i + 1 >= RAW_INPUT_SIZE:
        input_open[i] = df.iloc[i+1-RAW_INPUT_SIZE:i+1].open.copy()
        input_high[i] = df.iloc[i+1-RAW_INPUT_SIZE:i+1].high.copy()
        input_low[i] = df.iloc[i+1-RAW_INPUT_SIZE:i+1].low.copy()
        input_close[i] = df.iloc[i+1-RAW_INPUT_SIZE:i+1].close.copy()
        input_volume[i] = df.iloc[i+1-RAW_INPUT_SIZE:i+1].volume.copy()
        input_timestamp[i] = df.iloc[i+1-RAW_INPUT_SIZE:i +
                                     1].timestamp.copy().reset_index(drop=True)
df['input_open'] = input_open
df['input_high'] = input_high
df['input_low'] = input_low
df['input_close'] = input_close
df['input_volume'] = input_volume
df['input_timestamp'] = input_timestamp
df = df.iloc[RAW_INPUT_SIZE:].copy().reset_index(drop=True)

df = df[df['label'] != 'wait'].reset_index(drop=True).copy()
df.drop(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df = df.copy()
gc.collect()

DS_LENGTH = len(df['input_close'])
denoised_input_open = [None]*DS_LENGTH
denoised_input_high = [None]*DS_LENGTH
denoised_input_low = [None]*DS_LENGTH
denoised_input_close = [None]*DS_LENGTH
denoised_input_volume = [None]*DS_LENGTH

for i in range(0, DS_LENGTH):
    denoised_input_open[i] = denoise(df['input_open'][i])
    denoised_input_high[i] = denoise(df['input_high'][i])
    denoised_input_low[i] = denoise(df['input_low'][i])
    denoised_input_close[i] = denoise(df['input_close'][i])
    denoised_input_volume[i] = denoise(df['input_volume'][i])

df['denoised_input_open'] = denoised_input_open
df['denoised_input_high'] = denoised_input_high
df['denoised_input_low'] = denoised_input_low
df['denoised_input_close'] = denoised_input_close
df['denoised_input_volume'] = denoised_input_volume
df.drop(columns=['input_open', 'input_high',
        'input_low', 'input_close', 'input_volume'])
df = df.copy()
gc.collect()

ind_list = ['qstick', 't3', 'cti', 'mad', 'ha', 'squeeze', 'aroon',
            'bbands', 'kc', 'vwap', 'stoch']
ind_columns = ['qstick', 't3', 'cti', 'mad', 'HA_low', 'SQZ_20_2.0_20_1.5',
               'AROONU_14', 'BBU_5_2.0', 'KCBe_20_2', 'vwap', 'STOCHd_14_3_3']

for indi in ind_list:
    indi_result = {}
    new_cols = []
    for i in range(0, DS_LENGTH):
        indi_input = pd.DataFrame()
        indi_input['open'] = df['denoised_input_open'][i].copy()
        indi_input['high'] = df['denoised_input_high'][i].copy()
        indi_input['low'] = df['denoised_input_low'][i].copy()
        indi_input['close'] = df['denoised_input_close'][i].copy()
        indi_input['volume'] = df['denoised_input_volume'][i].copy()
        indi_input['timestamp'] = df['input_timestamp'][i].copy(
        ).reset_index(drop=True)
        indi_input.set_index(pd.DatetimeIndex(
            indi_input["timestamp"]), inplace=True)
        indi_fn = getattr(indi_input.ta, indi)
        data = indi_fn()
        if len(new_cols) == 0:
            if not isinstance(data, pd.Series):
                new_cols = new_cols + data.columns.to_numpy().tolist()
            else:
                new_cols = new_cols + [indi]
            for col_name in new_cols:
                indi_result[col_name] = [None]*DS_LENGTH
        for col_name in new_cols:
            if not isinstance(data, pd.Series):
                indi_result[col_name][i] = data[col_name]
            else:
                indi_result[col_name][i] = data
    for col_name in new_cols:
        if col_name in ind_columns:
            df[col_name] = indi_result[col_name]

gaf_transformer = GramianAngularField(
    method='difference', image_size=INPUT_SIZE)
df_gaf_input_open = [None]*DS_LENGTH
df_gaf_input_high = [None]*DS_LENGTH
df_gaf_input_low = [None]*DS_LENGTH
df_gaf_input_close = [None]*DS_LENGTH
df_gaf_input_volume = [None]*DS_LENGTH
for i in range(0, DS_LENGTH):
    if len(df['denoised_input_close'][i]) > 0:
        df_gaf_input_open[i] = gaf_transformer.fit_transform(
            df['denoised_input_open'][i][-INPUT_SIZE:].reshape(1, -1))
        df_gaf_input_high[i] = gaf_transformer.fit_transform(
            df['denoised_input_high'][i][-INPUT_SIZE:].reshape(1, -1))
        df_gaf_input_low[i] = gaf_transformer.fit_transform(
            df['denoised_input_low'][i][-INPUT_SIZE:].reshape(1, -1))
        df_gaf_input_close[i] = gaf_transformer.fit_transform(
            df['denoised_input_close'][i][-INPUT_SIZE:].reshape(1, -1))
        df_gaf_input_volume[i] = gaf_transformer.fit_transform(
            df['denoised_input_volume'][i][-INPUT_SIZE:].reshape(1, -1))
df['gaf_open'] = df_gaf_input_open
df['gaf_high'] = df_gaf_input_high
df['gaf_low'] = df_gaf_input_low
df['gaf_close'] = df_gaf_input_close
df['gaf_volume'] = df_gaf_input_volume
df.drop(columns=['denoised_input_open', 'denoised_input_high', 'denoised_input_low',
        'denoised_input_close', 'denoised_input_volume', 'input_timestamp'])
df = df.copy()
gc.collect()

for col_name in ind_columns:
    print(col_name)
    gaf_col = [None]*DS_LENGTH
    for i in range(0, DS_LENGTH):
        if len(df['denoised_input_close'][i]) > 0:
            if isinstance(df[col_name][i], pd.Series):
                gaf_col[i] = gaf_transformer.fit_transform(
                    df[col_name][i][-INPUT_SIZE:].to_numpy().reshape(1, -1))
            else:
                gaf_col[i] = gaf_transformer.fit_transform(
                    df[col_name][i][-INPUT_SIZE:].reshape(1, -1))
    df[col_name] = gaf_col

df_train = df
pat = r'^(.*)_\d+.png'
images_path = '/kaggle/working/images/'
if not os.path.exists(images_path):
    os.makedirs(images_path)
files = get_image_files(images_path)
for f in files:
    os.remove(f)
files = get_image_files(images_path)
L = len(df_train['gaf_open'])
for i in range(0, L):
    i_open = df_train['gaf_open'].to_numpy()[i].squeeze()
    i_high = df_train['gaf_high'].to_numpy()[i].squeeze()
    i_low = df_train['gaf_low'].to_numpy()[i].squeeze()
    i_close = df_train['gaf_close'].to_numpy()[i].squeeze()
    i_volume = df_train['gaf_volume'].to_numpy()[i].squeeze()
    inputs_list = [i_open, i_high, i_low, i_close, i_volume] + \
        [df_train[col_name].to_numpy()[i].squeeze()
         for col_name in ind_columns]
    rows_list = [inputs_list[i:i + 4] for i in range(0, len(inputs_list), 4)]
    image_rows = [np.concatenate(row) for row in rows_list]
    image = np.concatenate(image_rows, axis=1)
    label = df_train['label'].to_numpy()[i]
    matplotlib.image.imsave(images_path + label + '_' + str(i) + '.png', image)
files = get_image_files(images_path)
dls = ImageDataLoaders.from_name_re(images_path, files, pat)
#dls.show_batch()
learn = vision_learner(dls, resnet34, metrics=error_rate)
