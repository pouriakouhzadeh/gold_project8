#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import ta
import warnings
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import logging
import multiprocessing
from functools import partial
import gc

from clear_data import ClearData
from custom_indicators import (
    KSTCustomIndicator,
    VortexCustomIndicator,
    CustomIchimokuIndicator,
    CustomWilliamsRIndicator,
    CustomVolumeRateOfChangeIndicator,
    CustomPivotPointIndicator,
    CustomCandlestickPattern
)
from numba import config as numba_config

# numba تنظیم لاگ‌های 
logging.getLogger('numba').setLevel(logging.CRITICAL)
numba_config.LOG_LEVEL = 'CRITICAL'

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.DEBUG,
    filename='prepare_data.log',
    format='%(asctime)s %(levelname)s:%(message)s'
)

from numba_utils import (
    numba_skew,
    numba_kurtosis,
    numba_median,
    numba_up_count,
    numba_last_local_max_idx,
    numba_last_local_min_idx
)

# برای sliding_window_view
from numpy.lib.stride_tricks import sliding_window_view

class PREPARE_DATA_FOR_TRAIN:
    def __init__(self, filepaths: dict = None, main_timeframe: str = '30T'):
        """
        اگر filepaths ارائه نشود، پیشفرض:
            {
              '30T': 'XAUUSD_M30.csv',
              '1H': 'XAUUSD_H1.csv',
              '15T': 'XAUUSD_M15.csv',
              '5T': 'XAUUSD_M5.csv'
            }
        """
        if filepaths is None:
            filepaths = {
                '30T': 'XAUUSD_M30.csv',
                '1H': 'XAUUSD_H1.csv',
                '15T': 'XAUUSD_M15.csv',
                '5T': 'XAUUSD_M5.csv'
            }
        self.filepaths = filepaths
        self.main_timeframe = main_timeframe
        logging.info(f"[INIT] filepaths={self.filepaths}, main_timeframe={self.main_timeframe}")
        self.train_columns_after_window = []
        self.train_raw_window = None

    def _convert_timedelta_to_seconds(self, df: pd.DataFrame):
        for col in df.columns:
            if pd.api.types.is_timedelta64_dtype(df[col]):
                df[col] = df[col].dt.total_seconds()

    def load_and_process_timeframe(self, timeframe_label: str, filepath: str) -> pd.DataFrame:
        """
        همان کدی که اندیکاتورها را اضافه می‌کند و داده را Resample می‌کند.
        """
        try:
            data = ClearData().clean(pd.read_csv(filepath))
            print(f"Data for timeframe {timeframe_label} loaded & clean successfully.")
            logging.info(f"[load_and_process_timeframe] {timeframe_label}: file loaded => shape={data.shape}")
        except Exception as e:
            logging.error(f"[load_and_process_timeframe] Error loading {timeframe_label}: {e}")
            raise

        if 'time' not in data.columns:
            raise ValueError(f"'time' column not found in {filepath}.")

        data['time'] = pd.to_datetime(data['time'])
        data.sort_values('time', inplace=True)
        data.set_index('time', drop=True, inplace=True)

        if timeframe_label != self.main_timeframe:
            logging.debug(f"[{timeframe_label}] Resampling to main timeframe={self.main_timeframe}")
            data = data.resample(self.main_timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.dropna(axis=0, how='all', inplace=True)

        data.reset_index(inplace=True)
        data.rename(columns={'time': f'{timeframe_label}_time'}, inplace=True)

        prefix = f"{timeframe_label}_"
        if f'{timeframe_label}_time' in data.columns:
            data[f'{prefix}Hour'] = data[f'{timeframe_label}_time'].dt.hour
            data[f'{prefix}DayOfWeek'] = data[f'{timeframe_label}_time'].dt.dayofweek
            data[f'{prefix}IsWeekend'] = data[f'{prefix}DayOfWeek'].isin([5, 6]).astype(int)

        # Rolling ساده
        data[f'{prefix}ma20'] = data['close'].rolling(20).mean()
        data[f'{prefix}ma50'] = data['close'].rolling(50).mean()
        data[f'{prefix}ma_volume20'] = data['volume'].rolling(20).mean()
        data[f'{prefix}ReturnDifference'] = data['close'].diff()
        data[f'{prefix}ROC'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-9) * 100

        data[f'{prefix}Rolling_Mean_20'] = data['close'].rolling(20).mean()
        data[f'{prefix}Rolling_Std_20'] = data['close'].rolling(20).std()
        data[f'{prefix}Rolling_Skew_20'] = data['close'].rolling(20).apply(numba_skew, raw=True)
        data[f'{prefix}Rolling_Kurt_20'] = data['close'].rolling(20).apply(numba_kurtosis, raw=True)
        data[f'{prefix}Rolling_Median_20'] = data['close'].rolling(20).apply(numba_median, raw=True)
        data[f'{prefix}Rolling_UpCount_20'] = data['close'].rolling(20).apply(numba_up_count, raw=True)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.dropna(axis=0, how='all', inplace=True)

        # ta اندیکاتورها
        indicators = ta.add_all_ta_features(
            data.copy(),
            open="open", high="high", low="low", close="close", volume="volume",
            fillna=False
        )
        indicators = indicators.add_prefix(prefix)
        # حذف ایچیموکو
        ichimoku_cols = [c for c in indicators.columns if 'ichimoku' in c.lower()]
        if ichimoku_cols:
            indicators.drop(columns=ichimoku_cols, inplace=True, errors='ignore')
            logging.debug(f"[{timeframe_label}] Dropped Ichimoku columns from ta")
        data = pd.concat([data, indicators], axis=1)
        data = data.loc[:, ~data.columns.duplicated()]

        from ta.trend import PSARIndicator
        data[f'{prefix}Parabolic_SAR'] = PSARIndicator(
            high=data['high'], low=data['low'], close=data['close'],
            step=0.018, max_step=0.2
        ).psar()

        data[f'{prefix}Momentum_14'] = data['close'] - data['close'].shift(14)
        data[f'{prefix}Trix_15'] = ta.trend.TRIXIndicator(data['close'], window=15).trix()
        data[f'{prefix}Ultimate_Osc'] = ta.momentum.UltimateOscillator(
            data['high'], data['low'], data['close'], 7, 14, 28
        ).ultimate_oscillator()
        data[f'{prefix}Daily_Range'] = data['high'] - data['low']

        logret = np.log(data['close'] / (data['close'].shift(1) + 1e-9))
        data[f'{prefix}HV_20'] = logret.rolling(20).std() * np.sqrt(24 * 365)

        try:
            hl_log = np.log(data['high'] / (data['low'] + 1e-9) + 1e-9)
            co_log = np.log(data['close'] / (data['open'] + 1e-9) + 1e-9)
            data[f'{prefix}GarmanKlass'] = (0.5 * (hl_log**2) - (2 * (co_log**2))**0.5).rolling(20).mean()
        except:
            data[f'{prefix}GarmanKlass'] = np.nan

        try:
            ln_hl = np.log(data['high'] / (data['low'] + 1e-9) + 1e-9)
            data[f'{prefix}Parkinson_20'] = (ln_hl**2).rolling(20).sum()
            data[f'{prefix}Parkinson_20'] = np.sqrt(data[f'{prefix}Parkinson_20'] * (1 / (4 * np.log(2) * 20)))
        except:
            data[f'{prefix}Parkinson_20'] = np.nan

        data[f'{prefix}UlcerIndex_14'] = ta.volatility.UlcerIndex(data['close'], 14).ulcer_index()
        data[f'{prefix}MFI'] = ta.volume.MFIIndicator(
            data['high'], data['low'], data['close'], data['volume'], 14, fillna=False
        ).money_flow_index()
        data[f'{prefix}EoM_14'] = ta.volume.EaseOfMovementIndicator(
            data['high'], data['low'], data['volume'], 14
        ).sma_ease_of_movement()

        period_dpo = 20
        data[f'{prefix}DPO_20'] = data['close'] - data['close'].rolling(period_dpo // 2 + 1).mean()

        data[f'{prefix}MACD'] = ta.trend.MACD(data['close']).macd()
        data[f'{prefix}RSI_14'] = ta.momentum.RSIIndicator(data['close'], 14).rsi()

        bb = ta.volatility.BollingerBands(data['close'], 20, 2)
        data[f'{prefix}Bollinger_High'] = bb.bollinger_hband()
        data[f'{prefix}Bollinger_Low'] = bb.bollinger_lband()
        data[f'{prefix}Bollinger_Width'] = bb.bollinger_wband()

        data[f'{prefix}SMA10'] = ta.trend.SMAIndicator(data['close'], 10).sma_indicator()
        data[f'{prefix}SMA50'] = ta.trend.SMAIndicator(data['close'], 50).sma_indicator()
        data[f'{prefix}SMA10_SMA50_diff'] = data[f'{prefix}SMA10'] - data[f'{prefix}SMA50']

        data[f'{prefix}EMA10'] = ta.trend.EMAIndicator(data['close'], 10).ema_indicator()
        data[f'{prefix}EMA50'] = ta.trend.EMAIndicator(data['close'], 50).ema_indicator()
        data[f'{prefix}EMA10_50_diff'] = data[f'{prefix}EMA10'] - data[f'{prefix}EMA50']

        data[f'{prefix}ATR_14'] = ta.volatility.AverageTrueRange(
            data['high'], data['low'], data['close'], 14
        ).average_true_range()
        data[f'{prefix}Bollinger_Width_Ratio'] = data[f'{prefix}Bollinger_Width'] / (data[f'{prefix}Rolling_Std_20'] + 1e-9)

        kst = KSTCustomIndicator(data['close'], 10, 15, 20, 30, 10, 10, 10, 15, 9, fillna=True)
        data[f'{prefix}KST_MAIN'] = kst.kst()
        data[f'{prefix}KST_SIGNAL'] = kst.kst_signal()
        data[f'{prefix}KST_DIFF'] = kst.kst_diff()

        vtx = VortexCustomIndicator(data['high'], data['low'], data['close'], 14, fillna=True)
        data[f'{prefix}Vortex_Pos'] = vtx.vortex_pos()
        data[f'{prefix}Vortex_Neg'] = vtx.vortex_neg()

        ichi = CustomIchimokuIndicator(data['high'], data['low'], data['close'], 9, 26, 52)
        data[f'{prefix}Ichimoku_Conversion_Line'] = ichi.ichimoku_conversion_line()
        data[f'{prefix}Ichimoku_Base_Line'] = ichi.ichimoku_base_line()
        data[f'{prefix}Ichimoku_A'] = (ichi.ichimoku_conversion_line() + ichi.ichimoku_base_line()) / 2
        data[f'{prefix}Ichimoku_B'] = (data['high'].rolling(52).max() + data['low'].rolling(52).min()) / 2

        will = CustomWilliamsRIndicator(data['high'], data['low'], data['close'], 14)
        data[f'{prefix}Williams_%R'] = will.williams_r()

        vroc = CustomVolumeRateOfChangeIndicator(data['volume'], 20)
        data[f'{prefix}VROC'] = vroc.volume_rate_of_change()

        piv = CustomPivotPointIndicator(data['high'], data['low'], data['close'], 5)
        data[f'{prefix}Pivot'] = piv.pivot()
        data[f'{prefix}Support_1'] = piv.support_1()
        data[f'{prefix}Support_2'] = piv.support_2()
        data[f'{prefix}Resistance_1'] = piv.resistance_1()
        data[f'{prefix}Resistance_2'] = piv.resistance_2()

        candle = CustomCandlestickPattern(data['open'], data['high'], data['low'], data['close'])
        data[f'{prefix}Engulfing'] = candle.engulfing()
        data[f'{prefix}Doji'] = candle.doji()

        ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        ha_open = ha_close.shift(1)
        ha_open.ffill(inplace=True)
        data[f'{prefix}HeikinAshi_Open'] = ha_open
        data[f'{prefix}HeikinAshi_Close'] = ha_close

        data[f'{prefix}Range_Close_Ratio'] = (data['high'] - data['low']) / (data['close'] + 1e-9)
        data[f'{prefix}Bull_Power'] = (data['close'] - data['low']) / ((data['high'] - data['low']) + 1e-9)

        data[f'{prefix}BarsFromLocalMax_20'] = data['close'].rolling(20).apply(numba_last_local_max_idx, raw=True)
        data[f'{prefix}BarsFromLocalMin_20'] = data['close'].rolling(20).apply(numba_last_local_min_idx, raw=True)

        data[f'{prefix}RSI_MACD'] = data[f'{prefix}RSI_14'] * data[f'{prefix}MACD']
        data[f'{prefix}MA20_MA50_Ratio'] = data[f'{prefix}ma20'] / (data[f'{prefix}ma50'] + 1e-9)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.dropna(axis=0, how='all', inplace=True)

        self._convert_timedelta_to_seconds(data)
        return data

    def select_features(self, X: pd.DataFrame, y: pd.Series, top_k=150):
        """
        کاهش پیش‌فرض به 150 تا مصرف رم کمتر شود
        """
        logging.info(f"[select_features] Start VarThreshold => X.shape={X.shape}, y.shape={y.shape}")
        vt = VarianceThreshold(0.01)
        vt.fit(X)
        X_var = X[X.columns[vt.get_support()]]
        print(f"Features after Variance Threshold: {len(X_var.columns)}")

        corrm = X_var.corr().abs()
        upper = corrm.where(np.triu(np.ones(corrm.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
        X_corr = X_var.drop(columns=to_drop, errors='ignore')
        print(f"Features after Correlation Threshold: {len(X_corr.columns)}")

        if len(X_corr.columns) == 0:
            print("No column left after correlation threshold => returning empty list.")
            return []

        if top_k > len(X_corr.columns):
            top_k = len(X_corr.columns)

        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X_corr)
        mi = mutual_info_classif(Xs, y, discrete_features='auto')
        mi_series = pd.Series(mi, index=X_corr.columns).sort_values(ascending=False)
        top_feats = mi_series.head(top_k).index.tolist()
        print(f"Features after Mutual Info: {len(top_feats)}")
        return top_feats

    def ready(self, data: pd.DataFrame, window: int = 1, selected_features: list = None, mode: str = 'train'):
        logging.info(f"Data ready is begining ... mode={mode}")
        main_close = f"{self.main_timeframe}_close"
        if main_close not in data.columns:
            raise ValueError(f"{main_close} not found in data columns.")

        if mode == 'train':
            target_close = data[main_close].copy()
            target = ((target_close.shift(-1) - target_close) > 0).astype(int)
            data = data.iloc[:-1].copy()
            target = target.iloc[:-1].copy()
        else:
            target = pd.Series([0] * len(data), index=data.index)

        time_cols = [c for c in data.columns if 'Hour' in c or 'DayOfWeek' in c or 'IsWeekend' in c]
        feat_cols = [c for c in data.columns if c not in time_cols + [main_close]]

        # Diff
        df_diff = data[feat_cols].diff()
        df_diff.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_diff.ffill(inplace=True)
        df_diff.bfill(inplace=True)
        df_diff.dropna(axis=0, how='all', inplace=True)
        df_diff.reset_index(drop=True, inplace=True)
        target.reset_index(drop=True, inplace=True)

        min_len = min(len(df_diff), len(target))
        df_diff = df_diff.iloc[:min_len].copy()
        target = target.iloc[:min_len].copy()

        self._convert_timedelta_to_seconds(df_diff)

        # انتخاب فیچر
        if selected_features is None:
            feats = self.select_features(df_diff, target, top_k=150)  # top_k =150
        else:
            feats = [f for f in selected_features if f in df_diff.columns]

        if len(feats) == 0:
            print("No features => empty.")
            logging.warning("[ready] X_filtered empty => returning empty")
            return pd.DataFrame(), pd.Series(dtype=int), feats

        X_filtered = df_diff[feats].copy()

        if window < 1:
            window = 1

        if window == 1:
            X_final = X_filtered
            y_final = target
        else:
            length = len(X_filtered)
            if length < window:
                print("Not enough rows for window.")
                logging.warning(f"[ready] length({length})<window({window}) => returning empty")
                return pd.DataFrame(), pd.Series(dtype=int), feats

            # --- استفاده از sliding_window_view برای کاهش مصرف RAM ---
            arr = X_filtered.to_numpy()
            # shape => (length - window + 1, window, num_features)
            result_3d = sliding_window_view(arr, (window, arr.shape[1]))
            # ولی sliding_window_view شکل (N-(w-1), 1, w, ncol) می‌دهد
            # بنابراین اصلاح می‌کنیم:
            # Actually, sliding_window_view(arr, window_shape=(window, arr.shape[1])) -> shape => (length-window+1, 1, window, arr.shape[1])
            # پس باید reshape مناسب انجام دهیم
            # simplest: sliding_window_view(arr, window_shape=(window, arr.shape[1]))[..., 0, :, :]
            # ولی numpy>=1.20
            # اینجا مطابق doc: new_shape => (N-window+1, window, n_features)
            # reshape => (N-window+1, window*n_features)
            # indexing:
            if result_3d.ndim == 4:
                # shape => (N-window+1, 1, window, n_features)
                result_3d = result_3d[:, 0, :, :]
            # حالا shape => (N-window+1, window, n_features)
            n_samp = result_3d.shape[0]
            n_feat = arr.shape[1]

            X_win_array = result_3d.reshape(n_samp, window * n_feat)

            # ساخت DataFrame
            X_win = pd.DataFrame(X_win_array)

            # نام‌گذاری ستون‌ها
            if X_win.shape[1] > 0:
                colnames = []
                for off in range(window):
                    stp = f"_tminus{off}"
                    for c in X_filtered.columns:
                        colnames.append(f"{c}{stp}")
                if len(colnames) == X_win.shape[1]:
                    X_win.columns = colnames

            # اندیس‌ها
            # تعداد ردیف X_win = length-window+1
            new_index = range(window-1, length)
            X_win.index = new_index

            # target
            y_win = target.loc[X_win.index].copy()

            X_win.reset_index(drop=True, inplace=True)
            y_win.reset_index(drop=True, inplace=True)

            X_win.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_win.ffill(inplace=True)
            X_win.bfill(inplace=True)
            X_win.dropna(axis=0, how='all', inplace=True)

            y_win = y_win.loc[X_win.index].copy()
            y_win.reset_index(drop=True, inplace=True)

            self._convert_timedelta_to_seconds(X_win)
            X_final = X_win
            y_final = y_win

        X_final.reset_index(drop=True, inplace=True)
        y_final.reset_index(drop=True, inplace=True)

        logging.info(f"Data ready finish => X_final.shape={X_final.shape}, y_final.shape={y_final.shape}")

        if mode == 'train':
            self.train_columns_after_window = X_final.columns.tolist()
            logging.info(f"[ready] train_columns_after_window is set with {len(self.train_columns_after_window)} columns.")

        return X_final, y_final, feats

    def ready_incremental(self, data_window: pd.DataFrame, window: int = 1, selected_features: list = None):
        """
        همان کد قبلی برای گرفتن فقط آخرین پنجره (1 سطر)
        """
        if window == 1:
            X_new, _, feats = self.ready(data_window, window=1, selected_features=selected_features, mode='predict')
            return X_new, feats
        else:
            X_temp, _, feats = self.ready(data_window, window=window, selected_features=selected_features, mode='predict')
            if X_temp.empty:
                return X_temp, feats
            else:
                # فقط آخرین ردیف پنجره را لازم داریم
                return X_temp.tail(1).copy(), feats

    def get_prepared_data(self, window: int = 1, mode: str = 'train'):
        logging.info(f"[get_prepared_data] Called with window={window}, mode={mode}")
        merged_df = self.load_data()
        X_fin, y_fin, feats = self.ready(merged_df, window=window, mode=mode)
        logging.info(f"[get_prepared_data] Done => X_fin.shape={X_fin.shape}, y_fin.shape={y_fin.shape}, feats={len(feats)}")
        return X_fin, y_fin, feats

    def load_data(self) -> pd.DataFrame:
        logging.info("[load_data] Start loading all timeframes.")
        n_processes = multiprocessing.cpu_count()
        main_df = self.load_and_process_timeframe(self.main_timeframe, self.filepaths[self.main_timeframe])
        main_df.set_index(f"{self.main_timeframe}_time", inplace=True, drop=False)

        tasks = [(tf_label, fp) for tf_label, fp in self.filepaths.items() if tf_label != self.main_timeframe]
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = pool.starmap(self.load_and_process_timeframe, tasks)

        for (tf_label, _), df in zip(tasks, results):
            df.set_index(f"{tf_label}_time", inplace=True, drop=False)
            main_df = main_df.join(df, how='outer', rsuffix=f"_{tf_label}")

        main_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        main_df.ffill(inplace=True)
        main_df.bfill(inplace=True)
        main_df.dropna(axis=0, how='all', inplace=True)
        main_df = main_df[~main_df.index.duplicated(keep='first')]
        self._convert_timedelta_to_seconds(main_df)

        print("All timeframes loaded and merged successfully.")
        logging.info(f"[load_data] All timeframes loaded and merged => shape={main_df.shape}")
        gc.collect()
        return main_df
