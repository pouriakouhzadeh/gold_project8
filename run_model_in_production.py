import time
import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.exceptions import NotFittedError

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from drift_checker import DriftChecker

logging.basicConfig(
    filename='production.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def remove_files_if_exists():
    xauusd_file = "XAUUSD30.acn"
    answer_file = "Answer.txt"
    try:
        if os.path.exists(xauusd_file):
            os.remove(xauusd_file)
        if os.path.exists(answer_file):
            os.remove(answer_file)
        print("Answer.txt && XAUUSD30.acn deleted")
    except:
        pass

class LivePredictor:
    def __init__(self):
        saved_data = joblib.load("best_model.pkl")

        self.pipeline = saved_data['pipeline']
        self.hyperparams = saved_data['hyperparams']
        self.feat_mask = saved_data['feat_mask']
        self.neg_thr = saved_data['neg_thr']
        self.pos_thr = saved_data['pos_thr']
        self.scaler = saved_data['scaler']
        self.window_size = saved_data['window_size']
        self.feats = saved_data.get('feats', None)
        self.train_window_cols = saved_data.get('train_window_cols', [])
        self.train_raw_window = saved_data.get('train_raw_window', None)

        if self.window_size == 1:
            self.train_raw_window = None
        else:
            if self.train_raw_window is None:
                logging.error("Missing train_raw_window in saved model. Cannot perform incremental prediction.")
                raise ValueError("Missing train_raw_window in model file.")

        logging.info(f"[LivePredictor] train_window_cols => {len(self.train_window_cols)} columns")

        self.filepaths = {
            '30T': 'XAUUSD.F_M30_live.csv',
            '1H':  'XAUUSD.F_H1_live.csv',
            '15T': 'XAUUSD.F_M15_live.csv',
            '5T':  'XAUUSD.F_M5_live.csv'
        }

        self.prep = PREPARE_DATA_FOR_TRAIN(filepaths=self.filepaths, main_timeframe='30T')
        self.answer_file = "Answer.txt"

        self.drift_checker = DriftChecker()
        self.drift_checker.load_train_distribution("train_distribution.json")

    def check_and_predict(self):
        for tf in self.filepaths:
            if not os.path.exists(self.filepaths[tf]):
                print(f"[Live] File {self.filepaths[tf]} not found => waiting...")
                return

        try:
            merged_df = self.prep.load_data()
            psi_val = self.drift_checker.compare_live(merged_df)
            logging.info(f"[Live] PSI on new data = {psi_val:.4f}")
            if psi_val > 0.05:
                logging.info("[Live] Data changed by more than 5% => Retraining recommended.")

            needed_rows = self.window_size + 1
            if self.train_raw_window is not None:
                combined_window = pd.concat([self.train_raw_window, merged_df], ignore_index=True).tail(needed_rows)
            else:
                combined_window = merged_df.tail(needed_rows)

            if self.window_size == 1:
                X_live, _ = self.prep.ready(
                    combined_window,
                    window=1,
                    selected_features=self.feats,
                    mode='predict'
                )
            else:
                X_live, _ = self.prep.ready_incremental(
                    combined_window,
                    window=self.window_size,
                    selected_features=self.feats
                )

            if X_live.empty:
                logging.info("[Live] X_live empty => skip.")
                self.clean_files()
                return

            print("========== DEBUG FEATURE NAMES CHECK ==========")
            self.train_window_cols = [str(c) for c in self.train_window_cols]
            X_live.columns = [str(c) for c in X_live.columns]

            set_train = set(self.train_window_cols)
            set_live = set(X_live.columns)
            print("train_window_cols count:", len(self.train_window_cols))
            print("X_live columns count:", len(X_live.columns))
            print("Missing in X_live =>", set_train - set_live)
            print("Extra in X_live =>", set_live - set_train)

            X_live = X_live.reindex(columns=self.train_window_cols, fill_value=0)
            print("After reindex =>", X_live.shape)
            print("========== END DEBUG FEATURE NAMES CHECK ==========")

            X_live = X_live.astype(float, copy=False)
            X_arr = X_live.to_numpy()

            try:
                scaler_step = self.pipeline.named_steps.get('scaler', None)
                if scaler_step is not None:
                    X_trans = scaler_step.transform(X_arr)
                else:
                    X_trans = X_arr

                clf_step = self.pipeline.named_steps.get('classifier', None)
                if clf_step is not None:
                    proba_array = clf_step.predict_proba(X_trans)
                else:
                    proba_array = self.pipeline.predict_proba(X_trans)

                proba = proba_array[:, 1][0]

            except NotFittedError as e:
                logging.error(f"[Live] Model not fitted => {e}")
                print(f"[Live] Model not fitted => {e}")
                self.clean_files()
                return
            except Exception as e:
                logging.error(f"[Live] Error at transform or predict_proba => {e}")
                print(f"[Live] Error at transform or predict_proba => {e}")
                self.clean_files()
                return

            if proba <= self.neg_thr:
                pred = 0
            elif proba >= self.pos_thr:
                pred = 1
            else:
                pred = -1

            if pred == -1:
                txt_line = f"NAN,{proba:.4f}"
            elif pred == 0:
                txt_line = f"SEL,{proba:.4f}"
            else:
                txt_line = f"BUY,{proba:.4f}"

            with open(self.answer_file, 'w') as f:
                f.write(txt_line + "\n")

            logging.info(f"[Live] Prediction => {txt_line}")
            print(f"[Live] Prediction => {txt_line}")

            self.clean_files()

        except Exception as e:
            logging.error(f"[Live] General Exception => {e}")
            print(f"[Live] General Exception => {e}")

    def clean_files(self):
        for tf in self.filepaths:
            if os.path.exists(self.filepaths[tf]):
                os.remove(self.filepaths[tf])

def main_loop():
    predictor = LivePredictor()
    while True:
        predictor.check_and_predict()
        time.sleep(1)
        remove_files_if_exists()

if __name__ == "__main__":
    main_loop()
