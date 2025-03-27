#!/usr/bin/env python3
import numpy as np
import pandas as pd
import logging
import random
import copy
import multiprocessing
from collections import Counter
import warnings
import os
import gc  # مدیریت حافظه
import signal

from deap import base, creator, tools
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from threshold_finder import ThresholdFinder
from model_pipeline import ModelPipeline
from drift_checker import DriftChecker

warnings.filterwarnings("ignore")

logging.basicConfig(
    filename='genetic_algorithm.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# ==== پارامترهای مربوط به الگوریتم ژنتیک ====
POPULATION_SIZE = 20
N_GENERATIONS   = 7
CX_PB           = 0.8
MUT_PB          = 0.4
EARLY_STOPPING_THRESHOLD = 0.95
MIN_FEATURES    = 10

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

def rand_C():
    return 10 ** random.uniform(-4, 2)

def rand_max_iter():
    return random.randint(100, 1000)

def rand_tol():
    return 10 ** random.uniform(-6, -2)

def rand_penalty():
    return random.choice(['l1', 'l2'])

def rand_solver():
    return random.choice(['lbfgs', 'liblinear', 'sag', 'saga'])

def rand_fit_intercept():
    return random.choice([True, False])

def rand_class_weight():
    return random.choice([None, 'balanced'])

def rand_multi_class():
    return random.choice(['auto', 'ovr', 'multinomial'])

POSSIBLE_WINDOW_SIZES = [1, 2, 3, 4, 5, 6]
def rand_window_size():
    return random.choice(POSSIBLE_WINDOW_SIZES)

def create_individual(feature_mask_length):
    penalty_ = random.choice(['l1', 'l2'])
    if penalty_ == 'l1':
        solver_ = random.choice(['liblinear', 'saga'])
    else:  # l2
        solver_ = random.choice(['lbfgs', 'liblinear', 'sag', 'saga'])

    c_        = 10 ** random.uniform(-4, 2)
    max_iter_ = random.randint(100, 1000)
    tol_      = 10 ** random.uniform(-6, -2)
    fitint_   = random.choice([True, False])
    cw_       = random.choice([None, 'balanced'])

    mc_ = random.choice(['auto', 'ovr', 'multinomial'])
    if solver_ == 'liblinear' and mc_ == 'multinomial':
        mc_ = random.choice(['auto', 'ovr'])

    w_size = random.choice([1, 2, 3, 4, 5, 6])

    feat_mask = [random.choice([0, 1]) for _ in range(feature_mask_length)]
    if sum(feat_mask) < MIN_FEATURES:
        idxs = random.sample(range(feature_mask_length), MIN_FEATURES - sum(feat_mask))
        for ix in idxs:
            feat_mask[ix] = 1

    return [c_, max_iter_, tol_, penalty_, solver_, fitint_, cw_, mc_, w_size, feat_mask]


def mutate_individual(individual, indpb=0.2):
    for i in range(len(individual)):
        if random.random() < indpb:
            if i == 0:
                individual[0] = 10 ** random.uniform(-4, 2)
            elif i == 1:
                individual[1] = random.randint(100, 1000)
            elif i == 2:
                individual[2] = 10 ** random.uniform(-6, -2)
            elif i == 3:
                individual[3] = random.choice(['l1', 'l2'])
            elif i == 4:
                individual[4] = random.choice(['lbfgs', 'liblinear', 'sag', 'saga'])
            elif i == 5:
                individual[5] = random.choice([True, False])
            elif i == 6:
                individual[6] = random.choice([None, 'balanced'])
            elif i == 7:
                individual[7] = random.choice(['auto', 'ovr', 'multinomial'])
            elif i == 8:
                individual[8] = random.choice([1, 2, 3, 4, 5, 6])
            elif i == 9:
                feat_mask = individual[9]
                feat_mask = [1 - bit if random.random() < indpb else bit for bit in feat_mask]
                individual[9] = feat_mask

    # سازگاری دوباره
    penalty_ = individual[3]
    solver_  = individual[4]
    mc_      = individual[7]
    feat_mask = individual[9]

    if penalty_ == 'l1' and solver_ not in ['liblinear','saga']:
        individual[4] = random.choice(['liblinear','saga'])
    elif penalty_ == 'l2' and solver_ not in ['lbfgs','liblinear','sag','saga']:
        individual[4] = random.choice(['lbfgs','liblinear','sag','saga'])

    if individual[4] == 'liblinear' and mc_ == 'multinomial':
        individual[7] = random.choice(['auto','ovr'])

    if sum(feat_mask) < MIN_FEATURES:
        idxs = random.sample(range(len(feat_mask)), MIN_FEATURES - sum(feat_mask))
        for ix in idxs:
            feat_mask[ix] = 1
        individual[9] = feat_mask

    return (individual,)

def init_individual(feature_mask_length):
    return creator.Individual(create_individual(feature_mask_length))

def evaluate_cv(individual, timeout=10000):

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"evaluate_cv timed out after {timeout} seconds.")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        global data_train_shared, prep_shared, selected_features_shared

        c_, mxiter_, tol_, penalty_, solver_, fitint_, cw_, mc_, w_size, feat_mask = individual

        # ناسازگاری
        if penalty_ == 'l1' and solver_ not in ['liblinear', 'saga']:
            return (0.0,)
        if penalty_ == 'l2' and solver_ not in ['lbfgs', 'liblinear', 'sag', 'saga']:
            return (0.0,)
        if solver_ == 'liblinear' and mc_ == 'multinomial':
            return (0.0,)

        if len(feat_mask) != len(selected_features_shared):
            return (0.0,)

        X_, y_, _ = prep_shared.ready(data_train_shared, window=w_size, selected_features=selected_features_shared, mode='train')
        if X_.empty or y_.empty:
            return (0.0,)

        X_.reset_index(drop=True, inplace=True)
        y_.reset_index(drop=True, inplace=True)

        expanded_mask = []
        for b_ in feat_mask:
            expanded_mask.extend([b_] * w_size)
        if len(expanded_mask) != X_.shape[1]:
            return (0.0,)

        use_cols = [col for col, b_ in zip(X_.columns, expanded_mask) if b_ == 1]
        if not use_cols:
            return (0.0,)

        X_f = X_[use_cols].copy()
        y_f = y_.copy()

        hyperparams = {
            'C': c_,
            'max_iter': mxiter_,
            'tol': tol_,
            'penalty': penalty_,
            'solver': solver_,
            'fit_intercept': fitint_,
            'class_weight': cw_,
            'multi_class': mc_
        }
        pipeline = ModelPipeline(hyperparams)

        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        from collections import Counter
        for tr_idx, ts_idx in tscv.split(X_f, y_f):
            X_tr, y_tr = X_f.iloc[tr_idx], y_f.iloc[tr_idx]
            X_ts, y_ts = X_f.iloc[ts_idx], y_f.iloc[ts_idx]

            num_ones = Counter(y_tr)[1]
            if num_ones >= 2:
                try:
                    from imblearn.over_sampling import SMOTE
                    sm = SMOTE(k_neighbors=min(3, num_ones - 1))
                    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
                except:
                    X_tr_res, y_tr_res = (X_tr, y_tr)
            else:
                X_tr_res, y_tr_res = (X_tr, y_tr)

            pipeline.fit(X_tr_res, y_tr_res)
            y_pred = pipeline.pipeline.predict(X_ts)
            from sklearn.metrics import f1_score, accuracy_score
            f1_ = f1_score(y_ts, y_pred, average='binary')
            acc_ = accuracy_score(y_ts, y_pred)
            scores.append(0.5 * (f1_ + acc_))

            gc.collect()

        if not scores:
            return (0.0,)

        return (float(np.mean(scores)),)

    except TimeoutError:
        logging.warning("TimeoutError: evaluate_cv took too long. Returning 0.0 for fitness.")
        return (0.0,)

    except Exception as e:
        logging.error(f"evaluate_cv error: {e}")
        return (0.0,)

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def pool_init(data_train, prep_obj, selected_features):
    global data_train_shared, prep_shared, selected_features_shared
    data_train_shared = data_train
    prep_shared = prep_obj
    selected_features_shared = selected_features

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_individual, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

class GeneticAlgorithmRunner:
    def __init__(self):
        self.final_window_cols_ = []
        self.train_raw_window = None

    def main(self):
        prep = PREPARE_DATA_FOR_TRAIN(main_timeframe='30T')
        raw_data = prep.load_data()
        global_data = raw_data.copy()

        hour_col = f"{prep.main_timeframe}_Hour"
        if hour_col not in global_data.columns:
            logging.error(f"Column '{hour_col}' not found.")
            raise ValueError(f"'{hour_col}' not found in data.")
        global_data['Hour'] = global_data[hour_col]

        n = len(global_data)
        if n <= 0:
            logging.error("No data after merges => no rows.")
            return None, 0.0

        train_ratio = 0.75
        thresh_ratio = 0.05
        test_ratio = 0.20
        train_end = int(train_ratio * n)
        thresh_end = train_end + int(thresh_ratio * n)
        test_end = n

        data_train = global_data.iloc[:train_end].copy()
        data_thresh = global_data.iloc[train_end:thresh_end].copy()
        data_test = global_data.iloc[thresh_end:test_end].copy()

        X_train_fs, y_train_fs, feats = prep.ready(data_train, window=1, selected_features=None, mode='train')
        logging.info(f"Number of selected features on Train: {len(feats)}")

        drift = DriftChecker()
        drift.fit_on_train(X_train_fs)
        drift.save_train_distribution("train_distribution.json")

        n_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=n_processes, initializer=pool_init, initargs=(data_train, prep, feats))
        toolbox.unregister("map")
        # chunksize=1 برای کاهش حجم حافظه در هر تسک
        toolbox.register("map", lambda func, iterable: list(pool.imap_unordered(func, iterable, chunksize=1)))

        num_feats = len(feats)
        toolbox.register("init_individual", init_individual, feature_mask_length=num_feats)

        population = [toolbox.init_individual() for _ in range(POPULATION_SIZE)]
        fitnesses = toolbox.map(evaluate_cv, population)
        for ind, fitv in zip(population, fitnesses):
            ind.fitness.values = fitv

        best_overall = 0.0
        for gen in range(1, N_GENERATIONS + 1):
            logging.info(f"Generation {gen}/{N_GENERATIONS}")

            offspring = toolbox.select(population, len(population))
            offspring = [copy.deepcopy(o) for o in offspring]

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CX_PB:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for mut in offspring:
                if random.random() < MUT_PB:
                    toolbox.mutate(mut)
                    del mut.fitness.values

            invalids = [ind for ind in offspring if not ind.fitness.valid]
            if invalids:
                fits = toolbox.map(evaluate_cv, invalids)
                for ind_, fv in zip(invalids, fits):
                    ind_.fitness.values = fv

            population[:] = offspring
            gc.collect()

            best_ind = tools.selBest(population, 1)[0]
            best_score = best_ind.fitness.values[0]
            if best_score > best_overall:
                best_overall = best_score

            logging.info(f"Gen {gen}: best_score={best_score:.4f}")

            if best_score >= EARLY_STOPPING_THRESHOLD:
                logging.info("Early stopping triggered.")
                break

        best_ind = tools.selBest(population, 1)[0]
        best_score = best_ind.fitness.values[0]
        logging.info(f"Optimization finished. Best score={best_score:.4f}")
        print(f"[GA] best individual => {best_ind}, best_score={best_score:.4f}")

        c_, mxiter_, tl_, pty_, slv_, ftint_, cw_, mc_, w_size, feat_mask = best_ind
        if w_size > 1:
            self.train_raw_window = data_train.tail(w_size - 1).copy()
        else:
            self.train_raw_window = None

        final_model = self.build_final_model(best_ind, data_train, prep, feats)
        if final_model is None:
            logging.warning("No final_model => skip")
            pool.close()
            pool.join()
            return best_ind, best_score

        self.run_threshold_finder(final_model, data_thresh, prep, best_ind, feats)
        self.evaluate_with_threshold(final_model, data_test, prep, best_ind, feats, label="Test")
        self.save_model_and_threshold(final_model, best_ind, feats)
        pool.close()
        pool.join()
        gc.collect()
        return best_ind, best_score

    def build_final_model(self, best_ind, data_train, prep, feats):
        c_, mxiter_, tl_, pty_, slv_, ftint_, cw_, mc_, w_size, feat_mask = best_ind

        if pty_ == 'l1' and slv_ not in ['liblinear', 'saga']:
            return None
        if pty_ == 'l2' and slv_ not in ['lbfgs', 'liblinear', 'sag', 'saga']:
            return None
        if slv_ == 'liblinear' and mc_ == 'multinomial':
            return None

        X_train, y_train, _ = prep.ready(data_train, window=w_size, selected_features=feats, mode='train')
        if X_train.empty or y_train.empty:
            return None

        expanded_mask = []
        for b_ in feat_mask:
            expanded_mask.extend([b_] * w_size)
        if len(expanded_mask) != X_train.shape[1]:
            return None
        use_cols = [c for c, b_ in zip(X_train.columns, expanded_mask) if b_ == 1]
        if not use_cols:
            return None

        X_train_final = X_train[use_cols].copy()

        from imblearn.over_sampling import SMOTE
        num_ones = Counter(y_train)[1]
        if num_ones >= 2:
            try:
                sm = SMOTE(k_neighbors=min(3, num_ones - 1))
                X_train_res, y_train_res = sm.fit_resample(X_train_final, y_train)
            except:
                X_train_res, y_train_res = (X_train_final, y_train)
        else:
            X_train_res, y_train_res = (X_train_final, y_train)

        hyperparams = {
            'C': c_,
            'max_iter': mxiter_,
            'tol': tl_,
            'penalty': pty_,
            'solver': slv_,
            'fit_intercept': ftint_,
            'class_weight': cw_,
            'multi_class': mc_
        }

        final_model = ModelPipeline(hyperparams)
        final_model.fit(X_train_res, y_train_res)
        logging.info("Final LR model trained.")

        self.final_window_cols_ = X_train_final.columns.tolist()
        logging.info(f"[build_final_model] final_window_cols_ => {len(self.final_window_cols_)} columns")

        return final_model

    def run_threshold_finder(self, model_pipeline, data_thresh, prep, best_ind, feats):
        if data_thresh.empty:
            logging.info("[Threshold] data_thresh empty => skip.")
            return

        c_, mxiter_, tl_, pty_, slv_, ftint_, cw_, mc_, w_size, feat_mask = best_ind
        X_thr, y_thr, _ = prep.ready(data_thresh, window=w_size, selected_features=feats, mode='train')
        if X_thr.empty or y_thr.empty:
            logging.warning("[Threshold] empty => skip threshold finder.")
            return

        expanded_mask = []
        for b_ in feat_mask:
            expanded_mask.extend([b_] * w_size)
        if len(expanded_mask) != X_thr.shape[1]:
            logging.warning("[Threshold] mismatch => skip.")
            return

        use_cols = [c for c, b_ in zip(X_thr.columns, expanded_mask) if b_ == 1]
        if not use_cols:
            return

        X_thr_f = X_thr[use_cols].copy()
        try:
            y_proba_thr = model_pipeline.predict_proba(X_thr_f)[:, 1]
        except:
            logging.error("[Threshold] predict_proba failed => skip.")
            return

        tfinder = ThresholdFinder(steps=200, min_predictions_ratio=2/3)
        neg_thr, pos_thr, best_acc_thr, w1_thr, l1_thr = tfinder.find_best_thresholds(y_proba_thr, y_thr.values)
        self.neg_thr = neg_thr
        self.pos_thr = pos_thr
        logging.info(f"[ThresholdFinder] => neg={neg_thr:.3f}, pos={pos_thr:.3f}, partial-acc={best_acc_thr:.3f}")
        print(f"[Threshold] neg={neg_thr:.3f}, pos={pos_thr:.3f}, partial-acc={best_acc_thr:.3f}")

    def evaluate_with_threshold(self, model_pipeline, data_part, prep, best_ind, feats, label="Test"):
        if data_part.empty:
            logging.info(f"[{label}] data_part empty => skip.")
            return
        neg_thr = getattr(self, 'neg_thr', 0.5)
        pos_thr = getattr(self, 'pos_thr', 0.5)

        c_, mxiter_, tl_, pty_, slv_, ftint_, cw_, mc_, w_size, feat_mask = best_ind
        X_p, y_p, _ = prep.ready(data_part, window=w_size, selected_features=feats, mode='train')
        if X_p.empty or y_p.empty:
            logging.warning(f"[{label}] empty => skip.")
            return

        expanded_mask = []
        for b_ in feat_mask:
            expanded_mask.extend([b_] * w_size)
        if len(expanded_mask) != X_p.shape[1]:
            logging.error(f"[{label}] mismatch => skip.")
            return

        use_cols = [c for c, b_ in zip(X_p.columns, expanded_mask) if b_ == 1]
        if not use_cols:
            logging.error(f"[{label}] no features => skip.")
            return

        X_part_f = X_p[use_cols].copy()
        try:
            y_proba = model_pipeline.predict_proba(X_part_f)[:, 1]
        except:
            logging.error(f"[{label}] predict_proba failed => skip.")
            return

        y_pred = np.full_like(y_p, -1, dtype=int)
        y_pred[y_proba <= neg_thr] = 0
        y_pred[y_proba >= pos_thr] = 1

        mask_confident = (y_pred != -1)
        confident_ratio = mask_confident.mean()
        y_pred_conf = y_pred[mask_confident]
        y_true_conf = y_p.values[mask_confident]

        from sklearn.metrics import f1_score, accuracy_score
        if len(y_pred_conf) > 0:
            f1_ = f1_score(y_true_conf, y_pred_conf, average='binary')
            acc_ = accuracy_score(y_true_conf, y_pred_conf)
        else:
            f1_, acc_ = 0.0, 0.0

        logging.info(f"[{label}] #samples={len(X_part_f)}, conf_ratio={confident_ratio:.2f}, F1={f1_:.4f}, Acc={acc_:.4f}")
        print(f"[{label}] #samples={len(X_part_f)}, conf_ratio={confident_ratio:.2f}, F1={f1_:.4f}, Acc={acc_:.4f}")

    def save_model_and_threshold(self, final_model, best_ind, feats):
        c_, mxiter_, tl_, pty_, slv_, ftint_, cw_, mc_, w_size, feat_mask = best_ind

        hyperparams = {
            'C': c_,
            'max_iter': mxiter_,
            'tol': tl_,
            'penalty': pty_,
            'solver': slv_,
            'fit_intercept': ftint_,
            'class_weight': cw_,
            'multi_class': mc_
        }

        scaler = None
        if 'scaler' in final_model.pipeline.named_steps:
            scaler = final_model.pipeline.named_steps['scaler']

        neg_thr = getattr(self, 'neg_thr', 0.5)
        pos_thr = getattr(self, 'pos_thr', 0.5)

        data_to_save = {
            'pipeline': final_model.pipeline,
            'hyperparams': hyperparams,
            'feat_mask': feat_mask,
            'neg_thr': neg_thr,
            'pos_thr': pos_thr,
            'scaler': scaler,
            'window_size': w_size,
            'feats': feats,
            'train_window_cols': self.final_window_cols_
        }

        if self.train_raw_window is not None:
            data_to_save['train_raw_window'] = self.train_raw_window

        # اعمال فشرده سازی 
        joblib.dump(data_to_save, "best_model.pkl", compress=3)
        logging.info(f"[save_model_and_threshold] final_window_cols_ => {len(self.final_window_cols_)} stored.")
        logging.info("[save_model_and_threshold] Model, thresholds, feats, w_size, final_window_cols saved.")

if __name__ == "__main__":
    runner = GeneticAlgorithmRunner()
    best_ind, best_score = runner.main()
    print(f"[MAIN] Done. best_ind => {best_ind}, best_score={best_score:.4f}")
