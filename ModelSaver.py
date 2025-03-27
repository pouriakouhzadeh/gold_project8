# ModelSaver.py

import joblib

class ModelSaver:
    def __init__(self, filename: str = 'best_logreg_pipeline.pkl'):
        """
        سازنده کلاس ModelSaver.
        :param filename: نام فایل برای ذخیره مدل
        """
        self.filename = filename

    def save(self, pipeline, scaler, feature_names, hyperparams):
        """
        ذخیره مدل نهایی به صورت فایل pickle.
        :param pipeline: شیء Pipeline آموزش‌داده‌شده
        :param scaler: (اختیاری) شیء scaler
        :param feature_names: لیست نام فیچرهای انتخاب‌شده
        :param hyperparams: دیکشنری شامل هایپرپارامترهای مدل
        """
        model_data = {
            'pipeline': pipeline,
            'scaler': scaler,
            'feature_names': feature_names,
            'hyperparams': hyperparams
        }
        joblib.dump(model_data, self.filename)
