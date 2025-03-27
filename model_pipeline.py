from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class ModelPipeline:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=self.hyperparams.get('C', 1.0),
                max_iter=self.hyperparams.get('max_iter', 100),
                tol=self.hyperparams.get('tol', 1e-4),
                penalty=self.hyperparams.get('penalty', 'l2'),
                solver=self.hyperparams.get('solver', 'lbfgs'),
                fit_intercept=self.hyperparams.get('fit_intercept', True),
                class_weight=self.hyperparams.get('class_weight', None),
                multi_class=self.hyperparams.get('multi_class', 'auto'),
                random_state=42,
                n_jobs=-1
            ))
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
