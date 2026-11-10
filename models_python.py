"""The five learners and their search spaces (created by the Phase 4 notebook)."""
import numpy as np  # numpy: numerical computing with arrays
import pyBlindOpt as pbo  # pyBlindOpt: evolutionary algorithms from the course
from sklearn.base import BaseEstimator, ClassifierMixin  # base classes to build our own model
from sklearn.calibration import CalibratedClassifierCV  # turns the SVM output into probabilities
from sklearn.naive_bayes import GaussianNB  # Bayesian learner
from sklearn.neural_network import MLPClassifier  # connectionist learner
from sklearn.pipeline import Pipeline  # chains preprocessing + model
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # preprocessing transformations
from sklearn.svm import SVC  # analogizer learner
from sklearn.tree import DecisionTreeClassifier  # symbolic learner

def build(scale, model):  # preprocessing + model in a single object (no data leakage)
    scaling = StandardScaler() if scale else "passthrough"  # standardize only if the learner needs it
    return Pipeline([("log1p", "passthrough"), ("scaling", scaling), ("model", model)])  # log1p switched off by default; Phase 5 decides whether to switch it on

LOG1P = FunctionTransformer(np.log1p)  # log(1 + x) transformation, to be switched on in Phase 5
LOG_OPTIONS = ["passthrough", LOG1P]  # two options: untransformed or with log1p

def population_mcc(Z, y, W):  # MCC of each individual in the population (each row of W is a classifier)
    pred = (Z @ W[:, :-1].T + W[:, -1]) > 0  # each column: predictions of one individual (w·x + b > 0)
    real = y.astype(bool)[:, None]  # true diagnosis as a True/False column
    tp, tn = (pred & real).sum(0), (~pred & ~real).sum(0)  # true positives and negatives of each individual
    fp, fn = (pred & ~real).sum(0), (~pred & real).sum(0)  # false positives and negatives of each individual
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).astype(float)  # denominator of the MCC
    return np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1.0), 0.0)  # MCC (0 if the denominator is 0)

class EvolutionaryClassifier(BaseEstimator, ClassifierMixin):  # our own model, compatible with scikit-learn
    def __init__(self, population=40, generations=100, bound=5.0, F=0.5, cr=0.7, seed=42):  # evolution parameters
        self.population, self.generations, self.bound = population, generations, bound  # size, number of generations and range of the weights
        self.F, self.cr, self.seed = F, cr, seed  # mutation strength, crossover probability and seed

    def fit(self, X, y):  # "learning" = making the population of weights evolve
        X, y = np.asarray(X, float), np.asarray(y)  # make sure we have numpy arrays
        self.classes_ = np.array([0, 1])  # the two classes
        bounds = np.array([[-self.bound, self.bound]] * (X.shape[1] + 1))  # each weight (30 weights + 1 bias) between -5 and 5
        objective = lambda W: -population_mcc(X, y, np.atleast_2d(W))  # DE minimizes, so we minimize -MCC
        evolution = pbo.DifferentialEvolution(objective, bounds, n_pop=self.population, n_iter=self.generations, F=self.F, cr=self.cr, seed=self.seed, debug=True)  # differential evolution
        best, score, _ = evolution.optimize()  # run the evolution and return the best individual
        self.coef_, self.intercept_ = best[:-1], best[-1]  # the 30 weights and the bias of the best individual
        self.train_mcc_ = -float(score)  # MCC of the best individual on the training set
        self.history_ = -evolution.history[:, 0]  # best MCC of each generation
        return self  # scikit-learn expects fit to return the model itself

    def decision_function(self, X):  # value w·x + b of each tumour
        return np.asarray(X, float) @ self.coef_ + self.intercept_  # dot product plus bias

    def predict_proba(self, X):  # "probability" of malignant (sigmoid; NOT calibrated)
        p = 1.0 / (1.0 + np.exp(-np.clip(self.decision_function(X), -50, 50)))  # sigmoid of the linear output
        return np.column_stack([1 - p, p])  # column 0 = benign, column 1 = malignant

    def predict(self, X):  # predicted class: malignant if w·x + b > 0
        return (self.decision_function(X) > 0).astype(int)  # 1 = malignant

learners = {  # dictionary: key -> (class, untrained pipeline, hyperparameter grid for Phase 5)
    "tree": ("Symbolic", build(False, DecisionTreeClassifier(random_state=42)),
             {"model__criterion": ["gini", "entropy"], "model__max_depth": [2, 3, 4, 5, 6, None], "model__min_samples_leaf": [1, 2, 5, 10]}),  # the tree does not use log1p
    "mlp": ("Connectionist", build(True, MLPClassifier(solver="lbfgs", max_iter=1000, random_state=42)),
            {"log1p": LOG_OPTIONS, "model__hidden_layer_sizes": [(4,), (8,), (16,), (32, 16)], "model__alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]}),  # alpha = regularization
    "evolutionary": ("Evolutionary", build(True, EvolutionaryClassifier()),
                     {"log1p": LOG_OPTIONS}),  # evolution is the learning: its budget is fixed and is not tuned
    "bayes": ("Bayesian", build(False, GaussianNB()),
              {"log1p": LOG_OPTIONS, "model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1]}),  # variance smoothing
    "svm": ("Analogizer", build(True, CalibratedClassifierCV(SVC(kernel="rbf", random_state=42), method="sigmoid", ensemble=False, cv=3)),
            {"log1p": LOG_OPTIONS, "model__estimator__C": [0.1, 1.0, 10.0, 100.0, 1000.0], "model__estimator__gamma": [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]}),  # C and gamma of the SVM
}  # end of the dictionary
