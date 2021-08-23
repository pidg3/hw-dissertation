# These helper functions are reused from the work by Slack et al.
# Refer to https://arxiv.org/abs/1911.02508 for the paper
# Refer to https://github.com/dylan-slack/Fooling-LIME-SHAP for the code

import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from copy import deepcopy

import shap


def one_hot_encode(y):
    """ One hot encode y for binary features.  We use this to get from 1 dim ys to predict proba's.
    This is taken from this s.o. post: https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array

    Parameters
    ----------
    y : np.ndarray

    Returns
    ----------
    A np.ndarray of the one hot encoded data.
    """
    y_hat_one_hot = np.zeros((len(y), 2))
    y_hat_one_hot[np.arange(len(y)), y] = 1
    return y_hat_one_hot


class Params:
    """Parameters object taken from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py
  
  Parameters
  ----------
  json_path : string

  Returns
  ----------
  Parameters object
  """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class Adversarial_Model(object):
    """	A scikit-learn style adversarial explainer base class for adversarial models.  This accetps 
	a scikit learn style function f_obscure that serves as the _true classification rule_ for in distribution
	data.  Also, it accepts, psi_display: the classification rule you wish to display by explainers (e.g. LIME/SHAP).
	Ideally, f_obscure will classify individual instances but psi_display will be shown by the explainer.

	Parameters
	----------
	f_obscure : function
	psi_display : function
	"""

    def __init__(self, f_obscure, psi_display):
        self.f_obscure = f_obscure
        self.psi_display = psi_display

        self.cols = None
        self.scaler = None
        self.numerical_cols = None

    def predict_proba(self, X, threshold=0.5):
        """ Scikit-learn style probability prediction for the adversarial model.  

		Parameters
		----------
		X : np.ndarray

		Returns
		----------
		A numpy array of the class probability predictions of the advesarial model.
		"""
        if self.perturbation_identifier is None:
            raise NameError("Model is not trained yet, can't perform predictions.")

        # generate the "true" predictions on the data using the "bad" model -- this is f in the paper
        predictions_to_obscure = self.f_obscure.predict_proba(X)
        # generate the "explain" predictions -- this is psi in the paper

        predictions_to_explain_by = self.psi_display.predict_proba(X)

        # in the case that we're only considering numerical columns
        if self.numerical_cols:
            X = X[:, self.numerical_cols]

        # MP - generate predictions of whether the point is on or off manifold
        pred_probs = self.perturbation_identifier.predict_proba(X)

        # allow thresholding for finetuned control over psi_display and f_obscure
        # MP - the [:,1] is just taking the second value, this is arbitrary, we could equally take [:,0] (first value)
        # and check whether <= threshold
        perturbation_preds = pred_probs[:, 1] >= threshold

        sol = np.where(
            np.array([perturbation_preds == 1, perturbation_preds == 1]).transpose(),
            predictions_to_obscure,
            predictions_to_explain_by,
        )
        return sol

    def predict(self, X):
        """	Scikit-learn style prediction. Follows from predict_proba.

		Parameters
		----------
		X : np.ndarray
		
		Returns
		----------
		A numpy array containing the binary class predictions.
		"""

        if X.ndim == 1:
            X = np.array([X])

        pred_probs = self.predict_proba(X)
        return np.argmax(pred_probs, axis=1)

    def score(self, X_test, y_test):
        """ Scikit-learn style accuracy scoring.

		Parameters:
		----------
		X_test : X_test
		y_test : y_test

		Returns:
		----------
		A scalar value of the accuracy score on the task.
		"""

        return np.sum(self.predict(X_test) == y_test) / y_test.size

    def get_column_names(self):
        """ Access column names."""

        if self.cols is None:
            raise NameError("Train model with pandas data frame to get column names.")

        return self.cols

    def fidelity(self, X):
        """ Get the fidelity of the adversarial model to the original predictions.  High fidelity means that
		we're predicting f along the in distribution data.
		
		Parameters:
		----------
		X : np.ndarray	

		Returns:
		----------
		The fidelity score of the adversarial model's predictions to the model you're trying to obscure's predictions.
		"""

        return np.sum(self.predict(X) == self.f_obscure.predict(X)) / X.shape[0]


class Adversarial_Kernel_SHAP_Model(Adversarial_Model):
    """ SHAP adversarial model.  Generates an adversarial model for SHAP style perturbations.

	Parameters:
	----------
	f_obscure : function
	psi_display : function
	"""

    def __init__(self, f_obscure, psi_display):
        super(Adversarial_Kernel_SHAP_Model, self).__init__(f_obscure, psi_display)

    def train(
        self,
        X,
        y,
        feature_names,
        background_distribution=None,
        perturbation_multiplier=10,
        n_samples=2e4,
        rf_estimators=100,
        n_kmeans=10,
        estimator=None,
    ):
        """ Trains the adversarial SHAP model. This method perturbs the shap training distribution by sampling from 
		its kmeans and randomly adding features.  These points get substituted into a test set.  We also check to make 
		sure that the instance isn't in the test set before adding it to the out of distribution set. If an estimator is 
		provided this is used.

		Parameters:
		----------
		X : np.ndarray
		y : np.ndarray
		features_names : list
		perturbation_multiplier : int
		n_samples : int or float
		rf_estimators : int
		n_kmeans : int
		estimator : func

		Returns:
		----------
		The model itself.
		"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise NameError(
                "X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(
                    type(X)
                )
            )

        self.cols = feature_names

        # This is the mock background distribution we'll pull from to create substitutions
        if background_distribution is None:
            background_distribution = shap.kmeans(X, n_kmeans).data
        repeated_X = np.repeat(X, perturbation_multiplier, axis=0)

        new_instances = []
        equal = []

        # We generate n_samples number of substutions
        # MP - this just seems to be selecting a random instance
        for _ in range(int(n_samples)):
            i = np.random.choice(X.shape[0])
            point = deepcopy(X[i, :])

            # iterate over points, sampling and updating
            # MP - seems here we are using data for random features in order to generate points off the manifold
            for _ in range(X.shape[1]):
                j = np.random.choice(X.shape[1])
                point[j] = deepcopy(
                    background_distribution[
                        np.random.choice(background_distribution.shape[0]), j
                    ]
                )

            new_instances.append(point)

        substituted_training_data = np.vstack(new_instances)
        all_instances_x = np.vstack((repeated_X, substituted_training_data))

        # make sure feature truly is out of distribution before labeling it
        xlist = X.tolist()
        ys = np.array(
            [
                1 if substituted_training_data[val, :].tolist() in xlist else 0
                for val in range(substituted_training_data.shape[0])
            ]
        )

        all_instances_y = np.concatenate((np.ones(repeated_X.shape[0]), ys))

        xtrain, xtest, ytrain, ytest = train_test_split(
            all_instances_x, all_instances_y, test_size=0.2
        )

        if estimator is not None:
            self.perturbation_identifier = estimator.fit(xtrain, ytrain)
        else:
            self.perturbation_identifier = RandomForestClassifier(
                n_estimators=rf_estimators
            ).fit(xtrain, ytrain)

        ypred = self.perturbation_identifier.predict(xtest)
        self.ood_training_task_ability = (ytest, ypred)

        return self

