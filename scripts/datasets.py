import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder

from scripts.consts import proj_paths

class Bucketizer(TransformerMixin, BaseEstimator):
    """Bucketize numerical features.
    
    For each feature, sort samples according to that feature. For each
    pair of consecutive samples with different class labels and
    different feature values, define a candidate threshold for a split
    as the average of these two observations’ feature values.
    
    Attributes
    ----------
    thresholds_ : dict
        Dictionary mapping features to list of thresholds.
    """
    def fit(self, X, y):
        """Finds all candidate split thresholds for each feature.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The feature vectors to be bucketized.
        
        y : pandas Series of shape (n_samples,)
            The target values (class labels).
        
        Returns
        -------
        self : Bucketizer
            Fitted bucketizer.
        """
        if y.name is None:
            y.name = 'labels'
        X_y = X.join(y)
        self.thresholds_ = {}
        for j in X.columns:
            thresholds = []
            sorted_X_y = X_y.sort_values([j, y.name])
            prev_feature_val = sorted_X_y.iloc[0][j]
            prev_label = sorted_X_y.iloc[0][y.name]
            for idx, row in sorted_X_y.iterrows():
                curr_feature_val, curr_label = row[j], row[y.name]
                if (curr_label != prev_label
                    and prev_feature_val < curr_feature_val):
                    thresh = (prev_feature_val + curr_feature_val)/2
                    thresholds.append(thresh)
                prev_feature_val, prev_label = curr_feature_val, curr_label
            self.thresholds_[j] = thresholds
        return self
    
    def transform(self, X):
        """Perform bucketization using candidate thresholds.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The feature vectors to be bucketized.
        
        Returns
        -------
        Xt : pandas DataFrame
            Transformed data.
        """
        check_is_fitted(self)
        Xt = {}
        for j in X.columns:
            for threshold in self.thresholds_[j]:
                Xt[f"X[{j}] <= {threshold}"] = (X[j] <= threshold)
        Xt = pd.DataFrame(Xt)
        return Xt

class QuantileBucketizer(TransformerMixin, BaseEstimator):
    """Bucketize numerical features using quantiles.
    
    Internally use scikit-learn's KBinsDiscretizer, with the
    modification that for each feature, bins to the left of the bin
    containing the feature value are also filled with ones. Thus,
    transformed features correspond to threshold tests.
    
    Parameters
    ----------
    n_quantiles : positive int, default=5
        The number of quantiles.
    
    Attributes
    ----------
    discretizer_ : KBinsDiscretizer
        KBinsDiscretizer instance.
    """
    def __init__(
        self,
        n_quantiles=5
    ):
        self.n_quantiles = n_quantiles
    
    def fit(self, X, y=None):
        """Fit a KBinsDiscretizer to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature vectors to be bucketized.
        
        y : None
            Ignored.
        
        Returns
        -------
        self : QuantileBucketizer
            Fitted bucketizer.
        """
        warnings.filterwarnings('ignore')
        
        self.discretizer_ = KBinsDiscretizer(n_bins=self.n_quantiles, encode='onehot-dense')
        self.discretizer_.fit(X)
        return self
    
    def transform(self, X):
        """Perform bucketization using quantiles.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature vectors to be bucketized.
        
        Returns
        -------
        Xt : ndarray
            Transformed data.
        """
        check_is_fitted(self)
        Xt = self.discretizer_.transform(X)
        for x in Xt:
            j = 0
            for n in self.discretizer_.n_bins_:
                saw_one = False
                for _ in range(n):
                    if x[j]:
                        saw_one = True
                    if not saw_one:
                        x[j] = 1
                    j += 1
        return Xt

def load_balance_scale():
    """Load the Balance Scale dataset.
    
    Contains only categorical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Balance+Scale
    """
    names = ["class","left weight","left distance","right weight","right distance"]
    df = pd.read_csv(proj_paths.data.balance_scale, names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_banknote_authentication():
    """Load the Banknote Authentication dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    """
    names = ["variance","skewness","curtosis","entropy","class"]
    df = pd.read_csv(proj_paths.data.banknote, names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_blood_transfusion():
    """Load the Blood Transfusion dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
    """
    names = ["recency","frequency","monetary","time","class"]
    df = pd.read_csv(proj_paths.data.transfusion, header=0, names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_climate_model_crashes():
    """Load the Climate Model Crashes dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes
    """
    df = pd.read_csv(proj_paths.data.pop_failures, delim_whitespace=True)
    y = df["outcome"]
    X = df.drop(columns=["Study","Run","outcome"])
    return X, y

def load_congressional_voting_records():
    """Load the Congressional Voting Records dataset.
    
    Contains only categorical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
    """
    names = list("a{}".format(j+1) for j in range(17))
    df = pd.read_csv(proj_paths.data.house_votes_84, names=names)
    y = df["a1"]
    X = df.drop(columns="a1")
    return X, y

def load_glass_identification():
    """ Load the Glass Identification dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/glass+identification
    """
    df = pd.read_csv(proj_paths.data.glass,
            names=["Id number","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type of glass"],
            index_col="Id number")
    y = df["Type of glass"]
    X = df.drop(columns="Type of glass")
    return X, y

def load_hayes_roth():
    """ Load the Hayes-Roth dataset.
    
    Contains only categorical attributes. Dataset already partitions examples
    into train and test sets.
    
    https://archive.ics.uci.edu/ml/datasets/Hayes-Roth
    """
    df = pd.read_csv(proj_paths.data.hayes_roth_test,
            names=["name","hobby","age","educational level","marital status","class"],
            index_col="name")
    y_train = df["class"]
    X_train = df.drop(columns="class")
    df = pd.read_csv(proj_paths.data.hayes_roth_test,
            names=["hobby","age","educational level","marital status","class"])
    y_test = df["class"]
    X_test = df.drop(columns="class")
    return X_train, X_test, y_train, y_test

def load_image_segmentation():
    """ Load the Image Segmentation dataset.
    
    Contains only numerical attributes. Dataset already partitions examples
    into train and test sets.
    
    http://archive.ics.uci.edu/ml/datasets/image+segmentation
    """
    names = ["class"] + list("a{}".format(j+1) for j in range(19))
    df = pd.read_csv(proj_paths.data.segmentation_train, skiprows=5, names=names)
    y_train = df["class"]
    X_train = df.drop(columns="class")
    df = pd.read_csv(proj_paths.data.segmentation_test, skiprows=5, names=names)
    y_test = df["class"]
    X_test = df.drop(columns="class")
    return X_train, X_test, y_train, y_test

def load_ionosphere():
    """ Load the Ionosphere dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/ionosphere
    """
    names = list("a{}".format(j+1) for j in range(34)) + ["class"]
    df = pd.read_csv(proj_paths.data.ionosphere, names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_parkinsons():
    """ Load the Parkinsons dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/parkinsons
    """
    df = pd.read_csv(proj_paths.data.parkinsons, index_col="name")
    y = df["status"]
    X = df.drop(columns="status")
    return X, y

def load_soybean_small():
    """ Load the Soybean (Small) dataset.
    
    Contains only categorical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Soybean+%28Small%29
    """
    names = list("a{}".format(j+1) for j in range(35)) + ["class"]
    df = pd.read_csv(proj_paths.data.soybean, names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_tictactoe_endgame():
    """ Load the Tic-Tac-Toe Endgame dataset.
    
    Contains only categorical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
    """
    names = list("a{}".format(j+1) for j in range(9)) + ["class"]
    df = pd.read_csv(proj_paths.data.tic_tac_toe, names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y
