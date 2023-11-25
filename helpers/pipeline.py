import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class LogFeatures(BaseEstimator, TransformerMixin):
    '''Transformer for applying logarithmic transformation to specified columns.'''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for column in self.columns:
            X[f'{column}_log'] = np.log1p(X[column].clip(lower=0.00001))
        return X

class LagFeatures(BaseEstimator, TransformerMixin):
    '''Transformer for creating lag features for specified columns and shift sizes.'''
    def __init__(self, features, shift_sizes):
        self.features = features
        self.shift_sizes = shift_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for feature in self.features:
            for shift_size in self.shift_sizes:
                X[f'{feature}_lag_{shift_size}'] = X.groupby('stock_id')[feature].shift(shift_size)
        return X

class RollingMeanFeatures(BaseEstimator, TransformerMixin):
    '''Transformer for calculating rolling mean of each feature over each specified window size.'''
    def __init__(self, features, window_sizes):
        self.features = features
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for feature in self.features:
            for window_size in self.window_sizes:
                X[f'{feature}_rolling_mean{window_size}'] = X.groupby('stock_id')[feature].rolling(window=window_size).mean().reset_index(level=0, drop=True)
        return X

class DiffFeatures(BaseEstimator, TransformerMixin):
    '''Transformer for computing the difference of each feature column (i.e., first discrete difference).'''
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for feature in self.features:
            X[f'{feature}_diff'] = X.groupby('stock_id')[feature].diff()
        return X

class ExpandingMeanFeatures(BaseEstimator, TransformerMixin):
    '''Transformer for calculating the expanding mean of each of the feature columns.'''
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for feature in self.features:
            X[f'{feature}_expanding_mean'] = X.groupby('stock_id')[feature].expanding().mean().reset_index(level=0, drop=True)
        return X

class DropColumns(BaseEstimator, TransformerMixin):
    '''Drops columns from the data frame.'''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        all_columns = X.columns
        drop_columns = []
        for column in self.columns:
            if column in all_columns:
                drop_columns.append(column)
        return X.drop(columns=drop_columns)

class ForwardFillValues(BaseEstimator, TransformerMixin):
    '''Transformer for forward filling NaN values in the DataFrame, grouped by stock_id.'''

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for column in X.columns:
            X[column] = X.groupby('stock_id')[column].transform(lambda group: group.ffill())
        return X


class FillZero(BaseEstimator, TransformerMixin):
    '''Transformer for replacing NaN values with zero in the DataFrame.'''
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.fillna(0)
        return X

class DataFrameWrapper(BaseEstimator, TransformerMixin):
    '''Wrapper for transforming output of a transformer to a DataFrame with optional specified columns.'''
    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        result = self.transformer.transform(X)
        if isinstance(result, pd.DataFrame):
            return result
        if self.columns is None:
            columns = X.columns
        else:
            columns = []
            for column in X.columns:
                if column in self.columns:
                    columns.append(column)
        return pd.DataFrame(result, columns=columns)

class PolynomialFeaturesWrapper(BaseEstimator, TransformerMixin):
    '''Transformer for generating polynomial and interaction features with specified degree.'''
    def __init__(self, degree=2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.new_feature_names = None

    def fit(self, X, y=None):
        self.poly.fit(X)
        self.new_feature_names = self.poly.get_feature_names_out(X.columns)
        return self

    def transform(self, X):
        transformed_array = self.poly.transform(X)
        return pd.DataFrame(transformed_array, columns=self.new_feature_names)

class MissingValueImputer(BaseEstimator, TransformerMixin):
    '''Impute missing values in specified columns of a DataFrame using IterativeImputer.'''
    def __init__(self, columns, max_iter=10, random_state=0):
        self.columns = columns
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        columns_to_impute = [col for col in self.columns if col in X.columns]

        # Select the columns for imputation
        columns_data = X[columns_to_impute]

        # Check if there are any missing values and if the columns are numeric
        if columns_data.isna().any().any() and columns_data.select_dtypes(include='number').shape[1] == len(columns_to_impute):
            imputer = IterativeImputer(max_iter=self.max_iter, random_state=self.random_state)

            # Apply imputation
            imputed_data = imputer.fit_transform(columns_data)
            imputed_df = pd.DataFrame(imputed_data, columns=columns_to_impute, index=X.index)

            # Update the original DataFrame
            X.update(imputed_df)
        return X
