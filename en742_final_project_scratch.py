import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

kaggle_environment = True # True if running on Kaggle, don't forget to add the dataset!

if kaggle_environment:
    data_path = '/kaggle/input/'
else:
    data_path = 'kaggle/input/'

for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# change paths if running locally!
try:
    original_train = pd.read_csv(data_path + 'optiver-trading-at-the-close/train.csv')
except:
    original_train = pd.read_csv(r'C:\Users\marko\OneDrive\Documents\MSGIS_Assignments\Sixth_Semester\EN742_Neural_Networks\EN742_FINAL_PROJECT\train.csv')
# revealed_targets = pd.read_csv(data_path + 'optiver-trading-at-the-close/example_test_files/revealed_targets.csv')
# test = pd.read_csv(data_path + 'optiver-trading-at-the-close/example_test_files/test.csv')
# sample_submission = pd.read_csv(data_path + 'optiver-trading-at-the-close/example_test_files/sample_submission.csv')

# split_ratio = 0.8  # 80% for training, 20% for testing
# split_idx = int(len(original_train) * split_ratio)

train = original_train.copy()#.iloc[:split_idx]
# test = original_train.iloc[split_idx:]

# y_test = test['target'].values

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler, MaxAbsScaler, PolynomialFeatures, StandardScaler, PowerTransformer, MinMaxScaler


class LogFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for column in self.columns:
            X[f'{column}_log'] = np.log1p(X[column].clip(lower=0.00001))
        return X


class WapLagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, shift_sizes):
        self.shift_sizes = shift_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for shift_size in self.shift_sizes:
            X[f'wap_lag{shift_size}'] = X.groupby('stock_id')['wap'].shift(shift_size)
        return X


class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features, shift_sizes):
        self.features = features
        self.shift_sizes = shift_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for i in range(len(self.features)):
            feature = self.features[i]
            shift_sizes = self.shift_sizes[i]
            for shift_size in self.shift_sizes:
                X[f'{feature}_lag_{shift_size}'] = X.groupby('stock_id')[feature].shift(shift_size)
            return X


class WapRollingMeanFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for window_size in self.window_sizes:
            X[f'wap_rolling_mean{window_size}'] = X.groupby('stock_id')['wap'].rolling(
                window=window_size).mean().reset_index(level=0, drop=True)
        return X


class WapDiffFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['wap_diff'] = X.groupby('stock_id')['wap'].diff()
        return X


class WapExpandingMeanFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['wap_expanding_mean'] = X.groupby('stock_id')['wap'].expanding().mean().reset_index(level=0, drop=True)
        return X


class median_or_std_fill(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        columns = [c for c in list(X.columns) if np.sum(X[c].isnull()) > 0]
        for col in columns:
            median = X[col].median()
            std = X[col].std()
            # generate random number within std of current variation and divide by some noise
            try:
                rand_fill_num = np.random.randint(median - std, median + std) / (np.random.randint(101, 199) / 100)
            except:
                rand_fill_num = 0.0
            return X.fillna(rand_fill_num)


class ForwardFillValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.fillna(method='ffill', inplace=True)
        return X


class FillZero(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.fillna(0)
        return X


class DataFrameWrapper(BaseEstimator, TransformerMixin):
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


# Column Preprocessor
columns_to_keep = [
    'imbalance_size_log', 'matched_size_log',  # 'stock_id',
    'imbalance_buy_sell_flag', 'reference_price',
    'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap', 'wap_lag_1',
    'wap_lag_5',  # 'wap_lag10', 'wap_lag15', 'wap_lag20',
    'wap_rolling_mean2', 'wap_rolling_mean3',
    'wap_rolling_mean5', 'wap_diff',
    'wap_expanding_mean', 'seconds_in_bucket',
    'matched_size_lag_1', 'matched_size_lag_3', 'matched_size_lag_5'
]
one_hot_cols = []
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), one_hot_cols),
        ('passthrough', 'passthrough', columns_to_keep)
    ],
    remainder='drop'  # Drop other columns
)

# Main Pipeline
pipeline = Pipeline([
    ('median_or_std_fill', median_or_std_fill()),
    ('logs', LogFeatures(['imbalance_size', 'matched_size'])),
    ('wap_lags', LagFeatures(['wap'], [1, 5])),
    ('matched_size_lags', LagFeatures(['matched_size'], [1, 3, 5])),
    ('wap_rolling_means', WapRollingMeanFeatures([2, 3, 5])),
    ('wap_diff', WapDiffFeature()),
    ('wap_expanding_mean', WapExpandingMeanFeature()),
    # ('forward_fill', ForwardFillValues()),
    # ('fill_zero', FillZero()),
    ('preprocessor', DataFrameWrapper(preprocessor, columns_to_keep)),
    # ('yeo_johnson', PowerTransformer()),
    # ('min_max_scalar', MinMaxScaler()),
    # ('std_scalar', StandardScaler()),
    # ('scaler', DataFrameWrapper(StandardScaler())),
    # ('poly', PolynomialFeaturesWrapper(2)),
    ('median_or_std_fill_again', median_or_std_fill()),
    # ('abs_max_scalar', MaxAbsScaler())
    # ('alleged_robust_scalar', RobustScaler())
    ('yeo_johnson', PowerTransformer()),
    # ('standard_scalar', StandardScaler())
    ('quantile_transformer', QuantileTransformer(output_distribution='normal'))
    # ('alleged_robust_scalar', RobustScaler(unit_variance=True))
])

x_fields = [c for c in list(train.columns) if c != 'target']
train_transformed = pipeline.fit_transform(train[x_fields])
# test_transformed = pipeline.transform(test)

X = pd.DataFrame(train_transformed)
# X_test = test_transformed


def generate_features(cumulative_test_df, current_test, pipeline):
    transformed_df = pipeline.transform(cumulative_test_df)
    # cumulative_test_df['wap_lag1'] = cumulative_test_df.groupby('stock_id')['wap'].shift(1)
    # cumulative_test_df['wap_lag5'] = cumulative_test_df.groupby('stock_id')['wap'].shift(5)
    # cumulative_test_df['wap_rolling_mean10'] = cumulative_test_df.groupby('stock_id')['wap'].rolling(window=10).mean().reset_index(level=0, drop=True)
    # cumulative_test_df['wap_diff'] = cumulative_test_df.groupby('stock_id')['wap'].diff()
    # cumulative_test_df['wap_expanding_mean'] = cumulative_test_df.groupby('stock_id')['wap'].expanding().mean().reset_index(level=0, drop=True)
    # cumulative_test_df.fillna(method='ffill', inplace=True)
    # cumulative_test_df = cumulative_test_df.drop(columns=['row_id'])
    # cumulative_test_df = cumulative_test_df.fillna(0)

    # Only return rows corresponding to the current test dataframe
    return pd.DataFrame(transformed_df).iloc[-len(current_test):]


# X_train = X
# y_train = y

# import lightgbm as lgb
# # from sklearn.pipeline import Pipeline
#
#
# # lgbm = lgb.LGBMRegressor(n_jobs=-1, random_state=0, force_col_wise=True,
# #                          verbose=-1, boosting_type='gbdt', num_leaves=10,
# #                          reg_alpha=0, reg_lambda=0.2, objective='regression_l1')
#
# lgbm = lgb.LGBMRegressor(n_jobs=-1, random_state=0, objective='regression_l1')
# lgbm.fit(X_train, y_train)
#
# # lgbm.score(X_test, y_test)
# std_scalar = StandardScaler()
#
# scaled_X = pd.DataFrame(std_scalar.fit_transform(X), columns=x_fields)


def y_transformer(in_ndarray, min_val_from_X, max_val_from_X, inverse=False):
    # used to normalize values in 'target' field, target field is also already gaussian distribution
    in_pd_series = pd.Series(in_ndarray)

    max_num = in_pd_series.max()
    min_num = in_pd_series.min()
    median = in_pd_series.median()

    def if_else_function(row):
        if row < median:
            x = min_num / min_val_from_X
        else:
            x = max_num / max_val_from_X
        if inverse:
            return row * x
        else:
            return row / x

    return pd.Series(np.vectorize(if_else_function)(in_pd_series))


y = train['target'].values
y_transformed = y_transformer(y, X.min().min(), X.max().max())
inverse = y_transformer(y_transformed, X.min().min(), X.max().max(), inverse=True)

historical_data = pd.DataFrame(pd.concat([X, pd.DataFrame(y_transformed,  columns=['target'])], axis=1),
                               columns=list(X.columns) + ['target'])

total_timesteps, num_features = historical_data.shape
num_training_sequences = 3500


def generate_sequences(data, sequence_size, prediction_size, step_size, num_sequences):
    past_sequences = []
    future_prices = []
    total_possible_sequences = (len(data) - prediction_size - sequence_size) // step_size
    starting_sequence = total_possible_sequences - num_sequences

    print (f'Data length: {len(data)}, starting_sequence: {starting_sequence}')

    for i in range(starting_sequence, total_possible_sequences):
        start_index = i * step_size
        end_index = start_index + sequence_size
        prediction_end_index = end_index + prediction_size

        if prediction_end_index < len(data):
          past_sequence = data.iloc[start_index:end_index, :].values
          future_price_sequence = data.iloc[end_index:prediction_end_index, data.columns.get_loc('target')].values

          past_sequences.append(past_sequence)
          future_prices.append(future_price_sequence)

        else:
            print(f'ERROR: Calculations were incorrect start index {start_index}, end index {end_index}, prediction end index {prediction_end_index}')

    return np.array(past_sequences), np.array(future_prices)


training_set_size = 500
validation_set_size = 500
num_validation_sequences = 50
num_predictions = 200 # change based on iter_test
sliding_window_step_size = 10

validation_start_index = -(validation_set_size + num_predictions + (num_validation_sequences * sliding_window_step_size))
validation_end_index = -num_predictions

validation_data = historical_data.iloc[validation_start_index:]
train_data = historical_data

print('Creating training data')
train_inputs, train_outputs = generate_sequences(
    train_data,
    training_set_size,
    num_predictions,
    sliding_window_step_size,
    num_training_sequences
)

validation_sequences = (validation_set_size - num_predictions) // sliding_window_step_size

print('Creating validation data')
validation_inputs, validation_outputs = generate_sequences(
    validation_data,
    validation_set_size,
    num_predictions,
    sliding_window_step_size,
    num_validation_sequences
)

from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
Init = keras.initializers.GlorotUniform(seed=42)

gru_size = 200
m = Sequential(
    [
        GRU(gru_size, return_sequences=True, input_shape=[None, num_features], name='GRU1', kernel_initializer=Init, recurrent_initializer=Init),
        Dropout(.2, name='d1'),
        GRU(gru_size, name='GRU2', kernel_initializer=Init, recurrent_initializer=Init),
        Dropout(.2, name='d2'),
        Dense(num_predictions, name='out', kernel_initializer=Init)
    ],
    name='RNN_model'
)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

m.summary()

m.compile(optimizer='adam', loss='mean_absolute_error')

validation_data = (validation_inputs, validation_outputs)


# def reverse_scalar(in_df, in_pred, scalar_obj):
#     # used to reverse transform scalar that was used to transform the 'target' column
#     copied_input_df = in_df.copy()
#     columns = list(copied_input_df.columns)
#     if 'target' in columns:
#         copied_input_df.drop(columns=['target'], axis=1, inplace=True)
#     else:
#         columns.append('target')
#
#     pred_df = pd.DataFrame(in_pred, columns=['target'])
#     joined_df = pd.concat([copied_input_df, pred_df], axis=1)
#
#     output = pd.DataFrame(scalar_obj.inverse_transform(joined_df), columns=columns)
#     return output


with tf.device('/CPU:0'):
    callbacks = []
    hist = m.fit(train_inputs, train_outputs, validation_data=validation_data,
        epochs=5, batch_size=16, callbacks=callbacks
    )

    from sklearn.metrics import mean_absolute_error

    # predictions = reverse_scalar(historical_data.iloc[-training_set_size:, :],
    #                              m.predict(historical_data.iloc[-training_set_size:, :].values[np.newaxis,...]).flatten(),
    #                              std_scalar)
    predictions = m.predict(historical_data.iloc[-training_set_size:, :].values[np.newaxis, ...]).flatten()
    # mae = mean_absolute_error(y_test, predictions)
    # print(f"Mean Absolute Error on the test set: {mae:.4f}")

import sys; sys.path.append(r'C:\Users\marko\OneDrive\Documents\MSGIS_Assignments\Sixth_Semester\EN742_Neural_Networks\EN742_FINAL_PROJECT')

import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()

counter = 0

# init 3 empty lists
test_ls, revealed_targets_ls, sample_prediction_ls = [], [], []
cumulative_test_df = pd.DataFrame()

for (test_in, revealed_targets, sample_prediction) in iter_test:
    if 'group_id' in list(test_in.columns):
        test_in.drop(columns=['group_id'], axis=1, inplace=True)
    # Append the dataframe that API return into the list.
    # test_ls.append(test_in.copy())
    # revealed_targets_ls.append(revealed_targets.copy())
    # sample_prediction_ls.append(sample_prediction.copy())

    cumulative_test_df = pd.concat([cumulative_test_df, test_in], axis=0, ignore_index=True)

    # Generate features
    test_transformed = generate_features(cumulative_test_df, test_in, pipeline)
    # test_transformed_scaled = pd.DataFrame(std_scalar.fit_transform(test_transformed), columns=list(test_transformed))

    # Writes our predictions
    # preds = m.predict(test_transformed_scaled.values[np.newaxis, ...]).flatten()
    # sample_prediction["target"] = reverse_scalar(test_transformed_scaled, preds, std_scalar)['target']
    max_num = test_transformed.max().max()
    min_num = test_transformed.min().min()
    pred = m.predict(test_transformed.values[np.newaxis, ...]).flatten()
    sample_prediction["target"] = y_transformer(pred, min_num, max_num, inverse=True)

    # This line submit our predictions.
    env.predict(sample_prediction)
    counter += 1
