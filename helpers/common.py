import pandas as pd
import matplotlib.pyplot as plt

def load_data(kaggle_environment=False):
    '''Loads and returns the data files.'''
    if kaggle_environment:
        data_path = '/kaggle/input/'
    else:
        data_path = 'kaggle/input/'

    train = pd.read_csv(data_path + 'optiver-trading-at-the-close/train.csv')
    test = pd.read_csv(data_path + 'optiver-trading-at-the-close/example_test_files/test.csv')
    revealed_targets = pd.read_csv(data_path + 'optiver-trading-at-the-close/example_test_files/revealed_targets.csv')
    sample_submission = pd.read_csv(data_path + 'optiver-trading-at-the-close/example_test_files/sample_submission.csv')

    return train, test, revealed_targets, sample_submission

def submit_dummy_predictions(env, predictions):
    '''Submits a dummy prediction to the Optiver API. Use when you have an error during processing
    of iter_test and you do not want to re-run the whole notebook to figure out if you fixed it.'''
    predictions["target"] = [0 for i in range(predictions.shape[0])]
    env.predict(predictions)

def plot_predictions(actuals, predicted):
    '''Plots actual vs predicted values to get a sense of whether the predictions make sense.'''

    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
