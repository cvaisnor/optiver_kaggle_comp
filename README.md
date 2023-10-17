# DNN-Final-Project

## Links:

[Competition](https://www.kaggle.com/competitions/optiver-trading-at-the-close)

[GitHub Repository](https://github.com/cvaisnor/DNN-Final-Project)

[Google Slides Presentation](https://docs.google.com/presentation/d/1Xc5F1_NveFi1il3GqHej2aqVmklR_jZU4kx6ZJhiDEM/edit?usp=sharing)

## Repository Structure:
- /images - contains images used in README.md
- /reference_notebooks - contains public notebooks from the competition

Ignored data files:
The below folder should contain the unzipped data files from the competition. The folder is ignored by git to avoid uploading large files to the repository. The folder structure should be as follows:
- /kaggle/input/optiver-trading-at-the-close/
    - /example_test_files
    - /optiver2023
    - train.csv
    - public_timerseries_testing_util.py

## TODO:
- Exploratory Data Analysis
- Preprocessing

Missing Value Handling:
Interpolation: Fill missing values with interpolated data based on nearby points.
Forward Fill or Backward Fill: Fill gaps with the previous (or next) non-null value.
Drop: Simply remove missing values if they are few.
Impute: Use statistical methods (like mean, median) or models to estimate missing values.
Outliers Detection and Treatment:
Visual Inspection: Using plots to detect anomalies.
Statistical Tests: e.g., Z-scores, IQR.
Treatment: Cap, replace, or remove outliers, depending on the context.
Decomposition:
Trend: Removing or accounting for the underlying trend in the data.
Seasonality: Adjusting for recurring patterns or cycles.
Residual: The remainder of the time series after removing trend and seasonality.
Stationarity:
Many models (like ARIMA) require the time series to be stationary.
Differencing: Taking the difference with a previous observation.
Transformation: e.g., logarithm, square root.
Ad Fuller Test: To check for stationarity.
Detrending:
Remove trends from data to make it more stationary.
Common methods include differencing and regression.
Normalization/Standardization:
Min-Max Scaling: Transforms data to range [0, 1].
Z-score Normalization: Mean of 0 and standard deviation of 1.
Feature Engineering:
Lagged Features: Use previous time steps as features.
Rolling Window Statistics: E.g., rolling mean, rolling standard deviation.
Domain-specific Features: Extracted based on domain knowledge.
Handling Unevenly Spaced Observations:
Resampling: Change the frequency of the data (e.g., from daily to monthly).
Aggregating: Summarizing data over a specific interval.
Encoding Cyclical Features:
For time-based features like hour of the day, day of the week, or month of the year, use sin/cos transformations to encode them so the model captures the cyclicity.
Temporal Split:
When splitting data into training and test sets, always ensure it's done temporally. This means that the future is never used to predict the past.
Removing Noise:
Smoothing: Techniques like moving average can help reduce noise.
Wavelet Denoising: Useful for certain types of data.

Characteristics

Trend: Over longer periods, stocks and indices may exhibit upward (bull market) or downward (bear market) trends. A stock might be outperforming (above the trend) or underperforming (below the trend) the synthetic index.
Seasonality: Certain stocks or sectors might exhibit recurring patterns or cycles, such as increased sales around holidays or cyclic industries like housing.
Volatility: Stock prices can be volatile, undergoing rapid and substantial changes. The volatility of individual stocks may be higher than the volatility of the synthetic index due to the diversification effect of the index.
Correlation: The movement of individual stocks might be correlated to the overall movement of the synthetic index. This correlation could be positive (stock moves in the same direction as the index) or negative (stock moves in the opposite direction).
Noise: There's always a certain amount of random noise in stock prices which is not attributed to any known factor or event.
Unexpected Events: Unexpected events (like geopolitical incidents, surprise earnings results, or global pandemics) can affect both individual stocks and the overall index. These shocks can lead to sudden and significant price movements.
Feedback Loops: Sometimes, the movement of a stock or an index can influence further movements in a reinforcing loop. For instance, if many investors begin selling a stock due to fear of a downturn, the price might drop further, inciting even more investors to sell.
Liquidity: Not all stocks have the same level of liquidity. Stocks with higher liquidity tend to have tighter bid-ask spreads and can handle larger trade volumes without significant price movements. Less liquid stocks might exhibit more erratic movements.
Earning Reports & Dividends: Quarterly earnings reports can lead to significant price changes for individual stocks. Similarly, the declaration or payment of dividends can influence stock price movements.
Economic Indicators: Broader economic data, like interest rate changes, unemployment rates, or GDP growth, can influence both individual stock prices and the overall direction of the synthetic index.
Sector-Specific Movements: Sometimes, entire sectors move in a specific direction irrespective of the broader market. For example, if there's a technological breakthrough in renewable energy, it might affect all stocks in that sector.
Lagged Reactions: Often, stocks or the broader market might not react instantaneously to news or events. There might be delayed reactions as investors process the information.

## Target CSV
![Target Submission Format](images/target_format.png)

## Score Slide as of 10/10/23:
![Score Slide](images/score_slide.png)
