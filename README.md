# Stock-prediction-model-using-XGBoost-algorithm-and-sentiment-analysis-using-Spark.
---

This project proposes a novel method for acquiring intraday stock data from the National Stock Exchange (NSE) and sentiment-laden news articles using web-scraping technology. It leverages Spark’s distributed computing framework for sentiment analysis and XGBoost for stock trend prediction. The integration of these components provides actionable stock recommendations (buy/sell) to investors.

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

This repository contains the implementation code for the paper:
> A. Chennupati, B. Prahas, B. A. Ghali and M. Venugopalan, "Integrative Day Trading Stock Trend Prediction using Web Scraping and Sentiment Analysis," 2024 IEEE 9th International Conference for Convergence in Technology (I2CT), Pune, India, 2024, pp. 1-7, doi: 10.1109/I2CT61223.2024.10543704.



## Features

- **Intraday Stock Data Acquisition**: Fetches real-time stock data from the NSE using web-scraping.
- **Sentiment Analysis**: Analyzes news articles related to a company using Spark’s distributed computing framework to quantify sentiment scores.
- **Stock Trend Prediction**: Predicts future stock prices using the XGBoost algorithm and historical stock data.
- **Cross-Validation**: Evaluates model performance using time series cross-validation.
- **Stock Recommendations**: Combines sentiment scores and XGBoost predictions to provide buy/sell recommendations.
- **Interactive Web Interface**: Built with Streamlit for user-friendly interaction and visualization.

![image](https://github.com/user-attachments/assets/ff8c2280-52b0-4233-9cfe-6fa22263ba1d)

## Requirements

You can install the required libraries using the following command:

```bash
pip install nltk pyspark pandas numpy matplotlib seaborn xgboost streamlit requests bs4 scikit-learn
```

Additionally, you need to download the `vader_lexicon` for sentiment analysis:

```python
import nltk
nltk.download('vader_lexicon')
```

## Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Aadigha-git/Stock_Prediction_using_XGBoost_and_Sentiment_Analysis_using_Spark.git
   ```

2. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

3. **Enter the company name** in the input field and click "Get Stock Identifier" to fetch the stock identifier.

4. **View the stock trend predictions** and sentiment analysis results in the interactive web interface.


## Key Components

### Stock Trend Prediction

- **Data Fetching**: Fetches historical stock data from the NSE India API.
- **Feature Engineering**: Creates time-based features and lag features for the prediction model.
- **Model Training**: Uses XGBoost regression to predict future stock prices.
- **Cross-Validation**: Performs time series cross-validation to evaluate model performance.
  ![image](https://github.com/user-attachments/assets/3d619a89-df0f-4a57-bbbf-e7af7d8b679a)
- **Visualization**: Plots training/test splits, cross-validation results, and future predictions.
 ![image](https://github.com/user-attachments/assets/94cfc6a3-208b-4042-8cef-fa93f40aeb63)


### Sentiment Analysis

- **News Article Scraping**: Fetches news articles related to the company from CNBC TV18.
- **Content Extraction**: Extracts relevant content from the articles.
- **Sentiment Scoring**: Uses the VADER sentiment analysis tool to score the sentiment of the extracted content.
- **Decision Making**: Provides a buy/sell recommendation based on the mean sentiment score.

## Research Contribution

This research proposes a new method of acquiring intraday stock data from the National Stock Exchange and sentiment-laden news articles via web-scraping technology. Spark’s distributed computing framework is employed to perform sentiment analysis and quantify sentiment scores for the chosen stock over the past month. Concurrently, the XGBoost algorithm processes the historical stock data to predict future values along with the cross-validation process. By integrating XGBoost predictions and sentiment scores, this model offers stock recommendations to the user to either buy or sell a certain stock.

The proposed model's performance is evaluated for future value predictions based on the average RMSE score, reported as **1.2479**. This systematic approach combines financial prediction, natural language processing, and machine learning to provide actionable recommendations for investors.

![image](https://github.com/user-attachments/assets/1bf61d74-a492-4007-8de1-53c7f476e0d9)


## Example

1. Enter the company name (e.g., "Tata Motors") in the input field.
2. Click "Get Stock Identifier" to fetch the stock identifier.
3. View the historical stock price data and predictions.
4. Analyze the sentiment of news articles related to the company.
5. Receive a buy/sell recommendation based on the sentiment analysis and XGBoost predictions.


Feel free to explore the project and provide feedback!
