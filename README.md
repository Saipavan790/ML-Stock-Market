# ML-Stock-Market
This project involves building a Machine learning model to predict the stock prices of Nifty50 stocks. The methodology involves collecting OHLCV data for all Nifty stocks between the period of 2010 and 2025, excluding the years affected by COVID-19.\
Features used in the model are closing price, EMAs, RSI, etc, with a lookback period of 60 days and the event horizon is 30 days. The algorithm used is a Transformer with both encoder and decoder layers. For further understanding of the algorithm, please refer to the paper below.\

The trained model achieved an F1 score of 52% on Validation data. I've used the F1 beta version to give more weight to precision, as it matters for my problem statement.

Paper - [Stock market index prediction using deep Transformer model](https://www.sciencedirect.com/journal/expert-systems-with-applications)




