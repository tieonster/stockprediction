# Stock prediction LSTM Model
Evaluates the accuracy of LSTM model in stock prediction of tataglobal 

Using the idea of deep learning and RNNs, I have created a stock prediction of tataglobal with heavy reference from the following source:
https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/

# However, note that this model is limited, in the sense that the model would require the actual stock prices as x input before being able to "predict" prices.

For example:

Model is given data from day 1 until day 100.
The LSTM network picks a block of say the first 30 days, to predict the stock price on the 31st day.
Then, it moves forward by 1 day, and uses a block from day 2-31 to predict the price on the 32nd day.
Then, it moves forward by another 1 day, and uses a block from day 3-32 to predict the price on the 33rd day
So on and so forth.

In theory, it predicts the stock price of the following day only, will release a possible update on maybe being able to predict the stock price of the following week more accurately.


This would not have been possible without the help of the following blog post:
https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/

For the full story, please refer to the website above

