import math

import tabula
import camelot
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
import datetime
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from keras.models import load_model
from copy import deepcopy


pdf_path1 = "C:\Desktop\study\python_project\crawler\sample_pdf.pdf"
pdf_path2 = "C:\Desktop\study\python_project\crawler\sample_pdf2.pdf"
sp_wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#
# def load_data(stock_ticker_symbol="^GSPC"):
#     """
#     load, organize and clean the S&P500 data for our model to use
#     :return: sp500 data frame from the ticker object with the needed information
#     """
#     stock_ticker = yf.Ticker(stock_ticker_symbol) #GSPC symbol is the S&P500 index
#     stock_dataframe = stock_ticker.history(period="max")
#     stock_dataframe = stock_dataframe.loc["2000-01-01":].copy() # get only the stocks from year 2000 and above
#     # print(sp500)
#     # sp500 = sp500.index #the date coulm will help us slice the data easily
#     stock_dataframe.plot.line(y="Close", use_index=True) # x - dates, y - close price
#     # plt.show()
#     del stock_dataframe["Dividends"]
#     del stock_dataframe["Stock Splits"]
#
#     # we will predict if the stock will increase its price rather then the price itself cuz thats what important for us
#     stock_dataframe["Tomorrow"] = stock_dataframe["Close"].shift(-1)  # creating new column by shifting all the Close column one day
#     # i.g if the date is 1/1/1990 then  its "tomorrow" price its 2/1/1990 "Close" (today) price
#
#     # base ot "Tomorrow" price we can try to predict, meaning set a target for our ML
#     stock_dataframe["Target"] = (stock_dataframe["Close"] < stock_dataframe["Tomorrow"]).astype(int)
#     # print(sp500)
#     return stock_dataframe
#
#
# def back_test(data, model, predictors, start=2500, step=250):
#     """
#
#     :param data: the dataframe of the company
#     :param model: ML model
#     :param predictors:the model predictors
#     :param start: 2500 by default, every trading year is about 250 days so train the model with "10 years" of data
#     :param step: 250 by default train for "each trading year"
#     :return:
#     """
#     all_predictions = []  # list of df, each entry is a prediction for a single year
#     for i in range(start, data.shape[0], step):
#         # train set is all the yeas before current year, test is the current year
#         train, test = data.iloc[0:i].copy(), data.iloc[i:(i+step)].copy()
#         predictions = predict(train, test, predictors, model)
#         all_predictions.append(predictions)
#     return pd.concat(all_predictions)  # take a list of dfs, and combine to single df
class Util:
    def __init__(self, stock_ticker_symbol="^GSPC", flag=False):
        self.stock_ticker = yf.Ticker(stock_ticker_symbol).history(period="max")
        if not flag:
            d, x, y = self.load_data()
            self._stock_dataframe = np.concatenate(d, x, y)
            self._dates = d
            self._data_table = x
            self._labels = y


    def load_data(self):
        stock_dataframe = self.stock_ticker.history(period="max")
        self._stock_dataframe = stock_dataframe.loc["2000-01-01":].copy()  # get only the stocks from year 2000 and above
        del stock_dataframe["Open"]
        del stock_dataframe["Volume"]
        del stock_dataframe["Dividends"]
        del stock_dataframe["Stock Splits"]
        prev_trade_day = str(stock_dataframe.reset_index()["Date"].iloc[-1])[:10]
        wind = self._df_to_windowed_df(stock_dataframe, "2010-01-07", prev_trade_day)
        # # important. if ill put 2000-01-01 then it wont be able to generate the previoud days data so need to give him
        # so later day to start
        d, x, y = self.windowed_df_to_date_x_y(wind)
        return d, x, y

    def _str_to_datetime(self, string):
        # string = string[:-15]
        split = string.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    def _df_to_windowed_df(self, dataframe, first_date_str, last_date_str, n=3):
        """
        step 1 of converting the data into supervised learning problem where
        index = data
        target = the close value of the stock in that day
        tatget - i = the close value of the target i days ago
        :param dataframe: the df with only dates and close balue
        :param first_date_str: the first day we want to extract info
        :param sec_date_str: the last day we want to extract info
        :param n: how many days before we look at
        :return: a dataframe with column named "Target Date     Target-3     Target-2     Target-1       Target"
        """
        first_date = self._str_to_datetime(first_date_str)
        last_date = self._str_to_datetime(last_date_str)

        target_date = first_date
        dates = []
        X, Y = [], []
        W = []
        last_time = False

        while True:
            z = dataframe.loc[:target_date]
            df_subset = z.tail(n + 1)
            values = df_subset['Close'].to_numpy()
            # adding a column named "Close Price Dist from High Low Average" which represent what the distance betwwen
            # the closing stock price of a day to the avarge bettwen the Highest and Lowest qutar price of the stock
            # positive ints represnet higher closing price then the avarege and negetive lower closing price
            # remark we taking the [3:4][0] entry cuz we need the data only for the "z" date and nots its tail for that
            high_val, low_val = df_subset['High'][3:4][0], df_subset["Low"][3:4][0]
            w = ((values[3:4][0]) - ((high_val + low_val) / 2)).astype(int)
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(x)
            Y.append(y)
            W.append(w)

            next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
            next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
            next_date_str = next_datetime_str.split('T')[0]
            year_month_day = next_date_str.split('-')
            year, month, day = year_month_day
            next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

            if last_time:
                break

            target_date = next_date

            if target_date == last_date:
                last_time = True

        ret_df = pd.DataFrame({})
        ret_df['Target Date'] = dates

        X = np.array(X)
        for i in range(0, n):
            # X[:, i]
            ret_df[f'Target-{n - i}'] = X[:, i]
            ret_df[f'Close Price Dist from High Low Average'] = W[:]

        ret_df['Target'] = Y
        ret_df = ret_df.loc[:, ["Target Date", "Target-3", "Target-2", "Target-1", "Target",
                                "Close Price Dist from High Low Average"]]

        # RSI is an oscillation indicator to measure stock's momentum which is bith the speed and size if price changes
        # good to detrmine if the stock is overbought or  oversold
        # good to under if to buy or sell a stock
        ret_df["RSI"] = ta.rsi(ret_df["Target"])  # need to drop na
        # 3 indicator for EMA which is exponantial moving average which give more wight for recent price move
        # we add fast, medium and slow moving average columns
        ret_df["Fast EMA"] = ta.ema(ret_df["Target"], length=20)
        ret_df["Med EMA"] = ta.ema(ret_df["Target"], length=100)
        ret_df["Slow EMA"] = ta.ema(ret_df["Target"], length=150)
        ret_df[f'Close Price Dist from High Low Average'].shift(2)

        ret_df["Bi Target"] = (ret_df["Target"] > ret_df["Target-1"]).astype(int)
        ret_df.dropna(inplace=True)

        return ret_df

    def windowed_df_to_date_x_y(self, windowed_dataframe):
        """
        step 2 of converting the data into supervised learning problem where
        :param: the windowed_dataframe from "df_to_windowed_df" func
        :return: date: the date of the stocks
                 X: data metric with column target - i
                 y: output vector co-respond to the real data aka target
        """
        df_as_np = windowed_dataframe.to_numpy()
        dates = df_as_np[:, 0]  # getting all the dates from the column
        mid_matrix = df_as_np[:, 1: -1].astype(np.float32)  # getting the target -i columns

        # # normalize and scaling the data fro training
        # mean = mid_matrix.mean(axis=0)
        # mid_matrix -= mean
        # std = mid_matrix.std(axis=0)
        # mid_matrix /= std
        # sc = MinMaxScaler(feature_range=(0, 1)) # scaling all the data to range [0,1]
        # mid_matrix = sc.fit_transform(mid_matrix)
        # print("noralize and scalred matrix is: ")
        # print(mid_matrix)

        X_matrix = mid_matrix.reshape(
            (len(dates), mid_matrix.shape[1], 1))  # reshape the matrix to make the 1d input for the LSTM
        y_output = df_as_np[:, -1]
        return dates, X_matrix.astype(np.float32), y_output.astype(np.float32)

    def split_lstm_data(self, dates, X_matrix, y_output):
        """
        spliting the data into train, validation and test set
        :param dates: the dates np list of the trading stock
        :param X_matrix: the target -i matrix in its correct form
        :param y_output: the output target vector
        :return:
        """

        train_cutoff = int(len(dates) * .8)
        test_cutoff = int(len(dates) * .9)
        dates_train, X_train, y_train = dates[:train_cutoff], X_matrix[:train_cutoff], y_output[:train_cutoff]
        dates_val, X_val, y_val = dates[train_cutoff:test_cutoff], X_matrix[train_cutoff:test_cutoff], y_output[
                                                                                                       train_cutoff:test_cutoff]
        dates_test, X_test, y_test = dates[test_cutoff:], X_matrix[test_cutoff:], y_output[test_cutoff:]
        return dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test


class LSTM:

    def __init__(self, X_train):
        self.feature_dim = X_train.shape[1]
        self.input_dim = X_train.shape[0]
        self.model = Sequential()
        self.model.add(layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(self.feature_dim, 1)))
        self.model.add(layers.LSTM(32, activation='tanh'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])

    def train_model(self, X_val, y_val, epochs=100):
        """
         training the lstm model
         :param dates_train: the dates of the train dataset
         :param X_train: train dataset
         :param y_train: train dataset real labels
         :param dates_val: The dates of the validation dataset
         :param X_val: validation set
         :param y_val: validation set real labels
         :return:
         """
        self.model.fit(x= self.X_train, y=self.y_train, validation_data=(X_val, y_val), epochs=100)
        result = self.save_lstm_model(self.model)
        return result

    # def model_data(self):
    #     stock_dataframe = self.stock_ticker.history(period="max")
    #     self.stock_dataframe = self.stock_dataframe.loc[
    #                            "2000-01-01":].copy()  # get only the stocks from year 2000 and above
    #     del stock_dataframe["Open"]
    #     del stock_dataframe["Volume"]
    #     del stock_dataframe["Dividends"]
    #     del stock_dataframe["Stock Splits"]
    #     prev_trade_day = str(stock_dataframe.reset_index()["Date"].iloc[-1])[:10]
    #     wind = self._df_to_windowed_df(stock_dataframe, "2010-01-07", prev_trade_day)
    #     # # important. if ill put 2000-01-01 then it wont be able to generate the previoud days data so need to give him
    #     # so later day to start
    #     d, x, y = self._windowed_df_to_date_x_y(wind)
    #     return d, x, y


    #
    # def train_model(self, dates_train, X_train, y_train, dates_val, X_val, y_val):
    #     """
    #     training the lstm model
    #     :param dates_train: the dates of the train dataset
    #     :param X_train: train dataset
    #     :param y_train: train dataset real labels
    #     :param dates_val: The dates of the validation dataset
    #     :param X_val: validation set
    #     :param y_val: validation set real labels
    #     :return:
    #     """
    #     print("a-------------------------------a \n", X_train)
    #     features = X_train.shape[1]
    #     input_dim = X_train.shape[0]
    #     model = Sequential()
    #     model.add(layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(features, 1)))
    #     model.add(layers.LSTM(32, activation='tanh'))
    #     model.add(layers.Dense(1, activation='sigmoid'))
    #     model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
    #     model.fit(x= X_train, y=y_train, validation_data=(X_val, y_val), epochs=100)
    #     result = self.save_lstm_model(model)


    def validate(self ,X_val, y_val):
        """

        :param model:
        :param dates_val:
        :param X_val:
        :param y_val:
        :return: the val prediction and its accuracy
        """

        val_predictions = self.model.predict(X_val).flatten().round()
        val_acc = accuracy_score(y_val, val_predictions)
        return val_acc, val_predictions

    def predict(self, X_test):
        """
        predict if a stock price of a model based company trained for will increase its price
        :param model: the train model (for a specific company)
        :param dates_train: the dates of the train dataset
        :param X_set: train dataset
        :return: prediction list, 1 == increase price, 0 == else
        """
        lstm_model = self.model
        prediction = lstm_model.predict(X_test).flatten().round()
        return prediction


    def save_lstm_model(self):
        """
        saving the trained model
        :param model: the trained model
        :return: 1 if its a new model to save and mange to save it, 0 otherwise
        """
        #TODO
        # should recive as input the name of the model and then i can save diffrent model with diffrent names
        if os.path.isfile("C:\Desktop\study\python_project\crawler\lstm_model_MSFT.h5") is False:
            self.model.save("C:\Desktop\study\python_project\crawler\lstm_model_MSFT.h5")
            return 1
        return 0

    def load_lstm_model(self):
        model = load_model("C:\Desktop\study\python_project\crawler\lstm_model_MSFT.h5")
        return model

    def type_of_conclusion(self, y_real, y_predict):
        """
        helper function for the score helper function
        :param y_real: the correct labels
        :param y_predict:  the predicted labels
        :return: number of True negative(TN), False positive(FP), True positive(TP), False negative(FN)
        """
        TN, FP, TP, FN = 0, 0, 0, 0
        for i in range(len(y_real)):
            # if y_predict[i] == 0: FP += 1
            # the label can be only  +-1, so if one of them is 0, i decided it will be false positive cuz its to hard to  abel it as true
            if y_real[i] == y_predict[i] and y_predict[i] == 0: TN += 1
            elif y_real[i] == y_predict[i] and y_predict[i] == 1: TP += 1
            elif y_real[i] != y_predict[i] and y_predict[i] == 0: FN += 1
            elif y_real[i] != y_predict[i] and y_predict[i] == 1: FP += 1

        return TN, FP, TP, FN

    def score_helper(self, y_real, y_predict):
        """
        calculate the error accuracy, false positive rate, true positive rate, precision and specificity
        :param y_real: the correct labels
        :param y_predict: the predicted labels
        :return: score dict
        """
        TN, FP, TP, FN = self.type_of_conclusion(y_real, y_predict)
        print(f"True negative(TN): {TN}, False positive(FP): {FP}, True positive(TP): {TP} ,False negative(FN): {FN} ")
        P = np.count_nonzero(y_real == 1)  # number of positive in the real label
        N = np.count_nonzero(y_real == 0)  # number of negative in the real label
        error_rate = (FP + FN) / (P + N)
        accuracy = (TP + TN) / (P + N)
        precision = TP / (TP + FP)
        specificity = TN / N
        return{"num samples": len(y_real), "error": error_rate, "accuracy": accuracy,
               "FPR": FP / N, "TPR": TP / P, "precision": precision, "specificity": specificity}

    def plot_confusion_matrix(self, y_test, y_predict):
        """
        plot the confusion matrix of a model
        :param y_test: the real labels
        :param y_predict: the predicted labels
        """
        conf_matrix = confusion_matrix(y_test, y_predict)
        ax = sns.heatmap(conf_matrix, annot=True, fmt="g")
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        plt.show()

    def plot_roc_curve(self, y_test, y_predict):
        """
        plots a ROC curve given false positive rate(fpr) and true positive rate(tpr) of a model
        :param fpr: false positive rate
        :param tpr:true positive rate
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_predict)
        plt.plot(fpr, tpr, color="orange", label="ROC")
        plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Guessing")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.show()

    def classify_today(self, stock_ticker_symbol="^GSPC"):
        util = Util(stock_ticker_symbol)

        now = datetime.datetime.today()
        three_days_ago = now - datetime.timedelta(days=3)

        stock_ticker = yf.Ticker(stock_ticker_symbol)  # GSPC symbol is the S&P500 index
        stock_dataframe = stock_ticker.history(period="200d")
        today_trade_day = str(stock_dataframe.reset_index()["Date"].iloc[-1])[:10]
        prev_trade_day = str(stock_dataframe.reset_index()["Date"].iloc[-150])[:10]
        del stock_dataframe["Open"]
        del stock_dataframe["Volume"]
        del stock_dataframe["Dividends"]
        del stock_dataframe["Stock Splits"]

        wind = util._df_to_windowed_df(stock_dataframe, prev_trade_day, today_trade_day)

        d, x, y = util.windowed_df_to_date_x_y(wind)
        model = self.load_lstm_model()
        classify = self.predict(x)
        print(classify)

        return classify
if __name__ == '__main__':

    util =Util()
    df = util._stock_dataframe
    d,x,y = util.
    d, x, y = model_data("MSFT")
    dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test = split_lstm_data(d, x, y)
    train_model(dates_train, X_train, y_train, dates_val, X_val, y_val)
    lstm_model = load_lstm_model()
    # lstm_model.summary()
    prediction = predict(lstm_model, X_test)
    score_dict = score_helper(y_test, prediction)
    # # # print(score_dict)
    plot_roc_curve(y_test, prediction)
    # #
    plot_confusion_matrix(y_test, prediction)
    # lstm_model()
    classify_today("MSFT")

#TODO
# after creating the model we need
# 1) create a fun that takes the ticker index name, and corrent date and create the needed data frame with the 9 deatures
# which the model trained for
# 2) use the data frame from "1" to make a prediction for today
# ideas to improve model:
# 2) if we use stock.institutional_holders we can see who hold the stock and maybe if we see that the holder change we can learn something
# from it
# 3) in the ticker object (before returning the data frame in "load_data()" theres a "news" optaion and myabe we can see
# some important words like "increase" or "sued" that suggests that the stock will go up or down

