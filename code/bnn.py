import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import math

from keras.callbacks import EarlyStopping

import time

# TODO: Remove this?
# # Load the data
# data = pd.read_csv('snowfall_data.csv')

# # Split the data into training and test sets
# train_data = data.sample(frac=0.8, random_state=0)
# test_data = data.drop(train_data.index)

# # Define the input and output variables
# train_input = train_data[['MaxTemp', 'MinTemp', 'Precipitation']]
# train_output = train_data['Snowfall']
# test_input = test_data[['MaxTemp', 'MinTemp', 'Precipitation']]
# test_output = test_data['Snowfall']

# Define the Bayesian neural network model
class BNN():
    def __init__(self, seed=4200):

        self.seed = seed
        tf.keras.utils.set_random_seed(seed)

        self.model = tf.keras.Sequential([
                                        #   tf.keras.layers.Dense(16, activation="relu"),
                                        #   tf.keras.layers.Dense(16, activation="sigmoid"),
                                          tf.keras.layers.Dense(16, activation="tanh"),
                                        #   tf.keras.layers.Dense(16, activation="linear"),
                                        #   tf.keras.layers.Dense(16),
                                          
                                        #   tf.keras.layers.Dropout(0.4),

                                        #   tfp.layers.DenseFlipout(8, activation="relu"), # almost universally bad
                                        #   tfp.layers.DenseFlipout(8, activation="sigmoid"),
                                        #   tfp.layers.DenseFlipout(8, activation="tanh"),
                                        #   tfp.layers.DenseFlipout(8, activation="linear"),
                                        #   tfp.layers.DenseFlipout(8),

                                        #   tf.keras.layers.Dropout(0.2),

                                        #   tf.keras.layers.Dense(8, activation="relu"),
                                        #   tf.keras.layers.Dense(8, activation="sigmoid"),
                                        #   tf.keras.layers.Dense(8, activation="tanh"),
                                        #   tf.keras.layers.Dense(8, activation="linear"),
                                        #   tf.keras.layers.Dense(8),

                                        #   tf.keras.layers.Dropout(0.1),

                                        #   tfp.layers.DenseFlipout(2, activation="sigmoid"),

                                        #   tf.keras.layers.Dropout(0.1),

                                        #   tfp.layers.DenseFlipout(8, activation="relu"),
                                        #   tfp.layers.DenseFlipout(8, activation="sigmoid"),
                                          tfp.layers.DenseFlipout(8, activation="tanh"),
                                        #   tfp.layers.DenseFlipout(8, activation="linear"),
                                        #   tfp.layers.DenseFlipout(8),

                                          tf.keras.layers.Dense(1, activation="relu")
                                        #   tf.keras.layers.Dense(1, activation="sigmoid")
                                        #   tf.keras.layers.Dense(1, activation="tanh")
                                        #   tf.keras.layers.Dense(1, activation="linear")
                                        #   tf.keras.layers.Dense(1)
                                        ])
        #,tfp.layers.DenseFlipout(1)
        #self.create(),tf.keras.layers.Dropout(.01)
        self.compile()

    # def create(self):
    #     self.model = 
        
    def compile(self):
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=neg_log_likelihood)

    def train(self,xTrain,yTrain):
        self.model.fit(xTrain, yTrain, epochs=1000, validation_split=.2, callbacks = EarlyStopping(monitor="loss",patience=3) )
        
    def round_small_values(self,predictions):
        new_predictions = predictions.flatten() # []
        # for pred in predictions:
        #     if pred[0] < 0.00:
        #         new_predictions.append(0)
        #     else:
        #         new_predictions.append(pred[0])
        #         # new_predictions.append(0)
        return new_predictions

    def predict(self,xTest,yTest):
        predictions = self.model.predict(xTest)
        assert predictions.shape == (len(yTest), 1), f"Invalid prediction shape returned. Expected ({len(yTest)}, 1). Got {predictions.shape}"
        
        # predictions = np.abs(predictions)
        predictions = self.round_small_values(predictions)

        MAE = np.mean(np.abs(predictions - yTest))
        print("MAE:", MAE)

        MSE = np.square(np.subtract(yTest,predictions)).mean() 
        RMSE = math.sqrt(MSE)
        print("RMSE:", RMSE)

        # 0s RMSE: 1.5504469401710315
        baseline = {4200: np.array([MAE < 0.5239999999999999, RMSE < 1.5220674820747655])}
        if baseline[self.seed].any():
            print("WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN WIN")
            print(baseline[self.seed])

        return predictions,MAE,RMSE
    
    def plotPredictions(self,yPred,yTest,dates,file,MAE,RMSE,type=2,save=False):
        t = time.localtime()
        current_time = time.strftime("%Hh_%Mm_%Ss", t)

        yMax = max([max(yTest), max(yPred)])

        if type != 1:
            # xRange = np.arange(0,yTest.shape[0])
            plt.scatter(dates, yTest.flatten(), color='blue', s=5, label='Truth')
            plt.scatter(dates, yPred, color='red', s=5, label='Prediction')

            plt.text(list(dates)[0], yMax-0.5, f"Errors\nMAE: {round(MAE, 3)}\nRMSE: {round(RMSE, 3)}", fontsize=7)
            plt.xlabel("Timeline")
            plt.ylabel("Snowfall")
            plt.title(f"{file[8:-4]} Predicted vs Actual Snowfall")
            plt.legend()

            if save:
                plt.savefig(f"../figures/snow/{file[8:-4]}_time_{current_time}.png")
            else:
                plt.show()

        if type != 0:
            truth = np.arange(0,yMax+1)
            
            plt.plot(truth, truth, color='blue', label='Equal')
            plt.scatter(yTest, yPred, color='red', s=5, label='Comparison')

            plt.text(0, yMax-1.7, f"Errors\nMAE: {round(MAE, 3)}\nRMSE: {round(RMSE, 3)}", fontsize=7)
            plt.xlabel("Truth Snowfall")
            plt.ylabel("Prediction Snowfall")
            plt.title(f"{file[8:-4]} Predicted vs Actual Proximity")
            plt.legend()

            if save:
                plt.savefig(f"../figures/comp/{file[8:-4]}_direct_{current_time}.png")
            else:
                plt.show()


# Define the log likelihood function
def neg_log_likelihood(y_true, y_pred):
    return -tfp.distributions.Normal(loc=y_pred, scale=1.0).log_prob(y_true)
