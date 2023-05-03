import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from keras.callbacks import EarlyStopping
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
    def __init__(self) :
        self.model = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu'),tfp.layers.DenseFlipout(1)])
        #self.create()
        self.compile()

    # def create(self):
    #     self.model = 
        
    def compile(self):
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=neg_log_likelihood)

    def train(self,xTrain,yTrain):
        self.model.fit(xTrain, yTrain, epochs=1000, validation_split=.2, callbacks = EarlyStopping(monitor="loss",patience=2) )
        
    def predict(self,xTest,yTest):
        predictions=  self.model.predict(xTest)
        mae = np.mean(np.abs(predictions - yTest))
        print("Mean absolute error:", mae)
        return predictions,mae
    def plotPredictions(self,yPred,yTest):
        xRange = np.arange(0,yTest.shape[0])
        # print(yTest.shape,yPred.shape, len(xRange))
        # print(yTest.flatten().shape, yPred.flatten().shape, xRange.shape)
        plt.scatter(xRange, yTest.flatten(), color = 'blue',label='Truth')
        plt.scatter(xRange, yPred.flatten(), color ='red',label='Pred')
        plt.legend()
        plt.show()


# Define the log likelihood function
def neg_log_likelihood(y_true, y_pred):
    return -tfp.distributions.Normal(loc=y_pred, scale=1.0).log_prob(y_true)
