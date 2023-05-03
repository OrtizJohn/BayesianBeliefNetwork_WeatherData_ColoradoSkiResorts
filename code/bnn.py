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
        tf.keras.utils.set_random_seed(4200)
        self.model = tf.keras.Sequential([tf.keras.layers.Dense(16, activation=tf.nn.relu),tfp.layers.DenseFlipout(8, activation="sigmoid"),tfp.layers.DenseFlipout(1)])#,tfp.layers.DenseFlipout(1)
        #self.create()tf.keras.layers.Dropout(.1),
        self.compile()

    # def create(self):
    #     self.model = 
        
    def compile(self):
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=neg_log_likelihood)

    def train(self,xTrain,yTrain):
        self.model.fit(xTrain, yTrain, epochs=1000, validation_split=.2, callbacks = EarlyStopping(monitor="loss",patience=2) )
        
    def round_small_values(self,predictions):
        new_predictions = []
        for pred in predictions:
            if pred[0] < 0.00:
                new_predictions.append(0)
            else:
                new_predictions.append(pred[0])
        return new_predictions

    def predict(self,xTest,yTest):
        predictions = self.model.predict(xTest)
        assert predictions.shape == (len(yTest), 1), f"Invalid prediction shape returned. Expected ({len(yTest)}, 1). Got {predictions.shape}"
        
        predictions = np.abs(predictions)
        predictions = self.round_small_values(predictions)
        mae = np.mean(np.abs(predictions - yTest))
        print("Mean absolute error:", mae)
        return predictions,mae
    
    def plotPredictions(self,yPred,yTest,type=2):
        if type != 1:
            xRange = np.arange(0,yTest.shape[0])
            plt.scatter(xRange, yTest.flatten(), color = 'blue',label='Truth')
            plt.scatter(xRange, yPred, color ='red',label='Pred')
            plt.legend()
            plt.show()

        if type != 0:
            yMax = max([max(yTest), max(yPred)])
            print(yMax, max(yTest), max(yPred))
            truth = np.arange(0,yMax)
            plt.plot(truth, truth, color='blue', label='equal')
            plt.scatter(yTest, yPred, color='red', s=1, label='comparison')
            plt.xlabel("Truth")
            plt.ylabel("Prediction")
            plt.legend()
            plt.show()


# Define the log likelihood function
def neg_log_likelihood(y_true, y_pred):
    return -tfp.distributions.Normal(loc=y_pred, scale=1.0).log_prob(y_true)
