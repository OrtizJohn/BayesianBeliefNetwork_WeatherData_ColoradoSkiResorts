# BayesianBeliefNetwork_WeatherData_ColoradoSkiResorts
Introduction
Accurately forecasting snowfall in the unpredictable Colorado mountains is a challenging task that can greatly impact the timing and success of ski trips. This presentation outlines an innovative approach to tackle this problem by employing a Bayesian Neural Network to predict precipitation and snowfall at popular Colorado ski resorts. By leveraging more certain weather forecasts, such as temperature, the proposed model aims to provide reliable predictions for the uncertain Colorado snowfall, empowering skiers and resort operators to make informed decisions.

Data Acquisition
The study acquired daily weather data from January 2018 through December 2022 for the ski resorts of Aspen, Steamboat, and Vail from the Colorado Weather API URL builder. The dataset includes high and low temperatures, precipitation, and snowfall measurements for each day.

Implementation
We employed a Bayesian Neural Network architecture with dense layers, dropout layers, and the tensorflow_probability.layers.DenseFlipout() method to approximate snowfall predictions given certain conditions using Kullback-Leibler Variational Inference.

Results
The current results demonstrate reasonable accuracy in predicting snowfall for some Colorado mountain resorts. However, the interdependencies between weather forecasts complicate the identification of independence features.

Conclusion and Next Steps
The use of a Bayesian Neural Network allowed for variational inference across states and the inclusion of Kullback-Leibler divergence in the model. If there was future work to be done this would include finalizing the networks for improved performance, exploring normalization techniques, refining parameter inference for precipitation and snowfall levels, and investigating potential model enhancements.


References: <br>
[1] Data API URL BUILDER. CoAgMET. (n.d.). Retrieved April 11, 2023, from
https://coagmet.colostate.edu/data/url-builder

[2] https://github.com/timzhang642/HMM-Weather/blob/master/Hidden%20Markov%20Model.ipynb
