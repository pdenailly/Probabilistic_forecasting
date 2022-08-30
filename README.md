# Deep Probabilistic forecasting with DeepNegPol

This repository gathers probabilistic prediction models based on recurrent neural networks found in the GluonTS and PytorchTS libraries. We have added another model based on a "sums and shares" distribution called DeepNegPol. The models can be applied to open source datasets: Pedestrians and Taxi.


# How it works
There are 7 models with different architectures to compare in this study. They are the following models: 
|  Model   | 
|  ----  | 
| LSTMIndScaling | 
| LSTMCOP | 
| GPCOP | 
| GP | 
| GPScaling | 
| DeepNegPol | 
| LSTMMAF |

A learning phase is performed on a dedicated training set and then the prediction capabilities are computed on a test set. Different variants of these models should be compared according to which hyperparameters are taken into account.


# Use
The user should use two files:
* hyperparams : modification of the hyperparameters to be tested.
* code_pedestrian or code_taxi : script to execute. This script launches the preprocessing of the data (division between training and testing), the training with the model and the chosen parameters, the calculation of test metrics on the test base and the creation of prediction graphs.

The name of the desired model is to be modified in the file "hyperparams", there are in particular "deepnegpol", "lstmmaf","lstmcop","lstmindscaling","gpcop","gpscaling".




