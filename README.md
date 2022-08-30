# Deep Probabilistic forecasting with DeepNegPol

This repository gathers probabilistic prediction models based on recurrent neural networks found in the GluonTS and PytorchTS libraries. We have added another model based on a "sums and shares" distribution called DeepNegPol (inspired from the work of [Jones and Marchand (2019)](https://doi.org/10.1016/j.jmva.2018.11.011))). The models can be applied to open source datasets: Pedestrians and Taxi.


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

LSTMIndScaling, LSTMCOP, GPCOP and GPScaling are models from [Salinas, David, et al. (2019)](https://doi.org/10.48550/arXiv.1910.03002)), LSTMMAF is from [Rasul, Kashif, et al. (2020)](https://doi.org/10.48550/arXiv.2002.06103)) and DeepNegPol is our model.

A learning phase is performed on a dedicated training set and then the prediction capabilities are computed on a test set. Different variants of these models should be compared according to which hyperparameters are taken into account.


# Use
The user should use two files:
* hyperparams : modification of the hyperparameters to be tested.
* code_pedestrian or code_taxi : script to execute. This script launches the preprocessing of the data (division between training and testing), the training with the model and the chosen parameters, the calculation of test metrics on the test base and the creation of prediction graphs.

The name of the desired model is to be modified in the file "hyperparams", there are in particular "deepnegpol", "lstmmaf","lstmcop","lstmindscaling","gpcop","gpscaling".




