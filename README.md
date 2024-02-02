# Deep Probabilistic forecasting with DeepNegPol

This repository gathers probabilistic prediction models based on recurrent neural networks found in the GluonTS and PytorchTS libraries. We have added another model based on a "sums and shares" distribution called DeepNegPol (inspired from the work of [Jones and Marchand (2019)](https://doi.org/10.1016/j.jmva.2018.11.011))). The models are applied to open source count datasets.


# How it works
There are 8 models with different architectures to compare in this study. They are the following models: 
|  Model   | 
|  ----  | 
| LSTMIndScaling | 
| LSTMCOP | 
| GPCOP | 
| GP | 
| GPScaling | 
| DeepNegPol | 
|TimeGrad|
| LSTMMAF |

LSTMIndScaling, LSTMCOP, GPCOP and GPScaling are models from [Salinas, David, et al. (2019)](https://doi.org/10.48550/arXiv.1910.03002)), LSTMMAF is from [Rasul, Kashif, et al. (2020)](https://doi.org/10.48550/arXiv.2002.06103)), TimeGrad is from [Rasul, Kashif, et al. (2021)](https://doi.org/10.48550/arXiv.2101.12072) and DeepNegPol is our model.

A learning phase is performed on a dedicated training set and then the prediction capabilities are computed on a test set. Different variants of these models should be compared according to which hyperparameters are taken into account.


# Use
The user should use two files:
* hyperparams : modification of the hyperparameters to be tested.
* example_taxi.py, example_traffic.py, example_wikipedia.py or example_bike.py: script to execute. These scripts launche the preprocessing of the data (division between training and testing), the training with the model and the chosen parameters, the calculation of test metrics on the test base.





