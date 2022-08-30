import numpy as np
import pandas as pd

import torch

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from deepnegpol import DeepNEGPOLEstimator

from hyperparams import Hyperparams
from train_and_plot_predictions import data_creation, metrics_rolling_dataset, plot
params = Hyperparams()


##############################################################################################
# IMPORTATION DATA
##############################################################################################
train_ds, test_ds, target_dim, freq, prediction_length = data_creation(params, data = "taxi")
	    

##############################################################################################
# TRAINING
##############################################################################################
if params.modele == "deepnegpol":
   from pts import Trainer
   estimator = DeepNEGPOLEstimator(
        input_size1 = 12,
        input_size2 = 7290,
        num_cells1 = params.num_cells1,
        num_cells2 = params.num_cells2,
        num_layers1=params.num_layers,
        num_layers2=params.num_layers,
        dropout_rate=params.dropout_rate,
        target_dim=target_dim,
        prediction_length=prediction_length,
        freq=freq,
        scaling=True,
        lags_seq = params.lags_seq,
        trainer=Trainer(
         epochs=params.epochs,
         batch_size=params.batch_size,
         learning_rate=params.learning_rate,
         num_batches_per_epoch=params.num_batches_per_epoch      )
    )


if params.modele == "lstmmaf":
   from pts import Trainer
   estimator = TempFlowEstimator(
    target_dim=target_dim,
    prediction_length=prediction_length,
    cell_type = 'LSTM',
    input_size = 8504,
    lags_seq = params.lags_seq,
    freq=freq,
    scaling = True,
    dequantize = True,
    flow_type = 'MAF',
    trainer=Trainer(
         epochs=params.epochs,
         batch_size=params.batch_size,
         learning_rate=params.learning_rate,
         num_batches_per_epoch=params.num_batches_per_epoch      )
    )




predictor = estimator.train(train_ds)


##############################################################################################
# CALCULATE METRICS ON ROLLING DATASET
##############################################################################################
targets, forecasts = metrics_rolling_dataset(test_ds, predictor, params)
		    
	
	
		    
##############################################################################################
#PLOT
##############################################################################################
plot(targets[0], forecasts[0], prediction_length = prediction_length, prediction_intervals=(50.0, 90.0), color='g', fname=None)






