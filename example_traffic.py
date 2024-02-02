import numpy as np
import pandas as pd

import torch

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts.model.time_grad import TimeGradEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

from gluonts.dataset.common import ListDataset
from gluonts.dataset.rolling_dataset import (
    StepStrategy,
    generate_rolling_dataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from deepnegpol import DeepNEGPOLEstimator
from pts.model.deepvar import DeepVAREstimator
from pts.modules import LowRankMultivariateNormalOutput
from pts.modules import NormalOutput


from gluonts.model.gpvar import GPVAREstimator
from gluonts.mx.distribution.lowrank_gp import LowrankGPOutput



from hyperparams import Hyperparams
from train_and_plot_predictions import data_creation, metrics_rolling_dataset1,metrics_rolling_dataset2, plot
params = Hyperparams()

import warnings
warnings.filterwarnings("ignore")



####################################################################################
# TRAFFIC HAMBOURG
####################################################################################

train_ds, test_ds, target_dim, freq, prediction_length = data_creation(params, data = "traffic")
	





##############################################################################################
# TRAINING
##############################################################################################
def estimator_model(test):
    
 modele = test['model']
 num_cells =  test['num_cells']
 num_cells1 = test['num_cells1']  
 num_cells2 = test['num_cells2']  
 lr = test['lr']
 
 
 if modele == "timegrad":
    from pts import Trainer
    estimator = TimeGradEstimator(
        input_size = 1574,
        num_cells = num_cells,
        num_layers=params.num_layers,
        dropout_rate=params.dropout_rate,
        target_dim=target_dim,
        prediction_length=prediction_length,
        context_length = prediction_length,
        cell_type='LSTM',
        freq=freq,
        loss_type='l2',
        diff_steps=100,
        beta_end=0.1,
        beta_schedule="linear",
        scaling=True,
        lags_seq = params.lags_seq4,
        trainer=Trainer(
         epochs=params.epochs,
         batch_size=params.batch_size,
         learning_rate=lr,
         num_batches_per_epoch=params.num_batches_per_epoch      )
    )

      
 if modele == "deepnegpol":
   from pts import Trainer
   estimator = DeepNEGPOLEstimator(
        input_size1 = 9,
        input_size2 = 1182,
        num_cells1 = num_cells1,
        num_cells2 = num_cells2,
        num_layers1=params.num_layers,
        num_layers2=params.num_layers,
        dropout_rate=params.dropout_rate,
        target_dim=target_dim,
        prediction_length=prediction_length,
        freq=freq,
        scaling=True,
        lags_seq = params.lags_seq4,
        trainer=Trainer(
         epochs=params.epochs,
         batch_size=params.batch_size,
         learning_rate=lr,
         num_batches_per_epoch=params.num_batches_per_epoch      )
    )
   
   
 if modele == "lstmcop":
   from pts import Trainer
   estimator = DeepVAREstimator(
    target_dim=target_dim,
    prediction_length=prediction_length,
    num_cells = num_cells,
    cell_type='LSTM',
    input_size=1577,
    freq=freq,
    scaling=False,
    dropout_rate=params.dropout_rate,
    distr_output = LowRankMultivariateNormalOutput(target_dim,params.rank),
    rank = params.rank,
    lags_seq = params.lags_seq4,
    trainer=Trainer(
         epochs=params.epochs,
         batch_size=params.batch_size,
         learning_rate=lr,
         num_batches_per_epoch=params.num_batches_per_epoch      ),
    conditioning_length = params.conditioning_length,
    use_marginal_transformation = True
   )
   
   
 if modele == "lstmindscaling":
   from pts import Trainer
   estimator = DeepVAREstimator(
    target_dim=target_dim,
    prediction_length=prediction_length,
    cell_type='LSTM',
    num_cells = num_cells,
    input_size=1577,
    freq=freq,
    scaling=True,
    dropout_rate=params.dropout_rate,
    distr_output = LowRankMultivariateNormalOutput(target_dim,params.rank),
    rank = params.rank,
    lags_seq = params.lags_seq4,
    trainer=Trainer(
         epochs=params.epochs,
         batch_size=params.batch_size,
         learning_rate=lr,
         num_batches_per_epoch=params.num_batches_per_epoch      ),
    conditioning_length = params.conditioning_length,
    use_marginal_transformation = False
   )
   
   
   
   
 if modele == "gpscaling":
   from gluonts.mx.trainer import Trainer
   estimator = GPVAREstimator(
            target_dim=target_dim,
            num_cells = num_cells,
            dropout_rate=params.dropout_rate,
            prediction_length=prediction_length,
            cell_type="lstm",
            target_dim_sample=params.target_dim_sample,
            lags_seq = params.lags_seq4,
            conditioning_length = params.conditioning_length,
            scaling=True,
            freq=freq,
            rank = params.rank,
            use_marginal_transformation=False,
            distr_output=LowrankGPOutput(rank = params.rank, dim = target_dim),
            trainer=Trainer(
         epochs=params.epochs,
         batch_size=params.batch_size,
         learning_rate=lr,
         num_batches_per_epoch=params.num_batches_per_epoch      )
        )
 



 if modele == "gpcop":
   from gluonts.mx.trainer import Trainer
   estimator = GPVAREstimator(
            target_dim=target_dim,
            dropout_rate=params.dropout_rate,
            prediction_length=prediction_length,
            cell_type="lstm",
            num_cells = num_cells,
            target_dim_sample=params.target_dim_sample,
            lags_seq = params.lags_seq4,
            conditioning_length = params.conditioning_length,
            scaling=False,
            freq=freq,
            rank = params.rank,
            use_marginal_transformation=True,
            distr_output=LowrankGPOutput(rank = params.rank, dim = target_dim),
            trainer=Trainer(
         epochs=params.epochs,
         batch_size=params.batch_size,
         learning_rate=lr,
         num_batches_per_epoch=params.num_batches_per_epoch      ),
        )




 if modele == "lstmmaf":
   from pts import Trainer
   estimator = TempFlowEstimator(
    target_dim=target_dim,
    prediction_length=prediction_length,
    cell_type = 'LSTM',
    input_size = 1574,
    lags_seq = params.lags_seq4,
    num_cells = num_cells,
    freq=freq,
    n_blocks = 2,
    scaling = True,
    dequantize = True,
    flow_type = 'MAF',
    trainer=Trainer(
         epochs=params.epochs,
         batch_size=params.batch_size,
         learning_rate=lr,
         num_batches_per_epoch=params.num_batches_per_epoch      )
    )
  
   
 return estimator





test0 = {'model':'lstmmaf', 'num_cells':40  ,  'lr':1e-3, 'num_cells1':20, 'num_cells2':40}
test1 = {'model':'deepnegpol', 'num_cells':80  ,  'lr':1e-3, 'num_cells1':20, 'num_cells2':40}
test2 = {'model':'lstmcop', 'num_cells':40  ,  'lr':1e-3, 'num_cells1':20, 'num_cells2':40}
test3 = {'model':'lstmindscaling', 'num_cells':40  ,  'lr':1e-3, 'num_cells1':20, 'num_cells2':40}
test4 = {'model':'gpcop', 'num_cells':40  ,  'lr':1e-3, 'num_cells1':20, 'num_cells2':40}
test5 = {'model':'gpscaling', 'num_cells':40  ,  'lr':1e-3, 'num_cells1':20, 'num_cells2':40}
test6 = {'model':'timegrad', 'num_cells':40  ,  'lr':1e-3, 'num_cells1':20, 'num_cells2':40}




list_tests_str = ['test' + str(i) for i in range (0, 25)]
list_tests= []
for test in list_tests_str:
    if test in locals():
        list_tests.append(eval(test))



data = "traffic"
import os
import pathlib
for test in list_tests:

    for rep in range(0,3):

       print("Essai numéro "+str(rep) + " du modele " + str(test['model']) + " avec "+ str(test['num_cells']) + " cellules ")
       estimator = estimator_model(test)
       try:  
          predictor = estimator.train(train_ds)
          targets, forecasts = metrics_rolling_dataset2(test_ds, predictor, params, test,rep,data)
       except:
          print("An exception occurred") 
