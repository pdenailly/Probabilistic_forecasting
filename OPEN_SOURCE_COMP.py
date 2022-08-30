import numpy as np
import pandas as pd

import torch

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




######################### Plot function


import matplotlib.pyplot as plt

def plot(target, forecast, prediction_length, prediction_intervals=(50.0, 90.0), color='g', fname=None):
    label_prefix = ""
    rows = 3
    cols = 5
    fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
    axx = axs.ravel()
    seq_len, target_dim = target.shape
    
    ps = [50.0] + [
            50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
        ]
        
    percentiles_sorted = sorted(set(ps))
    
    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3
        
    for dim in range(0, min(rows * cols, target_dim)):
        ax = axx[dim]

        target[-2 * prediction_length :][dim].plot(ax=ax)
        
        ps_data = [forecast.quantile(p / 100.0)[:,dim] for p in percentiles_sorted]
        i_p50 = len(percentiles_sorted) // 2
        
        p50_data = ps_data[i_p50]
        p50_series = pd.Series(data=p50_data, index=forecast.index)
        p50_series.plot(color=color, ls="-", label=f"{label_prefix}median", ax=ax)
        
        for i in range(len(percentiles_sorted) // 2):
            ptile = percentiles_sorted[i]
            alpha = alpha_for_percentile(ptile)
            ax.fill_between(
                forecast.index,
                ps_data[i],
                ps_data[-i - 1],
                facecolor=color,
                alpha=alpha,
                interpolate=True,
            )
            # Hack to create labels for the error intervals.
            # Doesn't actually plot anything, because we only pass a single data point
            pd.Series(data=p50_data[:1], index=forecast.index[:1]).plot(
                color=color,
                alpha=alpha,
                linewidth=10,
                label=f"{label_prefix}{100 - ptile * 2}%",
                ax=ax,
            )






#############################################################################################################"
## TAXI 
#############################################################################################################

########### Prepare data set

dataset = get_dataset("taxi_30min", regenerate=False)
dataset.metadata

train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                   max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)

###########  Evaluator
evaluator = MultivariateEvaluator(
                    quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum}
                )




############ DEEPVAR
from pts.model.deepvar import DeepVAREstimator
from pts.modules import LowRankMultivariateNormalOutput

estimator = DeepVAREstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length,
    cell_type='LSTM',
    input_size=8507,
    freq=dataset.metadata.freq,
    scaling=True,
    dropout_rate = 0.01,
    distr_output = LowRankMultivariateNormalOutput(int(dataset.metadata.feat_static_cat[0].cardinality),10),
    rank = 10,
    lags_seq = [1,2,4,12,24,48],
    trainer=Trainer(device=device,
                    epochs=40,
                    learning_rate=1e-2,
                    num_batches_per_epoch=100,
                    batch_size=16),
    conditioning_length = 100,
    use_marginal_transformation = True
)




############ DEEPNEGPOL
from model.deepnegpol import DeepNEGPOLEstimator


estimator = DeepNEGPOLEstimator(
        input_size1 = 12,
        input_size2 = 7290,
        num_cells1 = 40,
        num_cells2 = 80,
        num_layers1=2,
        num_layers2=2,
        dropout_rate=0.01,
        target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        scaling=True,
        lags_seq = [1,2,4,12,24,48],
        trainer=Trainer(
         epochs=100,
         batch_size=16,  # todo make it dependent from dimension
         learning_rate=1e-2,
         patience=50,
         num_batches_per_epoch=100      )
    )






############ LSTM-MAF
from pts.model.tempflow import TempFlowEstimator
from pts import Trainer

estimator = TempFlowEstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length,
    cell_type = 'LSTM',
    input_size = 8504,
    lags_seq = [1,2,4,12,24,48],
    freq=dataset.metadata.freq,
    scaling = True,
    dequantize = True,
    flow_type = 'MAF',
    trainer=Trainer(
                    epochs=100,
                    learning_rate=1e-2,
                    num_batches_per_epoch=100,
                    batch_size=16,
                    patience = 50)
    )







predictor = estimator.train(dataset_train)



forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=predictor,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))



print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))
print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))



import pathlib
import os
os.makedirs('save_models', exist_ok=True)

predictor.serialize(pathlib.Path("taxi_lstmMAF"))


from gluonts.model.predictor import Predictor
saved_model = Predictor.deserialize(pathlib.Path("taxi_lstmMAF"))

##test saved model
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=saved_model,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))



plot(
    target=targets[4],
    forecast=forecasts[4],
    prediction_length=dataset.metadata.prediction_length,
)
plt.show()










############ LSTM-NVD
from pts.model.tempflow import TempFlowEstimator
from pts import Trainer

estimator = TempFlowEstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length,
    cell_type = 'LSTM',
    input_size = 8504,
    lags_seq = [1,2,4,12,24,48],
    freq=dataset.metadata.freq,
    scaling = True,
    dequantize = True,
    n_blocks=4,
    trainer=Trainer(
                    epochs=100,
                    learning_rate=1e-2,
                    num_batches_per_epoch=100,
                    batch_size=16)
    )




predictor = estimator.train(dataset_train)



forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=predictor,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))



print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))
print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))



import pathlib
import os
os.makedirs('save_models', exist_ok=True)

predictor.serialize(pathlib.Path("taxi_lstmNVD"))


from gluonts.model.predictor import Predictor
saved_model = Predictor.deserialize(pathlib.Path("taxi_lstmNVD"))











"""

#############################################################################################################"
## WIKIPEDIA
#############################################################################################################

########### Prepare data set

dataset = get_dataset("wiki-rolling_nips", regenerate=False)
dataset.metadata

train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                   max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)

###########  Evaluator
evaluator = MultivariateEvaluator(
                    quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum}
                )






############ DEEPNEGPOL
from model.deepnegpol import DeepNEGPOLEstimator


estimator = DeepNEGPOLEstimator(
        input_size1 = 5,
        input_size2 = 28607,
        num_cells1 = 40,
        num_cells2 = 80,
        num_layers1=2,
        num_layers2=2,
        dropout_rate=0.01,
        target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        scaling=True,
        lags_seq = [1,7,14],
        trainer=Trainer(
         epochs=100,
         batch_size=16,  # todo make it dependent from dimension
         learning_rate=1e-2,
         patience=50,
         num_batches_per_epoch=100      )
    )





predictor = estimator.train(dataset_train)



forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=predictor,
                                             num_samples=1)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))



print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))
print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))



import pathlib
import os
os.makedirs('save_models', exist_ok=True)

predictor.serialize(pathlib.Path("taxi_deepnegpol"))


from gluonts.model.predictor import Predictor
saved_model = Predictor.deserialize(pathlib.Path("taxi_deepnegpol"))

##test saved model
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=saved_model,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))



plot(
    target=targets[4],
    forecast=forecasts[4],
    prediction_length=dataset.metadata.prediction_length,
)
plt.show()


"""















































#############################################################################################################"
## PEDESTRIAN COUNTS
#############################################################################################################


from gluonts.dataset.common import ListDataset
from gluonts.dataset.rolling_dataset import (
    StepStrategy,
    generate_rolling_dataset,
)

########### Prepare data set ############################################################################################

dataset = get_dataset("pedestrian_counts", regenerate=False)

dataset.metadata

data_test_l = []
for item in dataset.test:
    if item['start'] == pd.Timestamp("2009-05-01 00:00:00" , freq='H') and int(item['target'].shape[0]) > 50000:
       data_test_l.append((item['target'])[:24000])

data_test = np.concatenate([i for i in data_test_l]).reshape(( len(data_test_l), len(data_test_l[0]) ))



# Creation of a new train and test

train_ds = ListDataset([{"start": pd.Timestamp("2009-05-01 00:00:00" , freq='H'),
                         "target": data_test[:,:800*24]
                          }], freq=dataset.metadata.freq, one_dim_target=False)

test_ds = ListDataset([{"start": pd.Timestamp("2009-05-01 00:00:00" , freq='H'),
                         "target": data_test
                          }], freq=dataset.metadata.freq, one_dim_target=False)



strategy=StepStrategy(
           prediction_length=48,
           step_size=24
       )


def truncate_features(timeseries: dict, max_len: int) -> dict:
    for key in (
       'feat_dynamic_real',
    ):
        if key not in timeseries:
            continue
        timeseries[key] = (timeseries[key])[:,:max_len]

    return timeseries


ds = []

item = (next(iter(test_ds)))
target = item["target"]
start = item["start"]

index = pd.date_range(start=start, periods=target.shape[1], freq=dataset.metadata.freq)
series = pd.DataFrame(target.T, index=index)

prediction_window = series
nb_j = 0
        
for window in strategy.get_windows(prediction_window):
    nb_j = nb_j + 1
    new_item = item.copy()
    new_item['target'] = np.concatenate(
        [window.to_numpy()]
        ).T
    new_item = truncate_features(
        new_item, new_item['target'].shape[1]
        )
    ds.append(new_item)
    if nb_j > 15:
        break




###########  Evaluator
evaluator = MultivariateEvaluator(
                    quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum}
                )


###########  TRAIN ############################################################################################


############ DEEPNEGPOL
from model.deepnegpol import DeepNEGPOLEstimator

estimator = DeepNEGPOLEstimator(
        input_size1 = 7,
        input_size2 = 46,
        num_cells1 = 40,
        num_cells2 = 80,
        num_layers1=2,
        num_layers2=2,
        dropout_rate=0.01,
        target_dim=int(data_test.shape[0]),
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        scaling=True,
        lags_seq = [1,24, 168],
        trainer=Trainer(
         epochs=100,
         batch_size=16,  # todo make it dependent from dimension
         learning_rate=1e-2,
         patience=50,
         num_batches_per_epoch=100      )
    )





predictor = estimator.train(train_ds)



import pathlib
import os
os.makedirs('people_count/deepnegpol', exist_ok=True)
predictor.serialize(pathlib.Path("people_count/deepnegpol"))


from gluonts.model.predictor import Predictor
saved_model = Predictor.deserialize(pathlib.Path("people_count/deepnegpol"))



############ LSTMCOP
from pts.model.deepvar import DeepVAREstimator
from pts.modules import LowRankMultivariateNormalOutput

estimator = DeepVAREstimator(
    target_dim=int(data_test.shape[0]),
    prediction_length=dataset.metadata.prediction_length,
    cell_type='LSTM',
    input_size=63,
    freq=dataset.metadata.freq,
    scaling=False,
    dropout_rate = 0.01,
    distr_output = LowRankMultivariateNormalOutput(int(data_test.shape[0]),5),
    rank = 5,
    lags_seq = [1,24, 168],
    trainer=Trainer(device=device,
                    epochs=100,
                    learning_rate=1e-2,
                    num_batches_per_epoch=100,
                    batch_size=16),
    conditioning_length = 100,
    use_marginal_transformation = True
)



predictor = estimator.train(train_ds)



import pathlib
import os
os.makedirs('people_count/lstmcop', exist_ok=True)
predictor.serialize(pathlib.Path("people_count/lstmcop"))


from gluonts.model.predictor import Predictor
saved_model = Predictor.deserialize(pathlib.Path("people_count/lstmcop"))





############ LSTM IND SCALING
from pts.model.deepvar import DeepVAREstimator
#from gluonts.mx.distribution.multivariate_independent_gaussian import MultivariateIndependentGaussianOutput
from pts.modules import NormalOutput


estimator = DeepVAREstimator(
    target_dim=int(data_test.shape[0]),
    prediction_length=dataset.metadata.prediction_length,
    cell_type='LSTM',
    input_size=63,
    freq=dataset.metadata.freq,
    scaling=True,
    dropout_rate = 0.01,
    distr_output = NormalOutput(int(data_test.shape[0])),
    lags_seq = [1,24, 168],
    trainer=Trainer(device=device,
                    epochs=100,
                    learning_rate=1e-2,
                    num_batches_per_epoch=100,
                    batch_size=16),
    use_marginal_transformation = False
)



predictor = estimator.train(train_ds)



import pathlib
import os
os.makedirs('people_count/lstmindscaling', exist_ok=True)
predictor.serialize(pathlib.Path("people_count/lstmindscaling"))


from gluonts.model.predictor import Predictor
saved_model = Predictor.deserialize(pathlib.Path("people_count/lstmindscaling"))







############ GP scaling
from gluonts.model.gpvar import GPVAREstimator
from gluonts.mx.distribution.lowrank_gp import LowrankGPOutput
from gluonts.mx.trainer import Trainer

estimator = GPVAREstimator(
            target_dim=int(data_test.shape[0]),
            dropout_rate=0.01,
            prediction_length=dataset.metadata.prediction_length,
            cell_type="lstm",
            target_dim_sample=5,
            lags_seq=[1,24, 168],
            conditioning_length=100,
            scaling=True,
            freq=dataset.metadata.freq,
            rank = 5,
            use_marginal_transformation=False,
            distr_output=LowrankGPOutput(rank = 5, dim = int(data_test.shape[0])),
            trainer=Trainer(
                    epochs=100,
                    learning_rate=1e-2,
                    num_batches_per_epoch=100,
                    batch_size=16),
        )


predictor = estimator.train(train_ds)



import pathlib
import os
os.makedirs('people_count/gpscaling', exist_ok=True)
predictor.serialize(pathlib.Path("people_count/gpscaling"))


from gluonts.model.predictor import Predictor
saved_model = Predictor.deserialize(pathlib.Path("people_count/gpscaling"))









############ GPCOP
from gluonts.model.gpvar import GPVAREstimator
from gluonts.mx.distribution.lowrank_gp import LowrankGPOutput
from gluonts.mx.trainer import Trainer

estimator = GPVAREstimator(
            target_dim=int(data_test.shape[0]),
            dropout_rate=0.01,
            prediction_length=dataset.metadata.prediction_length,
            cell_type="lstm",
            target_dim_sample=5,
            lags_seq=[1,24, 168],
            conditioning_length=100,
            scaling=False,
            freq=dataset.metadata.freq,
            rank = 5,
            use_marginal_transformation=True,
            distr_output=LowrankGPOutput(rank = 5, dim = int(data_test.shape[0])),
            trainer=Trainer(
                    epochs=100,
                    learning_rate=1e-2,
                    num_batches_per_epoch=100,
                    batch_size=16),
        )


predictor = estimator.train(train_ds)



import pathlib
import os
os.makedirs('people_count/gpcop', exist_ok=True)
predictor.serialize(pathlib.Path("people_count/gpcop"))


from gluonts.model.predictor import Predictor
saved_model = Predictor.deserialize(pathlib.Path("people_count/gpcop"))






############ LSTM-MAF
from pts.model.tempflow import TempFlowEstimator
from pts import Trainer

estimator = TempFlowEstimator(
    target_dim=int(data_test.shape[0]),
    prediction_length=dataset.metadata.prediction_length,
    cell_type = 'LSTM',
    input_size = 60,
    lags_seq=[1,24, 168],
    freq=dataset.metadata.freq,
    scaling = True,
    dequantize = True,
    flow_type = 'MAF',
    trainer=Trainer(
                    epochs=100,
                    learning_rate=1e-2,
                    num_batches_per_epoch=100,
                    batch_size=16)
    )


predictor = estimator.train(train_ds)


import pathlib
import os
os.makedirs('people_count/lstmMAF', exist_ok=True)
predictor.serialize(pathlib.Path("people_count/lstmMAF"))


from gluonts.model.predictor import Predictor
saved_model = Predictor.deserialize(pathlib.Path("people_count/lstmMAF"))







############ LSTM-REAL NVP
from pts.model.tempflow import TempFlowEstimator
from pts import Trainer


estimator = TempFlowEstimator(
    target_dim=int(data_test.shape[0]),
    prediction_length=dataset.metadata.prediction_length,
    cell_type = 'LSTM',
    input_size = 60,
    lags_seq=[1,24, 168],
    freq=dataset.metadata.freq,
    scaling = True,
    dequantize = True,
    n_blocks=4,
    trainer=Trainer(
                    epochs=100,
                    learning_rate=1e-2,
                    num_batches_per_epoch=100,
                    batch_size=16)
    )


predictor = estimator.train(train_ds)


import pathlib
import os
os.makedirs('people_count/lstmNVP', exist_ok=True)
predictor.serialize(pathlib.Path("people_count/lstmNVP"))


from gluonts.model.predictor import Predictor
saved_model = Predictor.deserialize(pathlib.Path("people_count/lstmNVP"))












##test saved model
forecast_it, ts_it = make_evaluation_predictions(dataset=ds,
                                             predictor=predictor,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=14)



print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))
print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))



plot(
    target=targets[5],
    forecast=forecasts[5],
    prediction_length=dataset.metadata.prediction_length,
)
plt.show()



