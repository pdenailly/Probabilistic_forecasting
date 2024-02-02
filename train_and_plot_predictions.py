# Standard library imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from gluonts.evaluation import MultivariateEvaluator, Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset

from hyperparams import Hyperparams


from gluonts.dataset.common import ListDataset

import warnings
warnings.filterwarnings("ignore")

import os

from gluonts.dataset.rolling_dataset import (
    StepStrategy,
    generate_rolling_dataset,
)
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas


params = Hyperparams(hybridize=True)
path_folder = params.path_folder 



#############################################################################################
# IMPORTATION DONNEES - COMPTAGES + COVARIABLES
##############################################################################################

def data_creation(params, data):
    
   if data == "ferre": 
       print("Experiment with ferre data")
       data_frame = pd.read_csv('ferre_data.csv', sep=",", index_col=0, parse_dates=True, decimal='.').iloc[: , 1:] 
       index = pd.date_range(start=pd.Timestamp('2021-01-01', freq='D'), periods=data_frame.shape[0], freq='D')
       data_frame = data_frame.set_index(index)
       
       
       data_frame_train = data_frame[:pd.Timestamp('2021-09-30', freq='D')]
       data_frame_train = data_frame_train.transpose()

       data_frame_test = data_frame[:pd.Timestamp('2021-12-30', freq='D')]
       data_frame_test = data_frame_test.transpose()
        
       train_ds = ListDataset([{"start": pd.Timestamp('2021-01-01', freq='D'), "target": data_frame_train}], freq='D', one_dim_target=False)
       test_ds = ListDataset([{"start": pd.Timestamp('2021-01-01', freq='D'), "target": data_frame_test}], freq='D', one_dim_target=False)
        
       strategy=StepStrategy(
             prediction_length=7,
             step_size=7
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

       index = pd.date_range(start=start, periods=target.shape[1], freq='D')
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
           if nb_j > 12:
              break
  
       test_ds = ds
       target_dim=int(test_ds[1]['target'].shape[0])
       freq='D'
       prediction_length = 7

    
    
    
   if data == "bike":
       print("Experiment with bike data")
       data_frame = pd.read_csv('bike_data.csv', sep=",", index_col=1, parse_dates=True, decimal='.').iloc[: , 1:]  
       data_frame_train = data_frame[:pd.Timestamp('2022-05-31 23:00:00', freq='H')]
       data_frame_train = data_frame_train.transpose()

       data_frame_test = data_frame
       data_frame_test = data_frame_test.transpose()
        
       train_ds = ListDataset([{"start": pd.Timestamp('2022-01-03 00:00:00', freq='H'), "target": data_frame_train}], freq='H', one_dim_target=False)
       test_ds = ListDataset([{"start": pd.Timestamp('2022-01-03 00:00:00', freq='H'), "target": data_frame_test}], freq='H', one_dim_target=False)
        
       strategy=StepStrategy(
             prediction_length=24,
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

       index = pd.date_range(start=start, periods=target.shape[1], freq='H')
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
           if nb_j > 30:
              break
  
       test_ds = ds
       target_dim=int(test_ds[1]['target'].shape[0])
       freq='H'
       prediction_length = 24



   if data == "traffic":
       print("Experiment with traffic data")
       data_frame = pd.read_csv('trafic_hambourg.csv', sep=",", index_col=1, parse_dates=True, decimal='.').iloc[: , 1:]  
       data_frame_train = data_frame[:pd.Timestamp('2016-10-15 23:45:00', freq='15min')]
       data_frame_train = data_frame_train.transpose()

       data_frame_test = data_frame[:pd.Timestamp('2016-10-31 23:45:00', freq='15min')]
       data_frame_test = data_frame_test.transpose()
        
       train_ds = ListDataset([{"start": pd.Timestamp('2016-09-01 00:00:00', freq='15min'), "target": data_frame_train}], freq='15min', one_dim_target=False)
       test_ds = ListDataset([{"start": pd.Timestamp('2016-09-01 00:00:00', freq='15min'), "target": data_frame_test}], freq='15min', one_dim_target=False)
        
       strategy=StepStrategy(
             prediction_length=24,
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
       index = pd.date_range(start=start, periods=target.shape[1], freq='15min')
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
           if nb_j > 63:
              break
              
       test_ds = ds
       target_dim=int(test_ds[1]['target'].shape[0])
       freq='15min'
       prediction_length = 24


    
  
   if data == "wikipedia":
      print("Experiment on Wiki data")
      
      dataset = get_dataset("wiki-rolling_nips", regenerate=False)
      

      data_test_l = []
      for item in dataset.test:
          data_test_l.append((item['target'])[:])
          
      data_test_l = data_test_l[-int(dataset.metadata.feat_static_cat[0].cardinality):]
      data_test = np.concatenate([i for i in data_test_l]).reshape(( len(data_test_l), len(data_test_l[0]) ))


      train_ds = ListDataset([{"start": pd.Timestamp('2012-01-01 00:00:00', freq='D'),
                         "target": data_test[0:2000,:762]
                          }], freq=dataset.metadata.freq, one_dim_target=False)

      test_ds = ListDataset([{"start": pd.Timestamp('2012-01-01 00:00:00', freq='D'),
                         "target": data_test[0:2000,:]
                          }], freq=dataset.metadata.freq, one_dim_target=False)



      strategy=StepStrategy(
           prediction_length=30,
           step_size=30
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
          if nb_j > 4:
             break
 
      test_ds = ds
      target_dim=int(test_ds[1]['target'].shape[0])
      freq=dataset.metadata.freq
      prediction_length = dataset.metadata.prediction_length


      
      
   if data == "taxi":
         print("Experiment on Taxi data")
         
         dataset = get_dataset("taxi_30min", regenerate=True)
         dataset.metadata

         train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

         test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                      max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

         train_ds = train_grouper(dataset.train)
         test_ds = test_grouper(dataset.test)
         target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
         freq=dataset.metadata.freq
         prediction_length = dataset.metadata.prediction_length
      
   if data == "pedestrians":
      print("Experiment on Pedestrians data")
      dataset = get_dataset("pedestrian_counts", regenerate=False)

      data_test_l = []
      for item in dataset.test:
          if item['start'] == pd.Timestamp("2009-05-01 00:00:00" , freq='H') and int(item['target'].shape[0]) > 50000:
              data_test_l.append((item['target'])[:24000])

      data_test = np.concatenate([i for i in data_test_l]).reshape(( len(data_test_l), len(data_test_l[0]) ))


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
 
      test_ds = ds
      target_dim=int(data_test.shape[0])
      freq=dataset.metadata.freq
      prediction_length = dataset.metadata.prediction_length

   return train_ds, test_ds, target_dim, freq, prediction_length







##############################################################################################
# CALCULATE METRICS ON ROLLING DATASET
##############################################################################################


def metrics_rolling_dataset1(test_ds, predictor,params, test, rep,data):
       
    
    name = str(test['model']) + str(rep) + '_' + 'lr:'+str(test['lr']) +'_'+'nbcells1:'+str(test['num_cells1'])+'nbcells2:'+str(test['num_cells2']) + '_'+'nbcells:'+str(test['num_cells'])

    
    #forecast rolling data
    forecast_it, ts_it = make_evaluation_predictions(
                test_ds, predictor=predictor, num_samples=params.num_eval_samples
            )


    print("predicting")
    forecasts = list(forecast_it)
    targets = list(ts_it)


    
    # evaluate
    evaluator = MultivariateEvaluator(
                    quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum}
                )

    agg_metric, item_metrics = evaluator(
                    targets, forecasts, num_series=len(test_ds)
                )


    print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
    print("ND: {}".format(agg_metric['ND']))
    print("NRMSE: {}".format(agg_metric['NRMSE']))
    print("MSE: {}".format(agg_metric['MSE']))
    print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
    print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
    print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
    print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))
    
    
    metrics_dic = {'metrics': ['CRPS','MSE','CRPS-Sum','MSE-Sum'], 'values':[agg_metric['mean_wQuantileLoss'],
                                                                             agg_metric['MSE'],
                                                                             agg_metric['m_sum_mean_wQuantileLoss'],
                                                                             agg_metric['m_sum_MSE']]}
    metrics_df = pd.DataFrame.from_dict(metrics_dic)
    
    path_folder = "."
    
    import os 
    plot_log_path = path_folder+"/results1/"+data+"/"+str(test['model'])+"_results/"
    directory = os.path.dirname(plot_log_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    metrics_df.to_csv('{}metrics_{}.csv'.format(plot_log_path,name))
    print('Done')


    return targets, forecasts        




def metrics_rolling_dataset2(test_ds, predictor,params, test, rep,data):
       
    
    name = str(test['model']) + str(rep) + '_' + 'lr:'+str(test['lr']) +'_'+'nbcells1:'+str(test['num_cells1'])+'nbcells2:'+str(test['num_cells2']) + '_'+'nbcells:'+str(test['num_cells'])

    
    #forecast rolling data
    forecast_it, ts_it = make_evaluation_predictions(
                test_ds, predictor=predictor, num_samples=params.num_eval_samples
            )


    print("predicting")
    forecasts = list(forecast_it)
    targets = list(ts_it)


    
    # evaluate
    evaluator = MultivariateEvaluator(
                    quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum}
                )

    agg_metric, item_metrics = evaluator(
                    targets, forecasts, num_series=len(test_ds)
                )


    print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
    print("ND: {}".format(agg_metric['ND']))
    print("NRMSE: {}".format(agg_metric['NRMSE']))
    print("MSE: {}".format(agg_metric['MSE']))
    print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
    print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
    print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
    print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))
    
    
    metrics_dic = {'metrics': ['CRPS','MSE','CRPS-Sum','MSE-Sum'], 'values':[agg_metric['mean_wQuantileLoss'],
                                                                             agg_metric['MSE'],
                                                                             agg_metric['m_sum_mean_wQuantileLoss'],
                                                                             agg_metric['m_sum_MSE']]}
    metrics_df = pd.DataFrame.from_dict(metrics_dic)
    
    path_folder = "."
    
    import os 
    plot_log_path = path_folder+"/results2/"+data+"/"+str(test['model'])+"_results/"
    directory = os.path.dirname(plot_log_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    metrics_df.to_csv('{}metrics_{}.csv'.format(plot_log_path,name))
    print('Done')
    
    
    metrics = pd.DataFrame.from_records(agg_metric,index = [test['model']]).transpose()
    metrics['Settings'] = name
    metrics.to_csv('{}ALL_metrics_{}.csv'.format(plot_log_path,name))


    return targets, forecasts        



##############################################################################################
#PLOTS FROM PARTICULAR PERIODS
##############################################################################################
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

