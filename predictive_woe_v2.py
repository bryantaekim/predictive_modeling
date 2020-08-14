#!/usr/bin/env python
"""
Created on Tue Aug 11 10:23:47 2020
@author: Bryan Kim
@title: Predictive model using Weight of Evidence

@data sets:
    select * from *.pred2_stopped_2006;
    select * from *.pred2_surrender_2006;
    select * from *.pred2_restart_2006;
@target audience:
    403b_k12_policies > 0
    
@target month: 20/06
    30 days out from - 20/05
    60 days out from - 20/04
    90 days out from - 20/03
    120 days out from - 20/02

@models:
    model1 - stopped
    model2 - surrender
    model3 - restart

@WoE/IV:
    # Variables with IV - 
    # useless : <.02
    # weak : between .02 and .1
    # medium : between .1 and .3
    # strong : between .3 and .5
    # Too good to be true : > .5
"""
from __future__ import division
print(__doc__)
from pyspark.sql import SparkSession, SQLContext, functions as F, Window as W
from pyspark.sql.types import *

from pyspark.ml.feature import Bucketizer, StringIndexer, OneHotEncoderEstimator
from pyspark.ml import Pipeline

import pandas as pd, numpy as np, pickle, matplotlib.pyplot as plt, sys
from scipy import interp
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.externals import joblib

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

_seed = 7878
_QUIET_SPARK = True
spark = SparkSession.builder.appName("predictive_model_gr_v3").getOrCreate()
if _QUIET_SPARK:
    spark.sparkContext.setLogLevel("WARN")
sqlContext = SQLContext(spark.sparkContext)
sqlContext.setConf("spark.sql.autoBroadcastJoinThreshold", "-1")
sqlContext.setConf('spark.sql.shuffle.partitions','100')
sqlContext.setConf("spark.sql.files.openCostInBytes", "8388608")
sqlContext.setConf("spark.sql.optimizer.maxIterations", "500")
sqlContext.setConf("spark.sql.crossJoin.enabled", "true")
sqlContext.setConf("spark.sql.broadcastTimeout", "36000")
sqlContext.setConf("spark.default.parallelism", "300")

model_loc = '/home/bkim/predictive_model/model/'

def main():
    
    print('*** Started at  *** : ' + format(datetime.today(), '%Y-%m-%d %H:%M:%S'))
    
    _trainModel1 = False 
    _trainModel2 = _trainModel3 = True
    
    _predictModel1 = False
    _predictModel2 = False  
    _predictModel3 = False

    
    if _trainModel1:
        input_tbl = '*.pred2_stopped_2006'
        targets = ['stopped_30', 'stopped_60', 'stopped_90', 'stopped_120']
        except_for_these =  ['party_id','cmdb_partition']
        
        params_30 = {'target' : 'stopped_30',
        'month_to_target' : '2005',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_stopped30_idx',
        'bin_tbl' : '*.pred2_stopped30_bin',
        'woeiv_tbl' : '*.pred2_stopped30_woeiv',
        'end_tbl' : '*.pred2_stopped30',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_stopped30',
        'save_fig' : True,
        'model_name' : 'gr_stopped30.model'
        }
        
        params_60 = {'target' : 'stopped_60',
        'month_to_target' : '2004',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_stopped60_idx',
        'bin_tbl' : '*.pred2_stopped60_bin',
        'woeiv_tbl' : '*.pred2_stopped60_woeiv',
        'end_tbl' : '*.pred2_stopped60',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_stopped60',
        'save_fig' : True,
        'model_name' : 'gr_stopped60.model'
        }
        
        params_90 = {'target' : 'stopped_90',
        'month_to_target' : '2003',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_stopped90_idx',
        'bin_tbl' : '*.pred2_stopped90_bin',
        'woeiv_tbl' : '*.pred2_stopped90_woeiv',
        'end_tbl' : '*.pred2_stopped90',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_stopped90',
        'save_fig' : True,
        'model_name' : 'gr_stopped90.model'
        }
        
        params_120 = {'target' : 'stopped_120',
        'month_to_target' : '2002',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_stopped120_idx',
        'bin_tbl' : '*.pred2_stopped120_bin',
        'woeiv_tbl' : '*.pred2_stopped120_woeiv',
        'end_tbl' : '*.pred2_stopped120',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_stopped120',
        'save_fig' : True,
        'model_name' : 'gr_stopped120.model'
        }
        
        for p in [params_30, params_90, params_120]:
            modelDriver(input_tbl, 1, \
                  targets, p['target'], except_for_these, p['month_to_target'], \
                  p['encodeString'], p['idx_tbl'], p['n_bin'], p['bin_tbl'], p['woeiv_tbl'], p['end_tbl'], \
                  p['n_feature'], p['iv_lb'], p['iv_ub'], p['split_rate'], p['balThreshold'], p['use_wgt'], \
                  p['fig_name'], p['save_fig'], \
                  p['model_name'])        

    if _trainModel2:
        input_tbl = '*.pred2_surrender_2006'
        targets = ['surrender_30', 'surrender_60', 'surrender_90', 'surrender_120']
        except_for_these =  ['party_id','cmdb_partition']
        
        params_30 = {'target' : 'surrender_30',
        'month_to_target' : '2005',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_surrender30_idx',
        'bin_tbl' : '*.pred2_surrender30_bin',
        'woeiv_tbl' : '*.pred2_surrender30_woeiv',
        'end_tbl' : '*.pred2_surrender30',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_surrender30',
        'save_fig' : True,
        'model_name' : 'gr_surrender30.model'
        }
        
        params_60 = {'target' : 'surrender_60',
        'month_to_target' : '2004',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_surrender60_idx',
        'bin_tbl' : '*.pred2_surrender60_bin',
        'woeiv_tbl' : '*.pred2_surrender60_woeiv',
        'end_tbl' : '*.pred2_surrender60',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_surrender60',
        'save_fig' : True,
        'model_name' : 'gr_surrender60.model'
        }
        
        params_90 = {'target' : 'surrender_90',
        'month_to_target' : '2003',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_surrender90_idx',
        'bin_tbl' : '*.pred2_surrender90_bin',
        'woeiv_tbl' : '*.pred2_surrender90_woeiv',
        'end_tbl' : '*.pred2_surrender90',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_surrender90',
        'save_fig' : True,
        'model_name' : 'gr_surrender90.model'
        }
        
        params_120 = {'target' : 'surrender_120',
        'month_to_target' : '2002',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_surrender120_idx',
        'bin_tbl' : '*.pred2_surrender120_bin',
        'woeiv_tbl' : '*.pred2_surrender120_woeiv',
        'end_tbl' : '*.pred2_surrender120',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_surrender120',
        'save_fig' : True,
        'model_name' : 'gr_surrender120.model'
        }
        
        for p in [params_30, params_90, params_120]:
            modelDriver(input_tbl, 1, \
                  targets, p['target'], except_for_these, p['month_to_target'], \
                  p['encodeString'], p['idx_tbl'], p['n_bin'], p['bin_tbl'], p['woeiv_tbl'], p['end_tbl'], \
                  p['n_feature'], p['iv_lb'], p['iv_ub'], p['split_rate'], p['balThreshold'], p['use_wgt'], \
                  p['fig_name'], p['save_fig'], \
                  p['model_name'])
    
    if _trainModel3:
        input_tbl = '*.pred2_restart_2006'
        targets = ['restart_30', 'restart_60', 'restart_90', 'restart_120']
        except_for_these =  ['party_id','cmdb_partition']
        
        params_30 = {'target' : 'restart_30',
        'month_to_target' : '2005',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_restart30_idx',
        'bin_tbl' : '*.pred2_restart30_bin',
        'woeiv_tbl' : '*.pred2_restart30_woeiv',
        'end_tbl' : '*.pred2_restart30',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_restart30',
        'save_fig' : True,
        'model_name' : 'gr_restart30.model'
        }
        
        params_60 = {'target' : 'restart_60',
        'month_to_target' : '2004',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_restart60_idx',
        'bin_tbl' : '*.pred2_restart60_bin',
        'woeiv_tbl' : '*.pred2_restart60_woeiv',
        'end_tbl' : '*.pred2_restart60',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_restart60',
        'save_fig' : True,
        'model_name' : 'gr_restart60.model'
        }
        
        params_90 = {'target' : 'restart_90',
        'month_to_target' : '2003',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_restart90_idx',
        'bin_tbl' : '*.pred2_restart90_bin',
        'woeiv_tbl' : '*.pred2_restart90_woeiv',
        'end_tbl' : '*.pred2_restart90',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_restart90',
        'save_fig' : True,
        'model_name' : 'gr_restart90.model'
        }
        
        params_120 = {'target' : 'restart_120',
        'month_to_target' : '2002',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_restart120_idx',
        'bin_tbl' : '*.pred2_restart120_bin',
        'woeiv_tbl' : '*.pred2_restart120_woeiv',
        'end_tbl' : '*.pred2_restart120',
        'n_feature' : 100,
        'n_bin' : 10,
        'iv_lb' : 0,
        'iv_ub' : 100,
        'split_rate' : .2,
        'balThreshold' : 1,
        'use_wgt' : False,
        'fig_name' : 'pred2_restart120',
        'save_fig' : True,
        'model_name' : 'gr_restart120.model'
        }
        
        for p in [params_30, params_90, params_120]:
            modelDriver(input_tbl, 1, \
                  targets, p['target'], except_for_these, p['month_to_target'], \
                  p['encodeString'], p['idx_tbl'], p['n_bin'], p['bin_tbl'], p['woeiv_tbl'], p['end_tbl'], \
                  p['n_feature'], p['iv_lb'], p['iv_ub'], p['split_rate'], p['balThreshold'], p['use_wgt'], \
                  p['fig_name'], p['save_fig'], \
                  p['model_name'])
            
    
    if _predictModel1:
        
        print('*** START - scoring new data set ...')
        input_tbl = '*.pred2_stopped_2005'
        targets = ['stopped_30', 'stopped_60', 'stopped_90', 'stopped_120']
        except_for_these =  ['party_id','cmdb_partition']
        
        params_30 = {'model_name' : 'gr_stopped30.model',
        'target' : 'stopped_30',
        'month_to_target' : '2004',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_stopped30_idx_2005',
        'bin_tbl' : '*.pred2_stopped30_bin_2005',
        'woeiv_tbl' : '*.pred2_stopped30_woeiv_2005',
        'end_tbl' : '*.pred2_stopped30_2005',
        'n_bin' : 10
        }
        
        params_60 = {'model_name' : 'gr_stopped60.model',
        'target' : 'stopped_60',
        'month_to_target' : '2003',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_stopped60_idx_2005',
        'bin_tbl' : '*.pred2_stopped60_bin_2005',
        'woeiv_tbl' : '*.pred2_stopped60_woeiv_2005',
        'end_tbl' : '*.pred2_stopped60_2005',
        'n_bin' : 10
        }
        
        params_90 = {'model_name' : 'gr_stopped90.model',
        'target' : 'stopped_90',
        'month_to_target' : '2002',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_stopped90_idx_2005',
        'bin_tbl' : '*.pred2_stopped90_bin_2005',
        'woeiv_tbl' : '*.pred2_stopped90_woeiv_2005',
        'end_tbl' : '*.pred2_stopped90_2005',
        'n_bin' : 10
        }
        
        params_120 = {'model_name' : 'gr_stopped120.model',
        'target' : 'stopped_120',
        'month_to_target' : '2001',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_stopped120_idx_2005',
        'bin_tbl' : '*.pred2_stopped120_bin_2005',
        'woeiv_tbl' : '*.pred2_stopped120_woeiv_2005',
        'end_tbl' : '*.pred2_stopped120_2005',
        'n_bin' : 10
        }
        
        for p in [params_30, params_90, params_120]:
            predictDriver(input_tbl, targets, except_for_these, p['model_name'], p['target'], p['month_to_target'], p['encodeString'], p['idx_tbl'], p['bin_tbl'], p['woeiv_tbl'], p['end_tbl'], p['n_bin'])
        
    if _predictModel2:
        
        print('*** START - scoring new data set ...')
        input_tbl = '*.pred2_surrender_2005'
        targets = ['surrender_30', 'surrender_60', 'surrender_90', 'surrender_120']
        except_for_these =  ['party_id','cmdb_partition']
              
        params_30 = {'model_name' : 'gr_surrender30.model',
        'target' : 'surrender_30',
        'month_to_target' : '2004',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_surrender30_idx_2005',
        'bin_tbl' : '*.pred2_surrender30_bin_2005',
        'woeiv_tbl' : '*.pred2_surrender30_woeiv_2005',
        'end_tbl' : '*.pred2_surrender30_2005',
        'n_bin' : 10
        }
        
        params_60 = {'model_name' : 'gr_surrender60.model',
        'target' : 'surrender_60',
        'month_to_target' : '2003',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_surrender60_idx_2005',
        'bin_tbl' : '*.pred2_surrender60_bin_2005',
        'woeiv_tbl' : '*.pred2_surrender60_woeiv_2005',
        'end_tbl' : '*.pred2_surrender60_2005',
        'n_bin' : 10
        }
        
        params_90 = {'model_name' : 'gr_surrender90.model',
        'target' : 'surrender_90',
        'month_to_target' : '2002',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_surrender90_idx_2005',
        'bin_tbl' : '*.pred2_surrender90_bin_2005',
        'woeiv_tbl' : '*.pred2_surrender90_woeiv_2005',
        'end_tbl' : '*.pred2_surrender90_2005',
        'n_bin' : 10
        }
        
        params_120 = {'model_name' : 'gr_surrender120.model',
        'target' : 'surrender_120',
        'month_to_target' : '2001',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_surrender120_idx_2005',
        'bin_tbl' : '*.pred2_surrender120_bin_2005',
        'woeiv_tbl' : '*.pred2_surrender120_woeiv_2005',
        'end_tbl' : '*.pred2_surrender120_2005',
        'n_bin' : 10
        }
        
        for p in [params_30, params_90, params_120]:
            predictDriver(input_tbl, targets, except_for_these, p['model_name'], p['target'], p['month_to_target'], p['encodeString'], p['idx_tbl'], p['bin_tbl'], p['woeiv_tbl'], p['end_tbl'], p['n_bin'])
    
    if _predictModel3:
        
        print('*** START - scoring new data set ...')
        input_tbl = '*.pred2_restart_2005'
        targets = ['restart_30', 'restart_60', 'restart_90', 'restart_120']
        except_for_these =  ['party_id','cmdb_partition']
        
        params_30 = {'model_name' : 'gr_restart30.model',
        'target' : 'restart_30',
        'month_to_target' : '2004',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_restart30_idx_2005',
        'bin_tbl' : '*.pred2_restart30_bin_2005',
        'woeiv_tbl' : '*.pred2_restart30_woeiv_2005',
        'end_tbl' : '*.pred2_restart30_2005',
        'n_bin' : 10
        }
        
        params_60 = {'model_name' : 'gr_restart60.model',
        'target' : 'restart_60',
        'month_to_target' : '2003',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_restart60_idx_2005',
        'bin_tbl' : '*.pred2_restart60_bin_2005',
        'woeiv_tbl' : '*.pred2_restart60_woeiv_2005',
        'end_tbl' : '*.pred2_restart60_2005',
        'n_bin' : 10
        }
        
        params_90 = {'model_name' : 'gr_restart90.model',
        'target' : 'restart_90',
        'month_to_target' : '2002',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_restart90_idx_2005',
        'bin_tbl' : '*.pred2_restart90_bin_2005',
        'woeiv_tbl' : '*.pred2_restart90_woeiv_2005',
        'end_tbl' : '*.pred2_restart90_2005',
        'n_bin' : 10
        }
        
        params_120 = {'model_name' : 'gr_restart120.model',
        'target' : 'restart_120',
        'month_to_target' : '2001',
        'encodeString' : True,
        'idx_tbl' : '*.pred2_restart120_idx_2005',
        'bin_tbl' : '*.pred2_restart120_bin_2005',
        'woeiv_tbl' : '*.pred2_restart120_woeiv_2005',
        'end_tbl' : '*.pred2_restart120_2005',
        'n_bin' : 10
        }
        
        for p in [params_30, params_90, params_120]:
            predictDriver(input_tbl, targets, except_for_these, p['model_name'], p['target'], p['month_to_target'], p['encodeString'], p['idx_tbl'], p['bin_tbl'], p['woeiv_tbl'], p['end_tbl'], p['n_bin'])
    
    print('*** Finished at *** : ' + format(datetime.today(), '%Y-%m-%d %H:%M:%S \n\n'))
    spark.stop()

def predictDriver(input_tbl, targets, except_for_these, model_name, target, month_to_target, encodeString, \
                      idx_tbl, bin_tbl, woeiv_tbl, end_tbl, n_bin):
    '''
    Load the trained model to score a new data set. Save a prediction table.
    '''
    
    targets.remove(target)
    ds = spark.table(input_tbl).drop(*targets).filter((F.col('403b_k12_policies') > 0) & (F.col('cmdb_partition') == month_to_target))
    processSample(ds, except_for_these, target, n_bin, idx_tbl, bin_tbl, woeiv_tbl, end_tbl, encodeString)
    
    ## Load the model and features to score new data set
    clf = joblib.load(model_loc + model_name)
    
    with open(model_loc + target + '_feature_list', 'rb') as r:
         clf_features = pickle.load(r)
    print('*** Model ' + model_name + ' is loaded...')
    
    ## Score the new data 
    print('\n*** *********************************** ***')
    print('*** Model Performance against new data set ' + input_tbl + ' ***')
    print('*** *********************************** ***')
    
    meta_data = spark.table(input_tbl).filter(F.col('cmdb_partition') == month_to_target).toPandas()
    features = spark.table(end_tbl).select(*clf_features).toPandas()
    actual_target = spark.table(end_tbl).select(target).toPandas()
    predict_target = clf.predict(features)
    
    score = clf.score(features, actual_target)
    print('Score: {0:.2f} %'.format(100 * score))
    
    f1_test = f1_score(predict_target, actual_target)
    print('Test set F1 score : {0:.2f}'.format(100*f1_test))
    
    fpr, tpr, _ = roc_curve(np.array(actual_target), clf.predict_proba(features)[:,1])
    auc_test = auc(fpr,tpr)
    print('Test set AUC : {0:.2f}'.format(auc_test))
    
    print('\n*** Confusion matrix ***')
    print(confusion_matrix(np.array(actual_target), predict_target))    
    
    predictions = pd.DataFrame({'pred_prob_0':clf.predict_proba(features)[:,0], 'pred_prob_1':clf.predict_proba(features)[:,1]})
    new_target = pd.DataFrame({'predicted_target':predict_target})
    meta_pid = meta_data[[target,'party_id']]
    df_test_prediction = pd.concat([meta_pid, predictions, new_target], axis=1)
    
    df_test_pred = spark.createDataFrame(df_test_prediction)
    df_test_pred.write.mode('overwrite').saveAsTable(input_tbl + '_prediction')
    
    print('*** END - New scores, ' + input_tbl + '_prediction, are saved for ' + input_tbl)
        
def modelDriver(input_tbl, undersample_rate, \
              targets, target, except_for_these, month_to_target, \
              encodeString, idx_tbl, n_bin, bin_tbl, woeiv_tbl, end_tbl, \
              n_feature, iv_lb, iv_ub, split_rate, balThreshold, use_wgt, \
              fig_name, save_fig, \
              model_name):
    '''
    1. checkImbalance
    2. processSample
    3. pickVars
    4. classifer
    5. joblib.dump
    '''
    print('\n****************** Training a model ******************') 
    ds = checkImbalance(input_tbl, 1, balThreshold, use_wgt, targets, target, month_to_target)
    processSample(ds, except_for_these, target, n_bin, idx_tbl, bin_tbl, woeiv_tbl, end_tbl, encodeString)
    
    ## Feature selection - stepwise       
    import feature_selection as fs
    
    final_vars = pickVars(woeiv_tbl, bin_tbl, except_for_these + [target], iv_lb, iv_ub, n_feature)
    df = (spark.table(end_tbl).toPandas())
    X = df[[x + '_woe' for x in final_vars]]
    y = df[target]
    features = fs.stepwise_selection(X, y)
    print('resulting features:' + str(features))
    
    ## Export features to a pickle
    with open(model_loc + target + '_feature_list','wb') as w:
        pickle.dump(features, w)
        
    print('*** Influence of predictors with regard to target variable ***')
    ## '_woe' = 4
    influence = (spark.table(woeiv_tbl).filter(F.col('feat_name').isin([x[:-4] for x in features]))
    .select('feat_name', 'bucket', 'pct_event').orderBy(['feat_name','pct_event'], ascending=[0,0])
    )
    influence.show(2000, truncate=False)    
    
    for f in [x[:-4] for x in features]:
        if '_indexed' in f:
            sql = 'select ' + f + '_bucket, min(' + f[:-8] + ') as lb, max(' + f[:-8] + ') as ub from ' + bin_tbl + ' group by 1 order by 1'
        else:
            sql = 'select ' + f + '_bucket, min(' + f + ') as lb, max(' + f + ') as ub from ' + bin_tbl + ' group by 1 order by 1'
        spark.sql(sql).show(truncate=False)

    clf = classifier(end_tbl, target, except_for_these, features, split_rate, fig_name, save_fig)
    
    ## Save the model
    joblib.dump(clf, model_loc + model_name)
    print('*** Saved a trained model... ***')
    
def checkImbalance(input_tbl, undersample_rate, balThreshold, use_wgt, targets, target, cmdb):
    
    print('*** Checks imbalance in data...')
    targets.remove(target)
    df = spark.table(input_tbl).drop(*targets)
    
    print('*** Distribution by target value ***' + target)
    ### 403b_k12_policies > 0
    ds = df.filter((F.col('403b_k12_policies') > 0) & (F.col('cmdb_partition') == cmdb))
    ds.groupBy(target).count().show()
    
     ## Check imbalance - weighting for the minority class (Tp)
    total_size = float(ds.count())
    tp_size = float(ds.select(target).filter(F.col(target) == 1).count())
    tn_size = float(total_size - tp_size)
    if tp_size/tn_size < balThreshold:
        print("Imbalance issue exists....target vs non-target ratio : " + str(tp_size/tn_size))
        if use_wgt:
            class_weight = tn_size / total_size
            ds = ds.withColumn('classWeight', F.when(F.col(target) == 0, class_weight).otherwise(1-class_weight))
        else:
            pass
    else:
        pass
    
    print('*** Undersampling major class (non-event) due to imbalanced data ***')
    base_tp = ds.filter(F.col(target) == 1)
    _count = base_tp.count()
    
    print('*** Non-event target size :' + str(_count*undersample_rate))
    base_tn = (ds.filter(F.col(target) == 0)
            .withColumn('rand_num', F.lit(F.rand(_seed)))
            .withColumn('row_num', F.row_number().over(W.orderBy('rand_num')))
            .filter(F.col('row_num') <= _count*undersample_rate)
            .drop('rand_num', 'row_num'))
    df = base_tp.unionByName(base_tn)
    
    return df

        
def processSample(ds, exclusion, target, n_bin, idx_tbl, bin_tbl, woeiv_tbl, result_tbl, encodeString = True):
    '''
    bucketize, calculate WOE/IV, apply WOE
    exclusion : nominal columns such as party_id and cmdb_partition
    '''
    print('*** Start of Processing ' + target + ' - bucketize, calculate WOE/IV, apply WOE *** ')

    ## Indexing
    if encodeString:
        df = indexCategorical(ds, exclusion, target)
        df.write.mode('overwrite').saveAsTable(idx_tbl)
        print('*** Indexing is completed...')
    
    ## WOE/IV
    df = spark.table(idx_tbl)
    decoded_cols = bucketize(df, target, exclusion, n_bin, bin_tbl)
    calculateIV(bin_tbl, target, decoded_cols, woeiv_tbl)
    useWOE(bin_tbl, exclusion + [target], woeiv_tbl, result_tbl)
    
    print('*** End of Processing ***')
    
 
def classifier(tbl, target, except_for_these, features, split, filename,_save_fig):
    '''
    target : dependent
    except_for_these : nominal - party_id, cmdb_partition
    features : selected features
    '''

    df_spark = spark.table(tbl).select(target, *except_for_these, *features)
    
    df_model = df_spark.toPandas()
    X_train_meta, y_train, X_test_meta, y_test = splitDataset(df_model,target,except_for_these, split)       
    X_train = X_train_meta[X_train_meta.columns.difference(except_for_these)]
    X_test = X_test_meta[X_test_meta.columns.difference(except_for_these)]
    ## Classifiers
    print('\n*** Fits a model ***')
    clf = LogisticRegression(class_weight='balanced', max_iter=20000,solver='lbfgs',random_state=_seed)
        
#    from sklearn.ensemble import RandomForestClassifier
#    clf = RandomForestClassifier(criterion='entropy', class_weight='balanced', max_features='auto', random_state=_seed, n_jobs=-1)
        
    ## Gridsearch to tune hyperparameters
    from sklearn.model_selection import GridSearchCV
    param_grid = {
                    'C':[.5,.55,.6]
                }
#    param_grid = {
#                    'n_estimators':list(range(250,300,10))
#                    ,'max_depth': list(range(50,60,1))
#                }
    
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, error_score=0)
    grid_search.fit(X_train, y_train)
            
    print('*** Best parameters....')
    print(grid_search.best_params_)
    best_clf = grid_search.best_estimator_

    ## Cross Validation
    fitModelCV(10, best_clf, except_for_these, X_train_meta, y_train, X_test_meta, y_test,filename)
        
    ## Plot graphics - Life and Gain
    scores_train = scoring(X_train,best_clf,y_train)
    scores_test = scoring(X_test,best_clf,y_test)
    
    print('*** Deciles - Train ***')
    decile_train = deciling(scores_train,['DECILE'],'TARGET','NONTARGET')
    print(decile_train)
    
    print('*** Deciles - Test ***')
    decile_test = deciling(scores_test,['DECILE'],'TARGET','NONTARGET')
    print(decile_test)
    
    lift_train = pd.concat([X_train,scores_train],axis=1)
    lift_test = pd.concat([X_test,scores_test],axis=1)
    
    ## Lift curves separately
#    print('*** Life curve for train set is saved, lift_train.jpg...')
#    gains(lift_train,['DECILE'],'TARGET','PROB', './fig/lift_train' + filename + '.jpg')
#    
#    print('*** Life curve for test set is saved, lift_test.jpg...')
#    gains(lift_test,['DECILE'],'TARGET','PROB', './fig/lift_test' + filename + '.jpg')
    
    ## Train and test combined
    lift_train.loc[:,'ind']='train'
    lift_test.loc[:,'ind']='test'
    lift_merged = pd.concat([lift_train, lift_test], ignore_index=True)
    
    print('*** Life curves for train and test sets are saved...')
    gains(lift_merged, ['DECILE'], 'TARGET', 'PROB', './fig/lift_train_test_' + filename + '.jpg')
    
    return best_clf

    
def pickVars(woe_iv,bucketed,exclusion, lb, ub, n_feat):
    '''
    Select features based on IV
    exclusion : target + except_for_these
    '''
    woe_iv = spark.table(woe_iv)    
    
    print('*** Feature importance by Information Value, Top ' + str(n_feat) + ' features ***')
    ivCalc = woe_iv.groupby('feat_name').agg(F.sum('iv').alias('iv')).orderBy('iv', ascending=False)
    ivCalc.show(n_feat, truncate=False)
    
    print('\n*** Selected features with IV between ' + str(lb)+ ' and ' + str(ub) + ' ***')
    select_feat_list = (woe_iv.groupby('feat_name').agg(F.sum('iv').alias('iv')).orderBy('iv', ascending=False)
    .filter((F.col('iv') >= lb) & (F.col('iv') <= ub)).select('feat_name').rdd.flatMap(lambda x:x).collect()
    )

#    df_bucketed = spark.table(bucketed)
#    df_bucketed.persist()        
#    df_model = df_bucketed.select(*exclusion, *[x+'_woe' for x in select_feat_list])
#    df_model.persist()
#    df_model.write.mode('overwrite').saveAsTable(df_final_set)
#    print('\n*** Finished creating a final data set with WOE for fitting...')
#    df_model.unpersist()
#    df_bucketed.unpersist()

    return select_feat_list

def useWOE(df_b,except_for_these,woe,result):
    
    df = spark.table(df_b)
    df_bucketed = df.select(*except_for_these, *[x for x in df.columns if x.endswith('_bucket')])
    
    woe_iv = spark.table(woe)
    woe_iv.persist()
    df_bucketed.persist() 
        
    print('*** Replace values with WOEs ***')    
    woe_list = [row.asDict() for row in woe_iv.select('feat_name','bucket','woe').collect()]
    def woe_mapper(feat, bucket):
        for d in woe_list:
            if d['feat_name'] == feat and d['bucket'] == bucket:
                return d['woe']
    
    woe_mapper_udf = F.udf(woe_mapper, DoubleType())
    for c in df_bucketed.columns:
        if c.endswith('_bucket'):
            df_bucketed = df_bucketed.withColumn(c.replace('_bucket','_woe'), F.lit(woe_mapper_udf(F.lit(c[:-len('_bucket')]), F.col(c).cast('string'))))
    
    df_bucketed2 = df_bucketed.drop(*[x for x in df_bucketed.columns if x.endswith('_bucket')])
    df_bucketed2.write.mode('overwrite').saveAsTable(result)
    print('\n*** Saved df_bucketed having WOEs ***')
#    df_bucketed2.show(truncate=False)
    
    woe_iv.unpersist()
    df_bucketed.unpersist()

def calculateIV(df_b,target, decoded_cols, woeiv_tbl):
    df_bucketed = spark.table(df_b)
    
    print('*** Calculating WOE and IV ....***')
    woe_iv = pd.DataFrame([], columns=['feat_name','bucket','event_non_event','event','tot_event','nonevent','tot_nonevent','pct_event','pct_nonevent','woe','iv'])    
    
    d = df_bucketed.select(target, *[x+'_bucket' for x in decoded_cols], *decoded_cols)
    d.persist()
    
    for c in decoded_cols:
        df_woe = (d.withColumn('feat_name', F.lit(c))
                  .withColumn('bucket', F.col(c+'_bucket').cast('string'))
                  .select('*', F.count('*').over(W.partitionBy('feat_name')).alias('feat_total'), \
                          F.count('*').over(W.partitionBy('feat_name', 'bucket')).alias('event_non_event'), \
                          F.sum(target).over(W.partitionBy('feat_name', 'bucket')).alias('event'))
                  .withColumn('nonevent', F.col('event_non_event')-F.col('event'))
                  .withColumn('tot_event', F.sum(target).over(W.partitionBy('feat_name')))
                  .withColumn('tot_nonevent', F.col('feat_total')-F.col('tot_event'))
                  .withColumn('pct_event', F.when(F.col('event')==0, (.5/F.col('tot_event'))).otherwise(F.col('event')/F.col('tot_event')))
                  .withColumn('pct_nonevent', F.when(F.col('nonevent')==0, (.5/F.col('tot_nonevent'))).otherwise(F.col('nonevent')/F.col('tot_nonevent')))
                  .withColumn('woe', F.log(F.col('pct_nonevent')/F.col('pct_event')))
                  .withColumn('iv', F.col('woe')*(F.col('pct_nonevent')-F.col('pct_event')))
                  .select('feat_name','bucket','event_non_event','event','tot_event','nonevent','tot_nonevent','pct_event','pct_nonevent','woe','iv')
                  .distinct()
                  .orderBy('feat_name', 'bucket'))
        woe_iv = woe_iv.append(df_woe.persist().toPandas(), ignore_index=True, sort=False)
        df_woe.unpersist()
    d.unpersist()
    
    woe_iv = spark.createDataFrame(woe_iv)
#    woe_iv.show(truncate=False)
    woe_iv.write.mode('overwrite').saveAsTable(woeiv_tbl)
    print('*** WOE/IV Table is saved *** ' + woeiv_tbl)
   
def bucketize(df, target, except_for_these, n_bin, df_b):
    df.persist()    
    print('*** Start of classing ***')
    decoded_cols = []
    col_decile = {}
    for c in df.drop(*except_for_these, target).dtypes:
#        if c[1] == 'string':
#            min_ = float(df.select(c[0]+'_indexed').summary('min').collect()[0][1])
#            max_ = float(df.select(c[0]+'_indexed').summary('max').collect()[0][1])
#            if min_ < max_ and (max_ - min_ != 0):
#                print(c[0] + str(np.linspace(min_, max_, 11).tolist()))
#                cat += [c[0]]
#                col_decile.setdefault(c[0], []).append(np.linspace(min_, max_, 11).tolist())
        if c[0].endswith('_indexed') or c[1] not in ('string', 'timestamp'):
            min_ = float(df.select(c[0]).summary('min').collect()[0][1])
            max_ = float(df.select(c[0]).summary('max').collect()[0][1])
            if min_ < max_ and (max_ - min_ != 0):
                decoded_cols += [c[0]]
                col_decile.setdefault(c[0], []).append(np.linspace(min_, max_, n_bin+1).tolist())
    
    ### All variables in buckets
    buckets = [Bucketizer(splits=col_decile[c][0], inputCol = c, outputCol = c+'_bucket', handleInvalid='keep') for c in decoded_cols]
    pipeline = Pipeline(stages=buckets)
    df_bucketed = pipeline.fit(df).transform(df)
    df_bucketed.persist()
    df_bucketed.write.mode('overwrite').saveAsTable(df_b)
    print('*** End of classing ***')
#    print('*** Decoded columns ***')
#    print(str(decoded_cols))
    
    df.unpersist()
    df_bucketed.unpersist()
    
    return decoded_cols

def indexCategorical(df,except_for_these, target):
    '''
    except_for_these : columns to exclude (list)
    target : target variable (str)
    '''
    print('*** Start of indexing string variables...')
    indexer = [StringIndexer(inputCol=s[0], outputCol=s[0]+'_indexed',handleInvalid='keep') for s in df.drop(*except_for_these, target).dtypes if s[1] == 'string']
#    encoder = [OneHotEncoderEstimator(inputCols=[s[0]+'_indexed'],outputCols=[s[0]+"_encoded"],handleInvalid='keep') for s in df.drop(*except_for_these, target).dtypes if s[1] == 'string']
    pipeline = Pipeline(stages=indexer)
    df2 = pipeline.fit(df).transform(df)
#    df2.show(truncate=False)
    print('*** End of indexing string variables...')
    return df2     


def splitDataset(df_model,target,except_for_these, split):
#    X = df_model[df_model.columns.difference(except_for_these)]
    train, test = train_test_split(df_model, test_size = split, random_state=_seed, stratify = df_model[[target]] )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    X_train = train[train.columns.difference([target])]
    y_train = train[target]
    X_test = test[test.columns.difference([target])]
    y_test = test[target]
    
    print('Train set : %d'%len(y_train))
    print('Test set : %d'%len(y_test))
    return X_train, y_train, X_test, y_test

def scoring(features,clf,target):
    score = pd.DataFrame(clf.predict_proba(features)[:,1], columns = ['PROB'])
    score['TARGET'] = target
    score['NONTARGET'] = 1 - target
    score['DECILE'] = pd.qcut(score['PROB'].rank(method = 'first'),10,labels=range(10,0,-1)).astype(float)
    return score

def fitModelCV(n_splits, clf, except_for_these, X_train_meta, y_train, X_test_meta, y_test,filename):
    # CV Model performance
#    scores = cross_val_score(clf, X_train,y_train,cv=n_splits, n_jobs=-1)
#    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#    scores = cross_val_score(clf, X_train,y_train,cv=n_splits, n_jobs=-1, scoring='f1')
#    print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    X_train = X_train_meta[X_train_meta.columns.difference(except_for_these)]
    X_test = X_test_meta[X_test_meta.columns.difference(except_for_these)]
    
    # plot ROC/AUC per CV fold
    plt.figure(1,figsize=(20, 5))
    fig, ax = plt.subplots()
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        clf.fit(X_train.iloc[train], y_train.iloc[train])
        fpr, tpr, _ = roc_curve(y_train.iloc[test], clf.predict_proba(X_train.iloc[test])[:,1])
        roc_auc = auc(fpr,tpr)
        ax.plot(fpr, tpr, lw=2, label='ROC curve for %d (AUC = %0.3f)'%(i, roc_auc))
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',  label='Random Guess', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[0, 1.05], ylim=[0, 1.05], title="ROC Curves with CV", xlabel='FPR', ylabel='TPR')
    ax.legend(loc="lower right")
    plt.show()
    plt.savefig('./fig/roc_'+filename+'.jpg')
    plt.close()
    
    # Final Model performance details
    print('\n*** *********************************** ***')
    print('*** Model Performance against train set ***')
    print('*** *********************************** ***')
    pred_train = clf.predict(X_train)
    accuracy_train = accuracy_score(pred_train,y_train)
    print('Train set accuracy : {}'.format(100*accuracy_train))
        
    f1_train = f1_score(pred_train,y_train)
    print('Train set F1 score : {}'.format(100*f1_train))
    
    fpr, tpr, _ = roc_curve(np.array(y_train), clf.predict_proba(X_train)[:,1])
    auc_train = auc(fpr,tpr)
    print('Train set AUC : {}'.format(auc_train))
    
    print('\n*** Confusion matrix - train set')
    print(pd.crosstab(y_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED']))
    
    import math
    coeff_mtx = pd.DataFrame(clf.coef_).T
    coeff_mtx.rename(columns={0:'coeff'}, inplace=True)
    coeff_mtx['feature']=X_train.columns
    coeff_mtx['odds']=coeff_mtx['coeff'].apply(lambda x: math.exp(x))
    coeff_mtx['prob_favor']=coeff_mtx['odds'].apply(lambda x: x / (1+x))
    coeff_mtx['prob_against']=coeff_mtx['odds'].apply(lambda x: 1 / (1+x))
    
    coeff_mtx2 = spark.createDataFrame(coeff_mtx)
    coeff_mtx2.write.mode('overwrite').saveAsTable('*.pred2_coeff_mtx_'+filename)
#    coeff_mtx2.show(2000, truncate=False)
    print('*** Coefficient/Odds ratio set is saved....')
    
    print('\n*** *********************************** ***')
    print('*** Model Performance against test set ***')
    print('*** *********************************** ***')
    pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(pred_test,y_test)
    print('Test set accuracy : {}'.format(100*accuracy_test))
    
    f1_test = f1_score(pred_test,y_test)
    print('Test set F1 score : {}'.format(100*f1_test))
    
    fpr, tpr, _ = roc_curve(np.array(y_test), clf.predict_proba(X_test)[:,1])
    auc_test = auc(fpr,tpr)
    print('Test set AUC : {}'.format(auc_test))
    
    print('\n*** Confusion matrix - test set')
    print(pd.crosstab(y_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED']))
    
#    predictions = pd.DataFrame({'pred_prob_0':clf.predict_proba(X_test)[:,0], 'pred_prob_1':clf.predict_proba(X_test)[:,1]})
#    new_target = pd.DataFrame({'predicted_target':pred_test})
#    df_test_prediction = pd.concat([X_test_meta,y_test,predictions, new_target], axis=1)
    
#    df_test_pred = spark.createDataFrame(df_test_prediction)
#    df_test_pred.write.mode('overwrite').saveAsTable('*.pred2_prediction_'+filename)
#    df_test_pred.show(truncate=False)
#    print('*** Prediction set is saved....')


# =============================================================================
# Visualization helper functions
# =============================================================================
def plots_org(agg1,target,graph_type,filename):
    plt.figure(1,figsize=(20, 5))

    plt.subplot(131)
    plt.plot(agg1['DECILE'],agg1['ACTUAL'],label='Actual')
    plt.plot(agg1['DECILE'],agg1['PRED'],label='Pred')
    plt.xticks(range(10,110,10))
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.title('Actual vs Predicted', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + " " + str(graph_type) + " %",fontsize=15)

    plt.subplot(132)
    X = agg1['DECILE'].tolist()
    X.append(0)
    Y = agg1['DIST_TAR'].tolist()
    Y.append(0)
    plt.plot(sorted(X),sorted(Y))
    plt.plot([0, 100], [0, 100],'r--')
    plt.xticks(range(0,110,10))
    plt.yticks(range(0,110,10))
    plt.grid(True)
    plt.title('Gain Chart', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + str(" DISTRIBUTION") + " %",fontsize=15)
    plt.annotate(round(agg1[agg1['DECILE'] == 30].DIST_TAR.item(),2),xy=[30,30], 
            xytext=(25, agg1[agg1['DECILE'] == 30].DIST_TAR.item() + 5),fontsize = 13)
    plt.annotate(round(agg1[agg1['DECILE'] == 50].DIST_TAR.item(),2),xy=[50,50], 
            xytext=(45, agg1[agg1['DECILE'] == 50].DIST_TAR.item() + 5),fontsize = 13)

    plt.subplot(133)
    plt.plot(agg1['DECILE'],agg1['LIFT'])
    plt.xticks(range(10,110,10))
    plt.grid(True)
    plt.title('Lift Chart', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel("Lift",fontsize=15)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
    
def gains_org(data,decile_by,target,score,filename):
    inputs = list(decile_by)
    inputs.extend((target,score))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['ACTUAL'] = grouped.mean()[target]*100
    agg1['PRED'] = grouped.mean()[score]*100
    agg1['DIST_TAR'] = grouped.sum()[target].cumsum()/grouped.sum()[target].sum()*100
    agg1.index.name = 'DECILE'
    agg1 = agg1.reset_index()
    agg1['DECILE'] = agg1['DECILE']*10
    agg1['LIFT'] = agg1['DIST_TAR']/agg1['DECILE']
    plots(agg1,target,'Distribution', filename)

def plots(agg1,target,graph_type,filename):
    plt.figure(1,figsize=(20, 5))

    plt.subplot(131)
    plt.plot(agg1['DECILE_TRAIN'],agg1['ACTUAL_TRAIN'],label='Actual_train')
    plt.plot(agg1['DECILE_TRAIN'],agg1['PRED_TRAIN'],label='Pred_train')
    plt.plot(agg1['DECILE_TEST'],agg1['ACTUAL_TEST'],label='Actual_test')
    plt.plot(agg1['DECILE_TEST'],agg1['PRED_TEST'],label='Pred_test')
    plt.xticks(range(10,110,10))
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.title('Actual vs Predicted', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + " " + str(graph_type) + " %",fontsize=15)

    plt.subplot(132)
    X = agg1['DECILE_TRAIN'].tolist()
    X.append(0)
    Y = agg1['DIST_TAR_TRAIN'].tolist()
    Y.append(0)
    
    X2 = agg1['DECILE_TEST'].tolist()
    X2.append(0)
    Y2 = agg1['DIST_TAR_TEST'].tolist()
    Y2.append(0)
    
    plt.plot(sorted(X),sorted(Y),label='Train')
    plt.plot(sorted(X2),sorted(Y2),label='Test')
    plt.plot([0, 100], [0, 100],'r--')
    plt.xticks(range(0,110,10))
    plt.yticks(range(0,110,10))
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.title('Gain Chart', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + str(" DISTRIBUTION") + " %",fontsize=15)
    plt.annotate(round(agg1[agg1['DECILE_TRAIN'] == 30].DIST_TAR_TRAIN.item(),2),xy=[30,30], 
            xytext=(25, agg1[agg1['DECILE_TRAIN'] == 30].DIST_TAR_TRAIN.item() + 5),fontsize = 13)
    plt.annotate(round(agg1[agg1['DECILE_TRAIN'] == 50].DIST_TAR_TRAIN.item(),2),xy=[50,50], 
            xytext=(45, agg1[agg1['DECILE_TRAIN'] == 50].DIST_TAR_TRAIN.item() + 5),fontsize = 13)


    plt.annotate(round(agg1[agg1['DECILE_TEST'] == 30].DIST_TAR_TEST.item(),2),xy=[30,30], 
            xytext=(25, agg1[agg1['DECILE_TEST'] == 30].DIST_TAR_TEST.item() + 5),fontsize = 13)
    plt.annotate(round(agg1[agg1['DECILE_TEST'] == 50].DIST_TAR_TEST.item(),2),xy=[50,50], 
            xytext=(45, agg1[agg1['DECILE_TEST'] == 50].DIST_TAR_TEST.item() + 5),fontsize = 13)

    plt.subplot(133)
    plt.plot(agg1['DECILE_TRAIN'],agg1['LIFT_TRAIN'],label='Train')
    plt.plot(agg1['DECILE_TEST'],agg1['LIFT_TEST'],label='Test')
    plt.xticks(range(10,110,10))
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.title('Lift Chart', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel("Lift",fontsize=15)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
    
def gains(data,decile_by,target,score,filename):
    inputs = list(decile_by) + ['ind']
    inputs.extend((target,score))
    decile = data[inputs]
    
    train_ = decile[decile["ind"]=='train']
    test_ = decile[decile["ind"]=='test']
    
    grouped_train = train_.groupby(decile_by)
    grouped_test = test_.groupby(decile_by)
    
    agg1 = pd.DataFrame({},index=[])
    agg1['ACTUAL_TRAIN'] = grouped_train.mean()[target]*100
    agg1['PRED_TRAIN'] = grouped_train.mean()[score]*100
    agg1['DIST_TAR_TRAIN'] = grouped_train.sum()[target].cumsum()/grouped_train.sum()[target].sum()*100
    agg1.index.name = 'DECILE_TRAIN'
    agg1 = agg1.reset_index()
    agg1['DECILE_TRAIN'] = agg1['DECILE_TRAIN']*10
    agg1['LIFT_TRAIN'] = agg1['DIST_TAR_TRAIN']/agg1['DECILE_TRAIN']
    
    agg1['ACTUAL_TEST'] = grouped_test.mean()[target]*100
    agg1['PRED_TEST'] = grouped_test.mean()[score]*100
    agg1['DIST_TAR_TEST'] = grouped_test.sum()[target].cumsum()/grouped_test.sum()[target].sum()*100
    agg1.index.name = 'DECILE_TEST'
    agg1 = agg1.reset_index()
    agg1['DECILE_TEST'] = agg1['DECILE_TEST']*10
    agg1['LIFT_TEST'] = agg1['DIST_TAR_TEST']/agg1['DECILE_TEST']
    
    plots(agg1,target,'Distribution', filename)

def deciling(data,decile_by,target,nontarget):
    inputs = list(decile_by)
    inputs.extend((target,nontarget))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['TOTAL'] = grouped.sum()[nontarget] + grouped.sum()[target]
    agg1['TARGET'] = grouped.sum()[target]
    agg1['NONTARGET'] = grouped.sum()[nontarget]
    agg1['PCT_TAR'] = grouped.mean()[target]*100
    agg1['CUM_TAR'] = grouped.sum()[target].cumsum()
    agg1['CUM_NONTAR'] = grouped.sum()[nontarget].cumsum()
    agg1['DIST_TAR'] = agg1['CUM_TAR']/agg1['TARGET'].sum()*100
    agg1['DIST_NONTAR'] = agg1['CUM_NONTAR']/agg1['NONTARGET'].sum()*100
    agg1['SPREAD'] = (agg1['DIST_TAR'] - agg1['DIST_NONTAR'])
    return agg1

if __name__ == '__main__':
    main()
