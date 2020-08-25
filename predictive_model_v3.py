#!/usr/bin/env python
"""
Created on Tue Aug 11 10:23:47 2020
@author: Bryan Kim
@title: Predictive model using Weight of Evidence
@run:
    for training: sh pykicker.sh gr_model_v3.py(0) train(1) gr_stop.json(2)
    for predicting: sh pykicker.sh gr_model_v3.py predict gr_stop_predict.json
    
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
import pandas as pd, numpy as np, pickle, matplotlib.pyplot as plt, sys, json, math
from scipy import interp
from datetime import datetime
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
from sklearn.externals import joblib
from sklearn.utils import resample

## Custom auxiliary functions
import feature_selection as fs
import plot_graphs as vis

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def predictDriver(input_tbl, targets, except_for_these, model_name, target, month_to_target, encodeString, \
                      idx_tbl, bin_tbl, woeiv_tbl, end_tbl, n_bin):
    '''
    Load the trained model to score a new data set. Save a prediction table.
    '''
    print('*** START - scoring new data set ...')
    processSample(input_tbl, except_for_these, targets, target, n_bin, idx_tbl, bin_tbl, woeiv_tbl, end_tbl, month_to_target, encodeString)
    
    ## Load the model and features to score new data set
    clf = joblib.load(model_loc + model_name)
    
    with open(model_loc + target + '_feature_list', 'rb') as r:
         clf_features = pickle.load(r)
    print('*** Model ' + model_name + ' is loaded...')
    
    ## Score the new data 
    print('\n*** *********************************** ***')
    print('*** Model Performance against new data set ' + input_tbl + ' ***')
    print('*** *********************************** ***')
        
    tmp = spark.table(end_tbl).persist()
    cols = tmp.drop(target, *except_for_these).columns

    for f in clf_features:
        if f not in cols:
            tmp = tmp.withColumn(f, F.lit(0.0))
        else:
            pass
    
    features = tmp.select(*clf_features).toPandas()
    actual_target = tmp.select(target).toPandas()
    predict_target = clf.predict(features)
    
    score = clf.score(features, actual_target)
    print('Score: {0:.3f} %'.format(100 * score))
   
#    f1_test = f1_score(actual_target, predict_target)
#    print('F1 score : {0:.3f}'.format(f1_test))
    
    fpr, tpr, thresholds = roc_curve(actual_target, clf.predict_proba(features)[:,1], pos_label=1)
    
    ## Thresholding via Cost
    ## FP (retention campaign fees) vs FN (loss of contributions) - which is more expensive?
    def thresholdingCost(fpr, tpr, thresholds, actual_target, filename):
        '''
        FP (retention campaign fees) vs FN (loss of contributions) - which is more expensive?
        '''
        print('*** Best threshold minimizing the costs for FP and FN ***')
        fnr = 1 - tpr

        fp_cost = 1
        fn_cost = 10

        costs = (fpr * fp_cost + fnr * fn_cost) * actual_target.size
        best = np.argmin(costs)
        best_cost = np.round(costs[best])
        best_threshold = np.round(thresholds[best], 3)

        print('Best FPR : ' + str(fpr[best]))
        print('Best TPR : ' + str(tpr[best]))

        plt.ticklabel_format(style='plain')
        plt.plot(thresholds, costs, label = 'cost ($)')
        label = 'best cost: {} at threshold {}'.format(best_cost, best_threshold)
        plt.axvline(best_threshold, linestyle = '--', lw = 2, color = 'black', label = label)
        plt.xlabel('threshold')  
        plt.ylabel('$')  
        plt.title('Cost as a Function of Threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_loc + 'cost_threshold_'+filename+'.jpg')
        plt.show()
    
    thresholdingCost(fpr, tpr, thresholds, actual_target, fig_name)
    
    auc_test = engine.auc(fpr,tpr)
    print('** AUC : {0:.3f}'.format(auc_test))
    
    print('\n*** Confusion matrix ***')
    print(confusion_matrix(np.array(actual_target), predict_target))    
    
    predictions = pd.DataFrame({'pred_prob_0':clf.predict_proba(features)[:,0], 'pred_prob_1':clf.predict_proba(features)[:,1]})
    predictions['decile'] = pd.qcut(predictions['pred_prob_1'].rank(method = 'first'), 10, labels = range(10,0,-1)).astype(float)
    new_target = pd.DataFrame({'predicted_target':predict_target})
    meta_pid = meta_data[[target,'party_id']]
    df_test_prediction = pd.concat([meta_pid, predictions, new_target], axis=1)
    
    tmp.unpersist()
    df_test_pred = spark.createDataFrame(df_test_prediction)
    df_test_pred.write.mode('overwrite').saveAsTable(input_tbl + '_prediction_' + target)
    
    print('*** END - New scores, ' + input_tbl + '_prediction_' + target + ', are saved for ' + input_tbl)
        
def modelDriver(input_tbl, targets, target, except_for_these, month_to_target, \
              encodeString, idx_tbl, n_bin, bin_tbl, woeiv_tbl, end_tbl, \
              threshold_in, threshold_out, n_feature, iv_lb, iv_ub, split_rate, \
              fig_name, save_fig, _resample, model_name):
    '''
    processSample -> pickVars -> classifer -> joblib.dump
    '''
    print('\n****************** Training a model for ' + target + ' ******************') 
    processSample(input_tbl, except_for_these, targets, target, n_bin, idx_tbl, bin_tbl, woeiv_tbl, end_tbl, month_to_target, encodeString)
    
    ## Feature selection - stepwise    
    final_vars = pickVars(woeiv_tbl, bin_tbl, except_for_these + [target], iv_lb, iv_ub, n_feature)
    df = (spark.table(end_tbl).toPandas())
    X = df[[x + '_woe' for x in final_vars]]
    y = df[target]
    
    ## Multicollinearity check
    vif = checkVif(X)

    while vif['vif'][vif['vif'] > 10].any():
        remove = vif.sort_values('vif',ascending=0)['features'][:1]
        print(remove)
        X.drop(remove, axis=1, inplace=True)
        vif = checkVif(X)

    print('*** VIF analysis is completed and features are selected with VIF <= 10.')
    print(list(vif['features']))

    ## Feature selection - stepwise
    features = fs.stepwise_selection(X, y, threshold_in, threshold_out)
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
            ## '_indexed' = 8
            sql = 'select ' + f + '_bucket, min(' + f[:-8] + ') as lb, max(' + f[:-8] + ') as ub from ' + bin_tbl + ' group by 1 order by 1'
        else:
            sql = 'select ' + f + '_bucket, min(' + f + ') as lb, max(' + f + ') as ub from ' + bin_tbl + ' group by 1 order by 1'
        spark.sql(sql).show(truncate=False)
        
#    with open(model_loc + target + '_feature_list', 'rb') as r:
#         features = pickle.load(r)

    clf = classifier(end_tbl, target, except_for_these, features, split_rate, fig_name, save_fig, _resample)
    
    ## Save the model
    joblib.dump(clf, model_loc + model_name)
    print('*** Saved a trained model... ***')
        
def processSample(input_tbl, exclusion, targets, target, n_bin, idx_tbl, bin_tbl, woeiv_tbl, result_tbl, month_to_target, encodeString = True):
    '''
    bucketize, calculate WOE/IV, apply WOE
    exclusion : nominal columns such as party_id and cmdb_partition
    '''
    print('*** Start of Processing ' + target + ' - bucketize, calculate WOE/IV, apply WOE *** ')
    print('*** Checks imbalance in data...')
    targets.remove(target)
    df0 = spark.table(input_tbl).drop(*targets)
    
    print('*** Distribution by target value ***' + target)
    ### 403b_k12_policies > 0
    ds = df0.filter((F.col('403b_k12_policies') > 0) & (F.col('cmdb_partition') == month_to_target))
    ds.groupBy(target).count().show()
    
    ## Indexing
    if encodeString:
        df = indexCategorical(ds, exclusion, target)
        df.write.mode('overwrite').saveAsTable(idx_tbl)
  
    ## WOE/IV
    df = spark.table(idx_tbl)
    decoded_cols = bucketize(df, target, exclusion, n_bin, bin_tbl)
    calculateIV(bin_tbl, target, decoded_cols, woeiv_tbl)
    useWOE(bin_tbl, exclusion + [target], woeiv_tbl, result_tbl)
    
    print('*** End of Processing ***')

def checkVif(features):
    # Multicollinearity check
    print('*** Checking Variance Inflation Factor ...')
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
#    df_numeric = features.select_dtypes(exclude=['object'])
    
    vif["features"] = features.columns
    vif["vif"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]    
    return vif

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
    for c in df.drop(target, *except_for_these).dtypes:
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
    indexer = [StringIndexer(inputCol=s[0], outputCol=s[0]+'_indexed',handleInvalid='keep') for s in df.drop(target, *except_for_these).dtypes if s[1] == 'string']
#    encoder = [OneHotEncoderEstimator(inputCols=[s[0]+'_indexed'],outputCols=[s[0]+"_encoded"],handleInvalid='keep') for s in df.drop(*except_for_these, target).dtypes if s[1] == 'string']
    pipeline = Pipeline(stages=indexer)
    df2 = pipeline.fit(df).transform(df)
#    df2.show(truncate=False)
    print('*** End of indexing string variables...')
    return df2 
 
def classifier(tbl, target, except_for_these, features, split, filename, _save_fig, _resample):
    '''
    target : dependent
    except_for_these : nominal - party_id, cmdb_partition
    features : selected features
    '''

    df_model = spark.table(tbl).select(target, *except_for_these, *features).toPandas()
    
    X_train_meta, y_train, X_test_meta, y_test = splitDataset(df_model, target, split)
    
    ## Resampling
    X = pd.concat([X_train_meta, y_train], axis=1)
    minority = X[X[target] == 1]
    majority = X[X[target] == 0]
    
    if _resample == 'under':
        print('*** Undersampling majority ***')
        ## Undersampling majority  
        down_majority = resample(majority, replace = False, n_samples = len(minority), random_state= _seed )
        resampled = pd.concat([minority, down_majority])
        print('** Resamped size **')
        print(resampled[target].value_counts())
        
        X_train_meta = resampled[resampled.columns.difference([target])]
        X_train = resampled[resampled.columns.difference([target] + except_for_these)]
        y_train = resampled[target]
    
    elif _resample == 'over':
        print('*** Oversampling minority ***')
        ## Oversampling minority  
        up_minority = resample(minority, replace = True, n_samples = len(majority), random_state= _seed )
        resampled = pd.concat([majority, up_minority])
        print('** Resamped size **')
        print(resampled[target].value_counts())
        
        X_train_meta = resampled[resampled.columns.difference([target])]
        X_train = resampled[resampled.columns.difference([target] + except_for_these)]
        y_train = resampled[target]
        
    elif _resample == '':
        X_train = X_train_meta[X_train_meta.columns.difference(except_for_these)]
    
    X_test = X_test_meta[X_test_meta.columns.difference(except_for_these)]
    
    ## Classifiers
    print('\n*** Fits a model ***')
    clf = LogisticRegression(class_weight='balanced', max_iter=20000,solver='lbfgs',random_state=_seed)
#    clf = RandomForestClassifier(criterion='entropy', class_weight='balanced', max_features='auto', random_state=_seed, n_jobs=-1)
        
    ## Gridsearch to tune hyperparameters
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
    scores_train = scoring(X_train, best_clf, np.array(y_train))
    scores_test = scoring(X_test, best_clf, np.array(y_test))
    
    print('*** Deciles - Train ***')
    decile_train = vis.deciling(scores_train,['DECILE'],'TARGET','NONTARGET')
    print(decile_train)
    
    print('*** Deciles - Test ***')
    decile_test = vis.deciling(scores_test,['DECILE'],'TARGET','NONTARGET')
    print(decile_test)
    
    lift_train = pd.concat([X_train,scores_train],axis=1, join_axes=[X_train.index])
    lift_test = pd.concat([X_test,scores_test],axis=1, join_axes=[X_test.index])

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
    vis.gains(lift_merged, ['DECILE'], 'TARGET', 'PROB', graph_loc + 'lift_train_test_' + filename + '.jpg')
    
    return best_clf

def splitDataset(df_model,target,split):

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
    plt.savefig(graph_loc + 'roc_'+filename+'.jpg')
    plt.close()
    
    # Final Model performance details
    print('\n*** *********************************** ***')
    print('*** Model Performance against train set ***')
    print('*** *********************************** ***')
    pred_train = clf.predict(X_train)
    accuracy_train = accuracy_score(y_train,pred_train)
    print('Train set accuracy : {}'.format(100*accuracy_train))
        
    f1_train = f1_score(y_train,pred_train)
    print('Train set F1 score : {}'.format(f1_train))
    
    fpr, tpr, _ = roc_curve(np.array(y_train), clf.predict_proba(X_train)[:,1])
    auc_train = auc(fpr,tpr)
    print('Train set AUC : {}'.format(auc_train))
    
    print('\n*** Confusion matrix ***')
    print(confusion_matrix(np.array(y_train), pred_train))
    
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
    accuracy_test = accuracy_score(y_test,pred_test)
    print('Test set accuracy : {}'.format(100*accuracy_test))
    
    f1_test = f1_score(y_test,pred_test)
    print('Test set F1 score : {}'.format(f1_test))
    
    fpr, tpr, _ = roc_curve(np.array(y_test), clf.predict_proba(X_test)[:,1])
    auc_test = auc(fpr,tpr)
    print('Test set AUC : {}'.format(auc_test))
    
    print('\n*** Confusion matrix ***')
    print(confusion_matrix(np.array(y_test), pred_test))

if __name__ == '__main__':
    
    _QUIET_SPARK = True
    spark = SparkSession.builder.appName("gr_model").getOrCreate()
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

    _seed = 7878
    model_loc = './model/'
    graph_loc = './fig/'
    config_loc = './config/'
    
    print('*** Started at  *** : ' + format(datetime.today(), '%Y-%m-%d %H:%M:%S'))
    
    if sys.argv[1] == 'train':
        with open(config_loc + sys.argv[2], 'r') as r:
            data = json.load(r)
        
        for p in ['params_30', 'params_60', 'params_90','params_120']:
            modelDriver(data['process_data'], data['input_tbl'], data['targets'], data[p]['target'], data['except_for_these'], data[p]['month_to_target'], data[p]['encodeString'], \
                  data[p]['idx_tbl'], int(data[p]['n_bin']), data[p]['bin_tbl'], data[p]['woeiv_tbl'], data[p]['end_tbl'], \
                  float(data['threshold_in']), float(data['threshold_out']), int(data[p]['n_feature']), float(data[p]['iv_lb']), float(data[p]['iv_ub']), \
                  float(data[p]['split_rate']), data[p]['fig_name'], data['resample'], data[p]['model_name'], int(data['trainOnly']))
                
    if sys.argv[1] == 'predict':
        with open(config_loc + sys.argv[2], 'r') as r:
            data = json.load(r)
            
        for p in ['params_30', 'params_60', 'params_90', 'params_120']:
            predictDriver(data['process_data'], data['input_tbl'], data['targets'], data['except_for_these'], data[p]['model_name'], data[p]['target'], \
                          data[p]['month_to_target'], data[p]['encodeString'], data[p]['idx_tbl'], data[p]['bin_tbl'], data[p]['woeiv_tbl'], \
                          data[p]['end_tbl'], data[p]['fig_name'], int(data[p]['n_bin']))
            
    print('*** Finished at *** : ' + format(datetime.today(), '%Y-%m-%d %H:%M:%S \n\n'))
    spark.stop()
