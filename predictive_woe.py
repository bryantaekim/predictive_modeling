#!/usr/bin/env python
'''
@author: Bryan Kim
@title: Predictive model Weight of Evidence
'''
from __future__ import print_function
from pyspark.sql import SparkSession, SQLContext, functions as F, Window as W
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from datetime import datetime
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score

print(__doc__)

pd.set_option('max_colwidth', 70)
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

_seed = 7878
_QUIET_SPARK = True
spark = SparkSession.builder.appName("predmodel_woe").getOrCreate()
if _QUIET_SPARK:
    spark.sparkContext.setLogLevel("WARN")
sqlContext = SQLContext(spark.sparkContext)
sqlContext.setConf("spark.sql.autoBroadcastJoinThreshold", "-1")
sqlContext.setConf("spark.sql.files.openCostInBytes", "8388608")

def main():
    print('*** Started at  *** : ' + format(datetime.today(), '%Y-%m-%d %H:%M:%S'))
    n_sample = 100000
    data_pipeline = """_some query_
                    order by rand(""" + str(_seed) + """)
                    limit """ + str(n_sample) + """"""
    sample_set = 'upred_sample'
    target = 'some_flag'
    except_for_these = [target] + ['some columns to be excluded']
    _undersample = False
    undersample_rate = 1.3
    _save_fig = True
    nonnull_cutoff =  int(n_sample * .1)
    unique_num_value = 3
    balThreshold = .8
    n_feat = 100
    train_test_split = .3
    _createSample = True
    
    print('*** Modeling setttings  ***')
    print('*** Sample size : ' + str(n_sample))
    print('*** Tran/test split rate : ' + str(train_test_split))
    print('*** Cutoff for imbalance check - TP/TN ratio = ' + str(balThreshold) + ' : 1')
    print('*** Undersampling major class : ' + str(_undersample))
    if _undersample:
        print('*** Undersampling ratio - major/minor ratio = ' + str(undersample_rate) + ' : 1')
    print('*** Cuttoff for non-null value : ' + str(nonnull_cutoff))
    print('*** Cuttoff for distinct numeric value (greater than) : ' + str(unique_num_value))
    print('*** The number of features shown : ' + str(n_feat))
    print('\n\n')
    
    if _createSample:
        df, num, cat = create_base(_undersample,undersample_rate, _save_fig,data_pipeline,target,except_for_these,nonnull_cutoff,unique_num_value,
                                     balThreshold,n_feat, use_wgt=False)
        create_sample(df, target, num, cat, n_feat, sample_set)
    
    df_spark = spark.table(sample_set).persist()
    df_model = df_spark.toPandas()
        
    X_train, y_train, X_test, y_test = splitDataset(df_model, target, split=train_test_split)
    print('\n*** Final data set ***')
    print(df_model.head())
    
    #Classifier
    print('\n*** Fit a model ***')
#    from sklearn.linear_model import LogisticRegression
#    clf = LogisticRegression(solver='lbfgs', max_iter=20000, class_weight='balanced', random_state=_seed, n_jobs=-1)
    
    #from sklearn.ensemble import RandomForestClassifier
    #clf = RandomForestClassifier(n_estimators=150, max_depth=15, criterion='entropy', class_weight='balanced', random_state=_seed, n_jobs=-1)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=_seed, n_jobs=-1, penality='elasticnet', l1_ratio=.5)
        
    ## Gridsearch to tune hyperparameters
        from sklearn.model_selection import GridSearchCV
        param_grid = {
                        'max_iter':list(range(1000,20000,100))
                    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, error_score=0)
    grid_search.fit(X_train, y_train)

    print('*** Best parameters....')
    print(grid_search.best_params_)
    best_clf = grid_search.best_estimator_
        
    # Cross Validation
    fitModelCV(5, best_clf, X_train, y_train, X_test, y_test)
    # Non-CV
#    fitModel(best_clf, X_train, X_test, y_train, y_test)
    
    # Plot graphics - Life and Gain
    scores_train = scoring(X_train,best_clf,y_train)
    scores_test = scoring(X_test,best_clf,y_test)
    lift_train = pd.concat([X_train,scores_train],axis=1)
    lift_test = pd.concat([X_test,scores_test],axis=1)
    
    print('*** Life curve for train set is saved, lift_train.jpg...')
    gains(lift_train,['DECILE'],'TARGET','PROB', './fig/lift_train.jpg')
    
    print('*** Life curve for test set is saved, lift_test.jpg...')
    gains(lift_test,['DECILE'],'TARGET','PROB', './fig/lift_test.jpg')
    
    print('*** Finished at *** : ' + format(datetime.today(), '%Y-%m-%d %H:%M:%S'))

def create_base(_undersample,undersample_rate, _save_fig,data_pipeline, target,except_for_these,nonnull_cutoff,unique_num_value,balThreshold,n_feat, use_wgt):
    base = spark.sql(data_pipeline)
    base.persist()
    
    #Drop all Nulls or less than a threshold
    cols_drop = []
    for col_name in base.drop(*except_for_these).columns:
        if base.select(col_name).filter((F.col(col_name).isNotNull()) | (F.col(col_name).cast('double') == 0.0)).count() <= nonnull_cutoff:
            cols_drop.append(col_name)
    base2 = base.drop(*cols_drop)
    print('*** Dropped null-only or columns having non-null values less than or equal to ' + str(nonnull_cutoff) + '...')
    print(str(cols_drop))
    base.unpersist()
    base2.persist()
    
    print('\n*** Excluded features including a target...')
    print(str(except_for_these))
    
    num = []
    cat = []
    for c in base2.drop(*except_for_these).dtypes:
        if c[1] == 'string':
            cat += [c[0]]
            base2 = base2.withColumn(c[0], F.when((F.col(c[0]) == '') | (F.col(c[0]) == ' ') | (F.col(c[0]).isNull()), F.lit('None')).otherwise(F.col(c[0])))
        if c[1] not in ('string', 'timestamp') and base2.select(F.approx_count_distinct(c[0])).collect()[0][0] > unique_num_value:
            num += [c[0]]
            base2 = ( base2.withColumn(c[0], F.when((F.col(c[0]) == '') | (F.col(c[0]) == ' '), F.lit(None)).otherwise(F.col(c[0])))
                           .withColumn(c[0], F.col(c[0]).cast('double')))
    
    print('\n*** Distribution by target value ***')
    base2.groupBy(target).count().show()
    
    # Create batches for faster processing
    base = (df1.join(df_seg, ['party_id', 'cmdb_partition'], 'left').withColumn('rnum',F.row_number().over(W.orderBy('party_id'))))
    # Replace empty strings to None string or null
    n_batch = 20
    batch_size = int(n_sample/n_batch)
    total_nulls = {}
    dfs = {}
    print('\n*** Replace empty strings to None string or null...***')
    for i in range(n_batch):
        print('*** batch :' + str(i+1) + ' out of ' + str(n_batch) + ' with a size of ' + str(batch_size))
        lb = i* batch_size + 1
        ub = i * batch_size + batch_size
        df_b = base.filter((F.col('rnum') >= lb) & (F.col('rnum') <= ub))
        for c in df_b.drop(*except_for_these).columns:
            if c in cat:
                df_b = df_b.withColumn(c, F.when((F.col(c) == '') | (F.col(c) == ' ') | (F.col(c).isNull()), F.lit('None')).otherwise(F.col(c)))
            if c in num:
                df_b = (df_b.withColumn(c, F.when((F.col(c) == '') | (F.col(c) == ' '), F.lit(None)).otherwise(F.col(c)))
                           .withColumn(c, F.col(c).cast('double')))
            nulls = df_b.select(c).filter((F.col(c).isNull()) | (F.col(c) == 'None')).count()
            total_nulls.setdefault(c, []).append(nulls)
        dfs[str(i)]=df_b
    
    drop_list = []
    for col, nulls in total_nulls.items():
        print('Column : ' + col + ' - null count : ' + str(nulls))
        if sum(nulls) > 0:
            drop_list.append(col)
    
    print('\n*** Drop list ***')
    print(str(drop_list))
    
    print('\n*** Merge batches into one data frame...')
    from functools import reduce
    def unionAll(dfs):
        return reduce(lambda df1, df2: df1.unionByName(df2), dfs)
    base2 = (unionAll(list(dfs.values())).drop(*drop_list))
    print('*** Merged as base2 ...')
    
    #Check imbalance - weighting for the minority class
    total_size = float(base2.count())
    tp_size = float(base2.select(target).filter(F.col(target) == 1).count())
    tn_size = float(total_size - tp_size)
    if tp_size/tn_size < balThreshold:
        print("Imbalance issue exists....target vs non-target ratio : " + str(tp_size/tn_size))
        if use_wgt:
            class_weight = tn_size / total_size
            base2 = base2.withColumn('classWeight', F.when(F.col(target) == 0, class_weight).otherwise(1-class_weight))
        else:
            pass
    else:
        pass
    
    print('\n*** Describe categorical variables in the data set ***')
    print(str(cat))
    for i in cat:    
        base2.groupBy(i).count().orderBy('count', ascending=False).show(2000,truncate=False)
    
    print('*** Describe numeric variables in the data set ***')
    print(str(num))
    for i in num:
        base2.select(i).summary().show(truncate=False)
        if _save_fig:
            tmp = base2.select(i).toPandas()
            tmp.hist(figsize=(12,10))
            plt.savefig('./fig/'+i+'.png')
            plt.close()    
    
    if _undersample:
        print('*** Undersampling major class due to imbalanced data ***')
        base_tp = base2.filter(F.col(target) == 1)
        _count = base_tp.count()
        print('*** Target size :' + str(_count*undersample_rate))
        base_tn = (base2.filter(F.col(target) == 0)
                .withColumn('rand_num', F.lit(F.rand(_seed)))
                .withColumn('row_num', F.row_number().over(W.orderBy('rand_num')))
                .filter(F.col('row_num') <= _count*undersample_rate)
                .drop('rand_num', 'row_num'))
        df = base_tp.unionByName(base_tn)
    else:
        df = base2
        
    base2.unpersist()    
    return df, num, cat

def create_sample(df, target, num, cat, n_feat, sample_set):
    '''
    Create a final sample set : calculate WOE/IV, select features based on IV, replace variable values with WOEs
    # Variables with IV - 
    # useless : <.02
    # weak : between .02 and .1
    # medium : between .1 and .3
    # strong : between .3 and .5
    # Too good to be true : > .5
    '''
    print('*** Calculating WOE and IV ....***')
    woe_iv = pd.DataFrame([], columns=['feat_name','bucket','all','event','tot_event','nonevent','tot_nonevent','pct_event','pct_nonevent','woe','iv'])
    # Continuous
    col_decile = {}
    for c in num:
        min_ = df.groupby().min(c).collect()[0][0]
        max_ = df.groupby().max(c).collect()[0][0]
        col_decile.setdefault(c, []).append(np.linspace(min_, max_, 11).tolist())
        
    for c in num:
        buckets = Bucketizer(splits=col_decile[c][0], inputCol = c, outputCol = 'bucket', handleInvalid='keep')
        bucketed = buckets.getSplits()   
        df_binned = buckets.transform(df)
        
#        buckets = QuantileDiscretizer(numBuckets = 10, inputCol = c, outputCol = "bucket",handleInvalid='keep') 
#        bucketed = buckets.fit(df).getSplits()
#        df_binned = buckets.fit(df).transform(df)
        
        print('Boundaries for ' + c +  ' : ' + str(bucketed))
        
        df_binned.persist()        
        df_binned = (df_binned.withColumn('feat_name', F.lit(c)).withColumn('bucket', F.col('bucket').cast('string'))
                              .select('*', F.count('*').over(W.partitionBy('feat_name')).alias('feat_total'), F.count('*').over(W.partitionBy('feat_name', 'bucket')).alias('all'), \
                                     F.sum(target).over(W.partitionBy('feat_name', 'bucket')).alias('event'))
                              .withColumn('nonevent', F.col('all')-F.col('event'))
                              .withColumn('tot_event', F.sum(target).over(W.partitionBy('feat_name')))
                              .withColumn('tot_nonevent', F.col('feat_total')-F.col('tot_event'))
                              .withColumn('pct_event', F.round(F.when(F.col('event')==0, (.5/F.col('tot_event'))).otherwise(F.col('event')/F.col('tot_event')), 3))
                              .withColumn('pct_nonevent', F.round(F.when(F.col('nonevent')==0, (.5/F.col('tot_nonevent'))).otherwise(F.col('nonevent')/F.col('tot_nonevent')), 3))
                              .withColumn('woe', F.log(F.col('pct_nonevent')/F.col('pct_event')))
                              .withColumn('iv', F.col('woe')*(F.col('pct_nonevent')-F.col('pct_event')))
                              .select('feat_name','bucket','all','event','tot_event','nonevent','tot_nonevent','pct_event','pct_nonevent','woe','iv')
                              .distinct()
                              .orderBy('feat_name', 'bucket'))
        df_tmp = df_binned.toPandas()
        woe_iv = woe_iv.append(df_tmp, ignore_index=True)
    
    # Categorical
    for c in cat:
        df_cat = (df.withColumn('feat_name', F.lit(c)).withColumnRenamed(c, 'bucket')
                  .select('*', F.count('*').over(W.partitionBy('feat_name')).alias('feat_total'), \
                          F.count('*').over(W.partitionBy('feat_name', 'bucket')).alias('event+non_event'), \
                          F.sum(target).over(W.partitionBy('feat_name', 'bucket')).alias('event'))
                  .withColumn('nonevent', F.col('event+non_event')-F.col('event'))
                  .withColumn('tot_event', F.sum(target).over(W.partitionBy('feat_name')))
                  .withColumn('tot_nonevent', F.col('feat_total')-F.col('tot_event'))
                  .withColumn('pct_event', F.round(F.when(F.col('event')==0, (.5/F.col('tot_event'))).otherwise(F.col('event')/F.col('tot_event')), 3))
                  .withColumn('pct_nonevent', F.round(F.when(F.col('nonevent')==0, (.5/F.col('tot_nonevent'))).otherwise(F.col('nonevent')/F.col('tot_nonevent')),3))
                  .withColumn('woe', F.log(F.col('pct_nonevent')/F.col('pct_event')))
                  .withColumn('iv', F.col('woe')*(F.col('pct_nonevent')-F.col('pct_event')))
                  .select('feat_name','bucket','event+non_event','event','tot_event','nonevent','tot_nonevent','pct_event','pct_nonevent')
                  .distinct()
                  .orderBy('feat_name', 'bucket'))
        df_tmp = df_cat.toPandas()
        woe_iv = woe_iv.append(df_tmp, ignore_index=True)
    
    woe_iv = spark.createDataFrame(woe_iv)
    print('*** WOE/IV Table ***')
    woe_iv.show(truncate=False)
    
    print('\n*** Feature importance by Information Value, Top ' + str(n_feat) + ' features ***')
    ivCalc = woe_iv.groupby('feat_name').agg(F.sum('iv').alias('iv')).orderBy('iv', ascending=False)
    ivCalc.show(n_feat, truncate=False)
    
    print('\n*** Selected features with IV >= .02 ***')
    select_feat = ivCalc.filter('iv >= .02').select('feat_name').rdd.flatMap(lambda x:x).collect()
    
    print('\n*** Replace values with WOEs ***')
    # Bucketize continuous variables except for irrelevant ones
    buckets = [Bucketizer(splits=col_decile[c][0], inputCol = c, outputCol = c+'_bucket', handleInvalid='keep') for c in num]
#    buckets = [QuantileDiscretizer(numBuckets = 10, inputCol = c, outputCol = c+"_bucket",handleInvalid='keep') for c in num]
    pipeline = Pipeline(stages=buckets)
    df_bucketed = pipeline.fit(df).transform(df)
        
    cols = [c for c in df_bucketed.columns if c.endswith('_bucket')] + cat
    woe_list = [row.asDict() for row in woe_iv.select('feat_name','bucket','woe').collect()]
    def woe_mapper(feat, bucket):
        for d in woe_list:
            if d['feat_name'] == feat and d['bucket'] == bucket:
                return d['woe']
    
    woe_mapper_udf = F.udf(woe_mapper, DoubleType())
    for c in cols:
        if c.endswith('_bucket'):
            df_bucketed = df_bucketed.withColumn(c.replace('_bucket','_woe'), F.lit(woe_mapper_udf(F.lit(c[:-len('_bucket')]), F.col(c).cast('string'))))
        else:
            df_bucketed = df_bucketed.withColumn(c+'_woe', F.lit(woe_mapper_udf(F.lit(c), F.col(c))))
    
    # Create a data set with select features
    df_model = df_bucketed.select(target, *[x+'_woe' for x in select_feat])
    df_model.printSchema()
    df_model.show(truncate=False)
    df_model.write.mode('overwrite').saveAsTable(sample_set)
    print('\n*** Finished creating a final data set for fitting...')

def splitDataset(df_model, target, split):
    train, test = train_test_split(df_model, test_size = split)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    X_train = train[train.columns.difference([target])]
    y_train = train[target]
    X_test = test[test.columns.difference([target])]
    y_test = test[target]
    
    print('Train set : %d'%len(y_train))
    print('Test set : %d'%len(y_test))
    return X_train, y_train, X_test, y_test

def fitModelCV(n_splits, clf, X_train, y_train, X_test, y_test):
    # CV Model performance
    scores = cross_val_score(clf, X_train,y_train,cv=n_splits, n_jobs=-1)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clf, X_train,y_train,cv=n_splits, n_jobs=-1, scoring='f1')
    print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    # plot ROC/AUC per CV fold
    cv = StratifiedKFold(n_splits=n_splits)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(1,figsize=(20, 5))
    fig, ax = plt.subplots()
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
    plt.savefig('./fig/roc_cv.jpg')
    plt.close()
    
    # Final Model performance details
    print('\n*** *********************************** ***')
    print('*** Model Performance against train set ***')
    print('*** *********************************** ***')
    pred_train = clf.predict(X_train)
    accuracy_train = accuracy_score(pred_train,y_train)
    print('Train set accuracy : {}'.format(100*accuracy_train))
    
    fpr, tpr, _ = roc_curve(np.array(y_train), clf.predict_proba(X_train)[:,1])
    auc_train = auc(fpr,tpr)
    print('Train set AUC : {}'.format(auc_train))
    
    print('\n*** Confusion matrix - train set')
    print(pd.crosstab(y_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED']))
    
    print('\n*** *********************************** ***')
    print('*** Model Performance against test set ***')
    print('*** *********************************** ***')
    pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(pred_test,y_test)
    print('Test set accuracy : {}'.format(100*accuracy_test))
    
    fpr, tpr, _ = roc_curve(np.array(y_test), clf.predict_proba(X_test)[:,1])
    auc_test = auc(fpr,tpr)
    print('Test set AUC : {}'.format(auc_test))
    
    print('\n*** Confusion matrix - test set')
    print(pd.crosstab(y_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED']))
    
def fitModel(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train,y_train)
    # Model performance
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    
    accuracy_train = accuracy_score(pred_train,y_train)
    accuracy_test = accuracy_score(pred_test,y_test)
    
    print('Train Accuracy : {} \nTest Accuracy : {}'.format(accuracy_train*100, 100*accuracy_test))
    
    fpr, tpr, _ = roc_curve(np.array(y_train), clf.predict_proba(X_train)[:,1])
    auc_train = auc(fpr,tpr)
    print('Train AUC : {}'.format(auc_train))
    
    fpr, tpr, _ = roc_curve(np.array(y_test), clf.predict_proba(X_test)[:,1])
    auc_test = auc(fpr,tpr)
    print('Test AUC : {}'.format(auc_test))
    
    print('\n*** Confusion matrix - Train')
    print(pd.crosstab(y_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED']))
    print('\n*** Confusion matrix - Test')
    print(pd.crosstab(y_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED']))
    
    # Plot ROC curves  
    fpr_train, tpr_train, _ = roc_curve(np.array(y_train), clf.predict_proba(X_train)[:,1])
    auc_train = auc(fpr_train,tpr_train)
    fpr_test, tpr_test, _ = roc_curve(np.array(y_test), clf.predict_proba(X_test)[:,1])
    auc_test = auc(fpr_test,tpr_test)
    
    plt.figure()
    plt.plot(fpr_train, tpr_train, color='orange', lw=2, label='ROC curve for Train (area = %0.3f)' % auc_train)
    plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='ROC curve for Test(area = %0.3f)' % auc_test)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',  label='Random Guess', alpha=.8)
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - Logistic regression with WOE')
    plt.legend(loc="lower right")
    
    plt.savefig('./fig/roc.jpg')
    plt.show()
    plt.close()

# =============================================================================
# Visualization helper functions
# =============================================================================
def plots(agg1,target,type,filename):
    plt.figure(1,figsize=(20, 5))

    plt.subplot(131)
    plt.plot(agg1['DECILE'],agg1['ACTUAL'],label='Actual')
    plt.plot(agg1['DECILE'],agg1['PRED'],label='Pred')
    plt.xticks(range(10,110,10))
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.title('Actual vs Predicted', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + " " + str(type) + " %",fontsize=15)

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
    plt.title('Gains Chart', fontsize=20)
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
    
def gains(data,decile_by,target,score,filename):
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

def scoring(features,clf,target):
    score = pd.DataFrame(clf.predict_proba(features)[:,1], columns = ['PROB'])
    score['TARGET'] = target
    score['NONTARGET'] = 1 - target
    score['DECILE'] = pd.qcut(score['PROB'].rank(method = 'first'),10,labels=range(10,0,-1)).astype(float)
    return score

if __name__ == '__main__':
    main()
