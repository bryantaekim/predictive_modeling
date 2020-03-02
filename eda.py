#!/usr/bin/env python
'''
@author: Bryan Kim
@title: EDA - Exploratory Data Analysis
'''
from __future__ import print_function
from pyspark.sql import SparkSession, SQLContext, functions as F, Window as W
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler,MinMaxScaler, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
import pandas as pd, seaborn as sb, numpy as np
import matplotlib.pyplot as plt
pd.set_option('max_colwidth', 70)
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
print(__doc__)
_seed = 7878
n_sample = 100000

data_pipeline = """__some query__
                order by rand(""" + str(_seed) + """)
                limit """ + str(n_sample) + """"""
target = 'some_flag'
feature = 'features'
except_for_these = [target] + ['some columns']
nonnull_cutoff =  n_sample * 0
save_fig = True

    
def perform_eda(data_pipeline, target, feature, nonnull_cutoff, except_for_these, save_fig):
    spark = _loadSpark()
    base = spark.sql(data_pipeline)
    base.persist()
    
    cols_drop = []
    for col_name in base.columns:
        if base.select(col_name).filter((F.col(col_name).isNotNull()) | (F.col(col_name).cast('double') == 0.0)).count() <= nonnull_cutoff:
            cols_drop.append(col_name)
    base2 = base.drop(*cols_drop)
    print('*** Dropped null-only or columns having non-null values less than or equal to ' + str(nonnull_cutoff) + '...')
    print(str(cols_drop))
    base.unpersist()
    base2.persist()
    
    print('\n*** Excluded features including a target...')
    print(str(except_for_these))
    
    print('\n*** Excluded timestamp type and single value numeric features...')
    num = []
    cat = []
    for c in base2.drop(*except_for_these).dtypes:
        if c[1] == 'string':
            cat += [c[0]]
            base2 = base2.withColumn(c[0], F.when((F.col(c[0]) == '') | (F.col(c[0]) == ' ') | (F.col(c[0]).isNull()), F.lit('None')).otherwise(F.col(c[0])))
        if c[1] not in ('string', 'timestamp') and base2.select(F.approx_count_distinct(c[0])).collect()[0][0] > 1:
            num += [c[0]]
            base2 = ( base2.withColumn(c[0], F.when((F.col(c[0]) == '') | (F.col(c[0]) == ' '), F.lit(None)).otherwise(F.col(c[0])))
                           .withColumn(c[0], F.col(c[0]).cast('double')))
    
    print('*** Distribution by target value ***')
    base2.groupBy(target).count().show()
    
    print('*** Describe categorical variables in the data set ***')
    for i in cat:    
        base2.groupBy(i).count().orderBy('count', ascending=False).show(2000,truncate=False)
    
    print('*** Describe numeric variables in the data set ***')
    for i in num:
        base2.select(i).summary().show(truncate=False)
        if save_fig:
            tmp = base2.select(i).toPandas()
            tmp.hist(figsize=(12,10))
            plt.savefig('./fig/'+i+'.png')
            plt.close()
    
#    print('\n*** Correlation Analysis - Highly correlated ***')
#    imputer = Imputer(strategy='mean',inputCols=num, outputCols=num)
#    assmblr = VectorAssembler(inputCols=num, outputCol=feature)
#    pipeline = Pipeline(stages=[imputer,assmblr])
#    transformed_df = pipeline.fit(base2).transform(base2).select(feature)
#    
#    corr_vect = Correlation.corr(transformed_df, feature).collect()[0][0]
#    corr_list = corr_vect.toArray().tolist()
#    pd_corr = pd.DataFrame(corr_list, columns=num, index=num)
#    
#    high_corr = pd_corr.abs().unstack().sort_values(ascending=False)
#    print(high_corr[high_corr > .8])
    
#    print('\n*** Multicollinearity check ***')
#    df_features = base2.select(*num).dropna().toPandas()
#    vif = calculate_vif(df_features)
#    
#    while vif['vif'][vif['vif'] > 10].any():
#        remove = vif.sort_values('vif',ascending=0)['features'][:1]
#        df_features.drop(remove,axis=1,inplace=True)
#        vif = calculate_vif(df_features)
#    
#    print(list(vif['features']))

def calculate_vif(df):
    # Multicollinearity check
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()

    vif["features"] = df.columns
    try:
        vif["vif"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    except:
        pass
    return vif
    
def _loadSpark():
    _QUIET_SPARK = True
    spark = SparkSession.builder.appName("ExploratoryDataAnalysis").getOrCreate()
    if _QUIET_SPARK:
        spark.sparkContext.setLogLevel("WARN")
    sqlContext = SQLContext(spark.sparkContext)
    sqlContext.setConf("spark.sql.autoBroadcastJoinThreshold", "-1")
    sqlContext.setConf("spark.sql.files.openCostInBytes", "8388608")
    return spark

if __name__ == '__main__':
    perform_eda(data_pipeline, target, feature, nonnull_cutoff, except_for_these, save_fig)
