#!/usr/bin/env python
'''
@author: Bryan Kim
@title: Predictive model - GR restarter
'''
from __future__ import print_function
from pyspark.sql import SparkSession, SQLContext, functions as F, Window as W
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler,MinMaxScaler, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier,GBTClassifier, LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pandas as pd
print(__doc__)

_seed = 7878
_QUIET_SPARK = True
spark = SparkSession.builder.appName("predmodel_rf").getOrCreate()
if _QUIET_SPARK:
    spark.sparkContext.setLogLevel("WARN")
sqlContext = SQLContext(spark.sparkContext)
sqlContext.setConf("spark.sql.autoBroadcastJoinThreshold", "-1")
sqlContext.setConf("spark.sql.files.openCostInBytes", "8388608")

def main():
    data_pipeline = """__some query__
                    order by rand(7878)
                    limit 10000
                    """
    target = 'some_flag'
    feature = 'features'
    except_for_these = [target] + ['some columns']
    _undersample = False
    _print = True
    _eda = False
    non_null = 100
    balThreshold = .8
    n_feat = 100
    split_rate = [0.7, 0.3]
    num, cat = create_sample(_undersample, _print,data_pipeline, _eda, 
                  target,except_for_these,non_null,balThreshold,split_rate, use_wgt = False)
    run_pipeline(target, feature, num, cat, n_feat)
    
def create_sample(_undersample, _print,data_pipeline, _eda, target,except_for_these,non_null,balThreshold,split_rate, use_wgt):
    base = spark.sql(data_pipeline)
    base.persist()
    
    #Drop all Nulls or less than a threshold
    cols_drop = []
    for col_name in base.columns:
        if base.select(col_name).filter((F.col(col_name).isNotNull()) | (F.col(col_name).cast('double') == 0.0)).count() <= non_null:
            cols_drop.append(col_name)
    base2 = base.drop(*cols_drop)
#    print('All null columns are dropped...' + str(cols_drop))
    base.unpersist()
    base2.persist()
    
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
#    print("*** Numeric columns ***")
#    for n in num:
#        print(n)
  
#    print("*** Categorical columns ***")
#    for c in cat:
#        print(c)
    
    if _eda:
        perform_eda(base2, target, num, cat)
    
    if _undersample:
        print('*** Undersampling due to imbalanced data ***')
        base_tp = base2.filter(F.col(target) == 1)
        _count = base_tp.count()
        print('*** Target size :' + str(_count))
        base_tn = (base2.filter(F.col(target) == 0)
                .withColumn('rand_num', F.lit(F.rand(_seed)))
                .withColumn('row_num', F.row_number().over(W.orderBy('rand_num')))
                .filter(F.col('row_num') <= _count)
                .drop('rand_num', 'row_num')
                 )
        df = base_tp.unionByName(base_tn)
    else:
        df = base2
    
    train, test = df.randomSplit(split_rate, seed=_seed)
    
    #Handles Imbalance - Weighting for the minority class
    train_size = float(train.count())
    tp_size = float(train.select(target).filter(F.col(target) == 1).count())
    tn_size = float(train_size - tp_size)
    if tp_size/tn_size < balThreshold:
        print("Imbalance issue exists....target vs non-target ratio : " + str(tp_size/tn_size))
        if use_wgt:
            class_weight = tn_size / train_size
            train = train.withColumn('classWeight', F.when(F.col(target) == 0, class_weight).otherwise(1-class_weight))
            test = test.withColumn('classWeight', F.when(F.col(target) == 0, class_weight).otherwise(1-class_weight))
        else:
            pass
    else:
        pass
    
    if _print:
        print('*** Train set ***')
        train.groupBy(target).count().show()
        train.show(10)
        
        print('*** Test set ***')
        test.groupBy(target).count().show()
        test.show(10)
    
    base2.unpersist()
    
    train.write.mode('overwrite').saveAsTable('predictmodel_train')
    test.write.mode('overwrite').saveAsTable('predictmodel_test')
    print('*** Train and test are created/saved.')
    return num, cat

def run_pipeline(target, feature, num, cat, n_feat):
    train = spark.table('predictmodel_train')        
    train.persist()
    stages = []
    
    indexer = [StringIndexer(inputCol=s, outputCol=s+'_indexed',handleInvalid='keep') for s in cat]
    encoder = [OneHotEncoderEstimator(inputCols=[s+'_indexed'],outputCols=[s+"_encoded"],handleInvalid='keep') for s in cat]
        
    imputer = [Imputer(strategy='mean',inputCols=num, outputCols=num)]
    
    num_assmblr = [VectorAssembler(inputCols=[n], outputCol=n+'_vect') for n in num]
    num_scaler = [MinMaxScaler(inputCol=n+'_vect', outputCol=n+"_scaled") for n in num]
    
#    pipeline_num = Pipeline(stages=indexer + encoder + imputer + num_assmblr + num_scaler)
#    train = pipeline_num.fit(train).transform(train)

#    print("*** show encoded categorical variables ....")
#    train.select(*[s+'_encoded' for s in cat]).show(10, truncate=False)
    
#    unpack_list = F.udf(lambda x: round(float(list(x)[0]),3), DoubleType())
#    for n in num:
#        train = train.withColumn(n+"_scaled", unpack_list(n+"_scaled")).drop(n+"_vect") 
#    print("*** show scaled numeric variables ....")
#    train.select(*[n+'_scaled' for n in num]).summary("count", "min", "25%", "75%", "max").show(10, truncate=False)
    
#    assembler = VectorAssembler(inputCols=[num_scaler.getOutputCol()] + [s+"_encoded" for s in cat], outputCol=feature)
    assembler = VectorAssembler(inputCols=[n+'_scaled' for n in num] + [s+"_encoded" for s in cat], outputCol=feature)
    
    target_indexed = target+"_indx"
    labelIndexer = StringIndexer(inputCol = target, outputCol = target_indexed, handleInvalid = 'keep')   
    
    model = clf_rf(feature, target_indexed)
#    model = clf_gbt(feature, target)
#    model = clf_lr(feature, target)
    validator = _val(target_indexed, model)
    
    stages += [assembler, labelIndexer, validator]
    print('*** stages are created and now are running... ***')
    
    pipeline = Pipeline(stages=indexer + encoder + imputer + num_assmblr + num_scaler + stages)
    pipeline_model = pipeline.fit(train)
    last_stage = pipeline_model.stages[-1]
    transformedData = pipeline_model.transform(train)
    
    transformedData.write.mode('overwrite').saveAsTable('us_marketing_usecase.transformedData')
    print('*** transformed data is saved for modeling... ***')
    
#    print('*** Transformed training set ***')
#    cols = num + cat
#    transformedData.select(target_indexed,feature,*cols).show(10, truncate=False)
# =============================================================================
# RandomForest/GradientBoosting
# =============================================================================
    print('*** Model performance ***')
    evaluate(transformedData,target_indexed)
    
    print('*** Feature Importances ***')
    featImp = last_stage.bestModel.featureImportances
    
    print('*** Show important ' + str(n_feat) +  ' features ***')
    list_extract = []
    for i in transformedData.schema['features'].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + transformedData.schema['features'].metadata["ml_attr"]["attrs"][i]
    
    pd.set_option('display.max_rows', 500)
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x:featImp[x])
    selected_feat = varlist.sort_values('score', ascending=False)
    
    print(selected_feat.iloc[0:n_feat, :])
    
    # Get the best hyperparameters:
    print('MaxDepth: ' + str(last_stage.bestModel._java_obj.getMaxDepth()))
    print('NumTrees: ' + str(last_stage.bestModel.getNumTrees))
    
# =============================================================================
# Logistic Regression
# =============================================================================
#    print('*** Model performance ***')
#    evaluate(transformedData,target)
#    
#    print('*** Model feature attributes ***')
#    trainingSummary = last_stage.bestModel.summary
#    trainingSummary.roc.show()
#    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

# =============================================================================
# Prediction and Evaluation
# =============================================================================
    predicted = predict(pipeline_model,target_indexed)
    evaluate(predicted,target_indexed)
    
    train.unpersist()


def clf_gbt(feature, target):
    gbt = GBTClassifier(featuresCol=feature, labelCol=target, maxIter=10, seed=_seed, cacheNodeIds=True)
    paramGrid = ( ParamGridBuilder()
        .addGrid(gbt.maxDepth, [10,15,20])
        .addGrid(gbt.stepSize, [.05,.1,.5])
        .build())
    return gbt, paramGrid

def clf_rf(feature, target):
    rf = RandomForestClassifier(featuresCol=feature, labelCol=target, impurity='entropy', seed=_seed, cacheNodeIds=True)
    paramGrid = ( ParamGridBuilder()
        .addGrid(rf.maxDepth, [10,15,20])
        .addGrid(rf.numTrees, [100,150,200])
        .build())
    return rf, paramGrid

def clf_lr(feature, target):
    lr = LogisticRegression(featuresCol=feature, labelCol=target, weightCol='classWeight')
    paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [.01, .5])
             .addGrid(lr.elasticNetParam, [.5, 1.0])
             .addGrid(lr.maxIter, [10,20])
             .build())    
    return lr, paramGrid

def _val(target, model):
    clf, paramGrid = model
    evaluator = BinaryClassificationEvaluator(labelCol=target, rawPredictionCol='prediction')
#    validator = TrainValidationSplit(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator)
    validator = CrossValidator(estimator=clf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)    
    return validator

def predict(pipeline_model,target):
    test = spark.table('predictmodel_test')
    test.persist()
    print('*** Predicting test set and show a confustion matrix ***')
    predicted = pipeline_model.transform(test)
    predicted.crosstab("prediction", target).orderBy("prediction_"+target).show()
    test.unpersist()
    return predicted

def evaluate(predicted,target):
    print('*** Evaluationg predicted labels. ***')
    evaluator = MulticlassClassificationEvaluator(labelCol=target, predictionCol='prediction', metricName="accuracy")
    evaluator2 = BinaryClassificationEvaluator(labelCol=target, rawPredictionCol='prediction', metricName="areaUnderPR")
    print("*** Accuracy ***")
    print(evaluator.evaluate(predicted))
    print("*** ROC ***")
    print(evaluator2.evaluate(predicted))

def perform_eda(df_in, target, num, cat):
    df_in.persist()
    print('*** Target distribution ***')
    df_in.groupBy(target).count().show()
    
    print('*** Describe categorical variables in the data set ***')
    for i in cat:    
        df_in.groupBy(i).count().orderBy('count', ascending=False).show(2000,truncate=False)
    
    print('*** Describe numeric variables in the data set ***')
    for i in num:        
        df_in.select(i).summary("count", "min", "25%", "75%", "max").show(truncate=False)
        print('********')
    df_in.unpersist()

if __name__ == '__main__':
    main()
