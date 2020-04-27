from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import DenseVector, Vectors
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *

def vifTest(df, except_for_these, vif_threshold):
    cols = df.drop(*except_for_these)
    colnum_max = 1000
    vif_max = vif_threshold + 1
    def vif_cal(df, vif_max, colnum_max, vif_threshold):
        vif_max = vif_threshold
        for i in range(len(cols)):
            train_t = df.rdd.map(lambda x: [Vectors.dense(x[2:i]+x[i+1:]), x[i]]).toDF(['features', 'label'])
            lr = LinearRegression(featuresCol = 'features', labelCol = 'label')
            predictions = lr.fit(train_t).transform(train_t)
            evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label')
            r_sq = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
            vif = 1/(1-r_sq)
            if vif_max < vif:
                vif_max = vif
                colnum_max = i
        return vif_max, colnum_max
    
    while vif_max > vif_threshold:
        vif_max, colnum_max = vif_cal(df, cols, vif_max, colnum_max, vif_threshold)
        if vif_max > vif_threshold:
            df = df.drop(df[colnum_max])
            print('***Columns left***')
            print(str(df.columns))
        else:
            return df
