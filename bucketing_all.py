def pickVars(tmp1, tmp2, df_woe,except_for_these, lb, ub):
    '''
    Create a data set, df_woe, with select features
    '''
    ivCalc = spark.table(tmp1)
    df_bucketed = spark.table(tmp2)
    
    ivCalc.persist()
    df_bucketed.persist()
    
    print('\n*** Selected features with IV between ' + str(lb)+ ' and ' + str(ub) + ' from the above table ***')
    select_feat = ivCalc.filter((F.col('iv') >= lb) & (F.col('iv') <= ub)).select('feat_name').rdd.flatMap(lambda x:x).collect()
    
    df_model = df_bucketed.select(*except_for_these, *[x+'_woe' for x in select_feat])
    df_model.persist()
    df_model.write.mode('overwrite').saveAsTable(df_woe)
    print('\n*** Finished creating a final data set with WOE for fitting...')
    
    ivCalc.unpersist()
    df_bucketed.unpersist()
    df_model.unpersist()
    
def indexCategorical(df,except_for_these):
    indexer = [StringIndexer(inputCol=s[0], outputCol=s[0]+'_indexed',handleInvalid='keep') for s in df.drop(*except_for_these).dtypes if s[1] == 'string']
#    encoder = [OneHotEncoderEstimator(inputCols=[s[0]+'_indexed'],outputCols=[s[0]+"_encoded"],handleInvalid='keep') for s in df.drop(*except_for_these).dtypes if s[1] == 'string']
    pipeline = Pipeline(stages=indexer)
    df2 = pipeline.fit(df).transform(df)
    return df2

def process_sample(df_final, target,except_for_these, n_feat, tmp1, tmp2):
    '''
    Calculate WOE/IV, select features based on IV, replace variable values with WOEs
    '''
    df = df_final
    df.persist()    
    
    decoded_cols = []
    col_decile = {}
    for c in df.drop(*except_for_these).dtypes:
        if c[1] not in ('string', 'timestamp'):
            min_ = float(df.select(c[0]).summary('min').collect()[0][1])
            max_ = float(df.select(c[0]).summary('max').collect()[0][1])
            if min_ < max_ and (max_ - min_ != 0):
                decoded_cols += [c[0]]
                col_decile.setdefault(c[0], []).append(np.linspace(min_, max_, 11).tolist())
    
    ### All variables in buckets
    buckets = [Bucketizer(splits=col_decile[c][0], inputCol = c, outputCol = c+'_bucket', handleInvalid='keep') for c in decoded_cols]
#    buckets = [QuantileDiscretizer(numBuckets = 10, inputCol = c, outputCol = c+"_bucket",handleInvalid='keep') for c in num]
    pipeline = Pipeline(stages=buckets)
    df_bucketed = pipeline.fit(df).transform(df)
    df_bucketed.persist()
    
    print('*** Calculating WOE and IV ....***')
    woe_iv = pd.DataFrame([], columns=['feat_name','bucket','event+non_event','event','tot_event','nonevent','tot_nonevent','pct_event','pct_nonevent','woe','iv'])    
    
    for c in decoded_cols:
        df_woe = (df_bucketed.withColumn('feat_name', F.lit(c))
                              .withColumn('bucket', F.col(c+'_bucket').cast('string'))
                              .select('*', F.count('*').over(W.partitionBy('feat_name')).alias('feat_total'), \
                                      F.count('*').over(W.partitionBy('feat_name', 'bucket')).alias('event+non_event'), \
                                      F.sum(target).over(W.partitionBy('feat_name', 'bucket')).alias('event'))
                              .withColumn('nonevent', F.col('event+non_event')-F.col('event'))
                              .withColumn('tot_event', F.sum(target).over(W.partitionBy('feat_name')))
                              .withColumn('tot_nonevent', F.col('feat_total')-F.col('tot_event'))
                              .withColumn('pct_event', F.when(F.col('event')==0, (.5/F.col('tot_event'))).otherwise(F.col('event')/F.col('tot_event')))
                              .withColumn('pct_nonevent', F.when(F.col('nonevent')==0, (.5/F.col('tot_nonevent'))).otherwise(F.col('nonevent')/F.col('tot_nonevent')))
                              .withColumn('woe', F.log(F.col('pct_nonevent')/F.col('pct_event')))
                              .withColumn('iv', F.col('woe')*(F.col('pct_nonevent')-F.col('pct_event')))
                              .select('feat_name','bucket','event+non_event','event','tot_event','nonevent','tot_nonevent','pct_event','pct_nonevent','woe','iv')
                              .distinct()
                              .orderBy('feat_name', 'bucket'))
        df_tmp = df_woe.toPandas()
        woe_iv = woe_iv.append(df_tmp, ignore_index=True, sort=False)
     
     woe_iv = spark.createDataFrame(woe_iv)
    
    woe_iv.persist()
    print('*** WOE/IV Table ***')
    woe_iv.show(truncate=False)
    woe_iv.write.mode('overwrite').saveAsTable(tmp1)
    
    print('\n*** Feature importance by Information Value >= .02, Top ' + str(n_feat) + ' features ***')
    ivCalc = woe_iv.groupby('feat_name').agg(F.sum('iv').alias('iv')).orderBy('iv', ascending=False)
    ivCalc.show(n_feat, truncate=False)
    
    print('\n*** Replace values with WOEs ***')    
    woe_list = [row.asDict() for row in woe_iv.select('feat_name','bucket','woe').collect()]
    def woe_mapper(feat, bucket):
        for d in woe_list:
            if d['feat_name'] == feat and d['bucket'] == bucket:
                return d['woe']
    
    woe_mapper_udf = F.udf(woe_mapper, DoubleType())
    for c in df_bucketed.columns:
        if c.endswith('_bucket'):
            df_bucketed = df_bucketed.withColumn(c.replace('_bucket','_woe'), F.lit(woe_mapper_udf(F.lit(c[:-len('_bucket')]), F.col(c).cast('string'))))
    
    df_bucketed.write.mode('overwrite').saveAsTable(tmp2)
    print('\n*** Saved df_bucketed having WOEs ***')
    df_bucketed.show(truncate=False)
    
    df.unpersist()
    woe_iv.unpersist()
    df_bucketed.unpersist()
