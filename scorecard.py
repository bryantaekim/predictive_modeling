import matplotlib.pyplot as plt, numpy as np
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('score_card').getOrCreate()
graph_loc = 'somewhere'

def scorecard(clf, input_tbl, df, target, filename):
    '''
    df : pandas df with WoE
    
    Scorecard - general formula used in credit scoring 
    factor = pdo / np.log(2)
    offset = target_score - (factor * np.log(odds))
    score = (beta * woe + alpha / n) * factor + offset/n
    
    '''
    df_feat = df.drop(['party_id', target], axis=1)
    
    incpt = clf.intercept_
    model_params = pd.DataFrame(clf.coef_.ravel(), columns=['coefficient'], index=df_feat.columns)
        
    scorecard = df_feat[model_params.index].apply(lambda x: x*model_params['coefficient'].T, axis=1)
    scorecard[target] = df[target]
    scorecard['logit'] = scorecard[scorecard.columns].sum(axis=1) + incpt
    scorecard['odds'] = np.exp(scorecard['logit'])
    scorecard['probs'] = scorecard.odds / (1 + scorecard.odds)
    
    # Scoring baseline
    target_score = 600
    pdo = 40
    _for = len(df[df[target] == 1])
    _against = len(df[df[target] == 0])
    target_odds = float(_for / _against)
    
    factor =  pdo / np.log(2)

    scorecard['score'] = target_score - (factor * np.log(target_odds)) + factor * scorecard['logit']
    scorecard['pred_prob_0'] = clf.predict_proba(df_feat)[:,0]
    scorecard['pred_prob_1'] = clf.predict_proba(df_feat)[:,1]
    scorecard['decile'] = pd.qcut(scorecard['pred_prob_1'].rank(method = 'first'), 10, labels = range(10,0,-1)).astype(float)
    
    print(scorecard.head())
    
    scores = spark.createDataFrame(scorecard)
    scores.write.mode('overwrite').saveAsTable(input_tbl + '_prediction_' + target)
    scores.groupby(target, 'decile').count().show()
    
    # Plot Distribution of Scores
    plt.figure(figsize=(12,8))
    
    plt.hist(scorecard['score'], bins=200, edgecolor='white', color = '#317DC2', linewidth=1.2)
    plt.title('Scorecard Distribution', fontweight="bold", fontsize=14)
    plt.axvline(scorecard['score'].mean(), color='k', linestyle='dashed', linewidth=1.5, alpha=0.5)
    plt.xlabel('Score')
    plt.ylabel('Count')
    
    plt.savefig(graph_loc + 'score_distr_'+ filename + '.jpg') 
    plt.show()
    plt.close()
