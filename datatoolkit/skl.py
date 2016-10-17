"""
Some misc utilities to make my life easier when using scikit-learn 
"""

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion

class Columns_Selector(BaseEstimator, TransformerMixin):
    """

    This can be handy in order to apply some transformation only on a set of columns. 

    Only works if the input is a pandas dataframes.

    Pipeline([
        ("num_cols", Columns_Selector(["temp", "humidity", "windspeed", "registered_hourly_distro"])),
        ("scaler", StandardScaler())        
        ]))

    This is heavily inspired by the excellent blog post by Zac Stewart: 
    http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html

    """
    
    def __init__(self, cols):
        self.cols = cols
            
    def fit(self, x, y=None, **fit_params):
        return self
    
    def transform(self, x, y=None, **fit_params):
        return x[self.cols]
    

class TransformingPredictorPipeline(Pipeline):
    """
    This allows to convert a predicting pipeline into a transforming pipeline. 
    """
    
    def __init__(self, steps):
        Pipeline.__init__(self, steps)
        
    def transform(self, X):
        return self.predict(X)    
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).predict(X)
    

class PredictingTransformerPipeline(Pipeline):
    """
    This is the reverse conversion of the above: we transform a transforming pipeline 
    into a predicting pipeline.     
    """
    
    def __init__(self, steps):
        Pipeline.__init__(self, steps)
        
    def predict(self, X):
        return self.transform(X)
    
    def fit_predict(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)    
    


class Label_Encoder_Pipeok(BaseEstimator, TransformerMixin):
    """
    Label encoder that can be used inside a pipeline: encoding each column separately 

    This enables to write pipelines like this: 

    Pipeline([
                ("nom_cols", skl.Columns_Selector(["season", "weather"])), 
                ("nom_to_num", skl.Label_Encoder_Pipeok()),        
                ("one_so_hot", OneHotEncoder(sparse=False))        
            ])
    
    But actually, this is not a good idea: 

    1) Due to cross validation behaviour, we can end up with some labels in 
       the validation set that have not been witnessed in the training set. 

    2) This leads to a negative performance impact during grid-search x-fold 
       cross-validation since we re-encode the data at every fold. 

    """
    
    def __init__(self):
        self.encoders = []
                
    def fit(self, x, y=None, **fit_params):
        self.encoders = [LabelEncoder().fit(x.ix[:,i].tolist()) for i in range(x.shape[1])]        
        return self
    
    def transform(self, x, **fit_params):        
        cols = [self.encoders[i].transform(x.ix[:,i]) for i in range(x.shape[1])]
        return np.array(cols).T

