class classifier_chains():
  def __init__(self):
    import numpy as np
    self.models = {}

  def classfiers(self,name):
      if name=='RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()
      if name=='svm':
        from sklearn import svm
        return svm.SVC()
      if name=='naive_bayes':  
        from sklearn.naive_bayes import GaussianNB       
        return GaussianNB()


  def train_classifiers(self,X,y, classfier_name):
      
      X = np.array(X)
      y = np.array(y)
      self.n_labels = y.shape[1]
      from sklearn.ensemble import RandomForestClassifier
      for i in range(y.shape[1]):
        self.models[i] = self.classfiers(classfier_name)
        self.models[i].fit(np.concatenate((X, y[:,:i]), axis=1), y[:,i])
      print('finished model training')
        # return models

  def models_(self):
      if len(self.models)==0:
        print("Error: models has not been defined neither trained yet. please initiate classifier_chains.train_classifiers(X,y)")
      else:
        return self.models

  def predict(self, X_test):
      X_test = np.array(X_test)
      pred_y = [0]*self.n_labels
      for i in range(self.n_labels):
        pred_y[i] = self.models[i].predict(np.concatenate((X_test, y[:,:i]), axis=1))# will be replaced by a vector 
      return pred_y

  def evaluate(self,pred_y, y_true ,measure='micro-F1'):
        from sklearn.metrics import confusion_matrix
        if measure =='micro-F1':
            tn, fp, fn, tp = sum([confusion_matrix(y_true[:,i], pred_y[i]).ravel() for i in range(self.n_labels)])
            return  ((2* tp) / (2*tp + fp + fn))
        
        if measure =='macro-F1':
            c_mat =   [confusion_matrix(y_true[:,i], pred_y[i]).ravel() for i in range(self.n_labels)]
            return (1/self.n_labels)* sum([((2* i[3]) / (2*i[3] + i[1] + i[2])) for i in c_mat])
			
			
### applying on eron dataset:

import numpy as np
import pandas as pd
data = pd.read_csv('./emotions.csv',header=None)
#data.head()
X, y = np.array(data.iloc[:,:-6]), np.array(data.iloc[:,-6:])

cc = classifier_chain()
cc.trained_models(X,y, 'naive_bayes')
print(cc.n_label())
print(cc.models_())
pred_y = cc.prediction(X)
print(cc.evaluate(pred_y, y , measure ='macro-F1'))			#measure ='micro-F1'
