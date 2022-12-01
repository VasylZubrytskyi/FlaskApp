import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import xgboost

class XGBoostHelper:
    def __init__(self, filePath, target_colum_name, train_size, test_size):
        self.filePath = filePath
        self.target_colum_name = target_colum_name
        self.train_size = train_size
        self.test_size =  test_size

    def __process_data(self, raw_data):
        features = raw_data.select_dtypes(exclude='number').columns
        new_raw_data = pd.get_dummies(raw_data, columns = features)
        scale_variables = raw_data.select_dtypes(include='number').columns
        scaler = MinMaxScaler()
        new_raw_data[scale_variables] = scaler.fit_transform(new_raw_data[scale_variables])
        return new_raw_data

    def startProcess(self):
        raw_data = pd.read_csv(self.filePath)
        new_raw_data = self.__process_data(raw_data)
        
        X = new_raw_data.drop(self.target_colum_name, axis = 1).values
        y = new_raw_data[self.target_colum_name]
        y = y.astype(int)

        fi_df = self.__get_features_importance(new_raw_data, X, y)
        
        colums_to_keep = fi_df[fi_df['Feature Importance'] > 0]
        colums = colums_to_keep['Feature']

        y = new_raw_data[self.target_colum_name]
        X = new_raw_data[colums].values
        
        #X_train, X_test, y_train, y_test = self.__split_data(X, y)
        final_model = self.__find_best_xgboost(X, y)
        final_model.fit(X, y)

        unseen_data = pd.read_csv('C:\\Users\\Vasyl\\Desktop\\practice\\new unseen data.csv', sep=',')
        unseen_data_to_process = self.__process_data(unseen_data)
        unseen_data_to_process = unseen_data_to_process[colums]
        X = unseen_data_to_process[colums].values
        
        pred_final = final_model.predict(X)
        pred_prob_final = final_model.predict_proba(X)
        print(pred_prob_final)
        output = unseen_data.copy()
        output['Predictions - Succeed or Not'] = pred_final
        output['Predictions - Probability to Succeed education'] = self.__column(pred_prob_final, 1)
        output['Predictions - Succeed or Fail Desc'] = 'Empty'
        output['Predictions - Succeed or Fail Desc'][output['Predictions - Succeed or Not'] == 0] = 'Fail'
        output['Predictions - Succeed or Fail Desc'][output['Predictions - Succeed or Not'] == 1] = 'Succeed'
        output.to_csv('output.csv', sep='\t')

        return {
            'CA': 1,
            'Log_loss': 1,
            'score': 1,
        }

    def __column(self, matrix, i):
        return [row[i] for row in matrix]

    def __split_data(self, X, y):
        return train_test_split(X, y, train_size = self.train_size, test_size=self.test_size, random_state=0)

    def __get_features_importance(self, raw_data, X, y):
        dt = DecisionTreeClassifier(random_state=15, criterion = 'entropy', max_depth = 10)
        dt.fit(X,y)
        fi_col = []
        fi = []
        for i,column in enumerate(raw_data.drop(self.target_colum_name, axis = 1)):
            fi_col.append(column)
            fi.append(dt.feature_importances_[i])
        
        fi_df = zip(fi_col, fi)
        fi_df = pd.DataFrame(fi_df, columns = ['Feature','Feature Importance'])
        fi_df = fi_df.sort_values('Feature Importance', ascending = False).reset_index()
        return fi_df

    def __find_best_xgboost(self, X,y):
        classifier=xgboost.XGBClassifier(tree_method='gpu_hist')
        params={
            "learning_rate":[0.05,0.10,0.15,0.20,0.25,0.30],
            "max_depth":[2,3,4,5,6,8,10,12,15],
            "min_child_weight":[1,3,5,7],
            "gamma":[0.0,0.1,0.2,0.3,0.4],
            "colsample_bytree":[0.3,0.4,0.5,0.7]}

        clf =RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',cv=5,verbose=3)
        clf.fit(X,y)
        return clf.best_estimator_
