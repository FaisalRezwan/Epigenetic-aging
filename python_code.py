import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from statistics import mean
from statistics import stdev
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge,LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.svm import LinearSVC,NuSVC
from sklearn.calibration import calibration_curve
from load_dataset import load_data
from sklearn.model_selection import train_test_split,KFold
from collections import Counter
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier

import csv
from sklearn.metrics import roc_curve, auc,make_scorer

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,StratifiedKFold,RepeatedStratifiedKFold,cross_val_predict


from sklearn import preprocessing
colors=np.array(['red','green','blue','orange','purple'])
def getKfolds(X,y):
    import matplotlib.pyplot as plt

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, random_state=33, shuffle=True)
    i=0
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        #print(len(X_train))
        #print(len(X_test))
        #plt.hist(X_train.flatten(),color="green")
        l="Fold#"+str(i+1)
        plt.hist(X_test.flatten(),density=True,label=l,histtype=u'step')
        i+=1
        #plt.show()

        y_train, y_test = y[train_index], y[test_index]
    plt.legend()
    plt.show()
min_max_scaler = preprocessing.MinMaxScaler()





def load_data():
    

    df = pd.read_csv('merged_data.csv')

    ##uncomment the variable if you want to include it as a predictor
    df = df[[
       #'DNAmAge.10',

       #'Age',
       'AgeAccChange',
    #'DNAmAge.18',
    'Gender.10',
      #'AgeAccelerationDiff.18',
       #'AgeAccelerationDiff.10',
       #'AgeAccelerationResidual.18',
       #'AgeAccelerationResidual',
       #'AAHOAdjCellCounts.18',
       #'AAHOAdjCellCounts',
       #'RssChange',
       #'CellCountChange',
       
            
                       
        #               'EVERHADASTHMA_18',
       #'EverEczema_26',
                      # 'ECZEMA_10',
       #'ECZEMA_18',
            
            #'Female',
         #              'HAYFEVER_18',
                       #'HEIGHTCM_10',
       #'WEIGHTKG',
       #'BMI',
            
       #'HAYFEVER_10',
       ##'HayfeverEver_26',
        #'HEIGHTCM_10',
       
       'WEIGHTKG_18',
                        #'FVC_10',
       'FEV1.10_New',
        #'FVC_10YR',
        #'FVC_18YR',



        #'FEV1Change',
       #'FEV1Change',
       #'BMI_18',
       #'BMI_26',
       #'FEV1_New',
        #    'DOYOUCURRENTLYSMOKE_18',
                       #'FEV1_18',
       #'FVC',
     'FEV1.18_New',
     #  #'EVERHADASTHMA_10',
     'HEIGHTCM_18',
       #'FEV1ByFVC_New',
       
       #'ECZEMA',
        #'FVC_18'
      #'EVERHADASTHMA_18',
       #'FEV1.18_New'
       #'FEV1ByFVC.18_New'
       #,'FEV1_PREDICTED_18'
       #'EverAsthma_26'
       #'HayfeverEver_26'
              ]]
    print("Column headings:")
    print(df.columns)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)
    pred_val_name = 'FEV1.18_New'#'FEV1.18_New'  # set the class column name
    print(len(df))
    print(df[0:5])
    
    X = df.drop(pred_val_name, axis=1)

    x = X.values #returns a numpy array
    
    x_scaled = min_max_scaler.fit_transform(x)
    dfX = pd.DataFrame(x_scaled,columns = X.columns)
    y = df.pop(pred_val_name).to_frame()

    X_scaled = dfX.copy()
    print(X_scaled[0:5])
    return pred_val_name,X_scaled,y
    #return pred_val_name,X,y


pred_val_name,X_scaled,y = load_data()

X = X_scaled.copy()

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.feature_selection import f_regression,mutual_info_regression

F,p = f_regression(X,y)
mi = mutual_info_regression(X,y)
print(F)
print(p)
print(X.columns)
print(mi)
estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=10)
selector = selector.fit(X, y)
print(selector.support_ )


print(selector.ranking_)

#kf = RepeatedStratifiedKFold(n_splits=5)


#clf = SGDClassifier(max_iter=1000, tol=1e-3)
#clf = GradientBoostingClassifier(random_state=0)
#clf = AdaBoostClassifier(random_state=0)
#clf = LogisticRegression(random_state=0)
#clf = RandomForestClassifier(n_estimators=50,random_state=0)
#clf = RandomForestRegressor()
#clf=GradientBoostingRegressor()
#clf = ExtraTreesRegressor()
#clf = AdaBoostRegressor(base_estimator=clf1)
#clf=Ridge(alpha=0.4)
#clf=Lasso(alpha=0.0001)
#clf = ElasticNet(alpha=0.001)
clf = LinearRegression()

scores = cross_val_score(clf, X.values, y.values.ravel(),cv=10,scoring='r2')
print("R^2: ",(round(mean(scores), 4))," +/- ",(round(stdev(scores), 4)))

scores = cross_val_score(clf, X.values, y.values.ravel(),cv=10,scoring='neg_root_mean_squared_error')
print("RMSE: ",(round(-mean(scores), 4))," +/- ",(round(stdev(scores), 4)))
#print((round(stdev(scores), 4)))

exit()
# results_ridge=[]
results=[]
alphas=np.linspace(0.001,.005 , num=10)
# print(alphas)
# for i in alphas:
    #print(i)

#print(scores)
# y_pred = cross_val_predict(clf, X.values, y.values, cv=10)
#
# actual_y=y.values
#
# for i in range(len(y_pred)):
#     if abs(y_pred[i]-actual_y[i])<=1:
#         y_pred[i]=actual_y[i]
# from sklearn.metrics import mean_squared_error
# import math
# print(math.sqrt(mean_squared_error(y.values,y_pred)))
# for i in range(len(alphas)):
#     clf = ElasticNet(alpha=alphas[i])
#
#     scores = cross_val_score(clf, X.values, y.values.ravel(), cv=10)
#     results.append((round(mean(scores),3)))
    #print((round(stdev(scores), 4)))

#print("R^2")
# print((round(mean(scores), 4)))
# print((round(stdev(scores), 4)))
# exit()

# results_ridge.append(round(mean(scores), 4))
# clf = Lasso(alpha=i)
#
# scores = cross_val_score(clf, X.values, y.values.ravel(), cv=10)
#
# results_lasso.append(round(mean(scores), 4))


# results_ridge=np.array(results_ridge)
#results=np.array(results)


# import matplotlib.pyplot as plt
# font = {'family' : 'serif',
#         'weight' : 'medium',
#         'size'   : 13}
#
# plt.rc('font', **font)


# plt.scatter(y.values,y_pred,color="springgreen")
# plt.plot(y.values,y.values,'r')
# plt.xlabel("True FEV1 at age 18")
# plt.ylabel("Predicted FEV1 at age 18")
# plt.title("True vs Predicted FEV1")
# plt.show()
# #
# plt.plot(alphas,results,color='red', marker='^', linestyle='dashed',
#     linewidth=2, markersize=12)
# #plt.plot(alphas,results_lasso,color='green', marker='^', linestyle='dashed',
# #    linewidth=2, markersize=12)
# plt.title("Impact of $alpha$ on ElasticNet Regression")
# plt.xlabel("$alpha$")
# plt.ylabel("Mean $R^2$")
# plt.grid(ls='--')
# #plt.ylim(0.748,0.749)
# plt.show()
from sklearn.linear_model import ElasticNet
#clf= ElasticNet(alpha=.001)
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import BayesianRidge,LinearRegression
#clf=LinearRegression()

#clf=BayesianRidge(alpha_1=.01, lambda_1=.01,lambda_2=.01)
#print(y)




