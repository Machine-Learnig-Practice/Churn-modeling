# Churn Modelling

Customer churn is basically predicting if the customer will let go of the business with the company or stop being the customer due to certain reasons. Either due to high pricing or not good customer service etc. Churn modelling helps predict which customer will be leaving the company's services and would require some attention if the company wants to retain the customer.

### Prerequisites

The library section:

* import numpy as np
* import pandas as pd
* from sklearn.preprocessing import LabelEncoder, OneHotEncoder
* from sklearn.preprocessing import MinMaxScaler
* from sklearn.model_selection import train_test_split
* import keras
* from keras.layers import *
* from keras.models import Sequential
* from keras.layers import Dense
* from sklearn import svm
* from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
* from sklearn.ensemble import RandomForestClassifier
* from sklearn.model_selection import KFold, cross_val_score
* from sklearn.metrics import precision_recall_fscore_support as score
* from sklearn.tree import DecisionTreeClassifier
* import lightgbm as lgbm
* from sklearn.metrics import mean_squared_error
* from sklearn.ensemble import AdaBoostClassifier
* from sklearn.ensemble import GradientBoostingClassifier
* from sklearn.model_selection import GridSearchCV 

* code was run on Python using anaconda navigator


## Dataset

Kaggle dataset by the name Churn_Modelling.csv

This file has 10,000 entries of customer information with credit score, geography, gender, estimated salary, etc.

### Models built

* Model 1: Neural Network model 1
* Model 2: Neural Network model 2
* Model 3: Support Vector classifier
* Model 4: Random Forest classifier
* Model 5: Decision Tree classifier
* Model 6: AdaBoost classifier
* Model 7: Gradient Boosting classifier


### Metrics used was accuracy, precision, and recall

Summary of models built and comparing their accuracies before and after parameter tuning.

![Image of Results before tuning the parameters](https://github.com/SughoshKulkarni/Churn-Modelling-Mini-Exam-2/blob/master/Result%20Summary%20(Accuracy).PNG)

![Image of Results after tuning the parameters](https://github.com/SughoshKulkarni/Churn-Modelling-Mini-Exam-2/blob/master/Results%20summary%20after%20tuning%20parameters.PNG)


## Built With

* Python from Anaconda navigator


## Acknowledgments

* https://github.com/zszazi
* https://github.com/dolph
* https://gist.github.com/PurpleBooth

