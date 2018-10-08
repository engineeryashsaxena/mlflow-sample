#!/usr/bin/env python
# coding: utf-8




import mlflow  #import mlflow
import mlflow.sklearn
import pandas as pd  #Import pandas module 
from sklearn.preprocessing import LabelEncoder  #Module to LabelEncode
from sklearn.tree import DecisionTreeClassifier  #importing Decision Tree module 
import sys
from sklearn.metrics import accuracy_score     #Importing accuracy_score

df=pd.read_csv('titanic train.csv')
#Defining a function which removes null values from Age column and label encoding "Sex" column.
def Wrangle(df):
    #remove null values
    df=df.dropna(subset=['Age'])
    #encode categorical variables
    l=LabelEncoder()
    df['Sex']=l.fit_transform(df['Sex'])
    return df

df=Wrangle(df) #wrangling the data
predictors=['Pclass','Age','Sex','Parch','SibSp']   #Choosing a set of predictors
X=df[predictors]  #subset of predictors
Y=df['Survived']  #subset of labels
from sklearn.cross_validation import train_test_split               #Data set split cross validation 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  #training and testing data prepared

random_state = int(sys.argv[1]) if len(sys.argv) > 1 else 0
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3

with mlflow.start_run():
    #mlflow.set_tracking_uri("http://localhost:5000")
    #print "Tracking URI at http://localhost:5000"
    
    
    print "Training Model..."
    clf = DecisionTreeClassifier(random_state=random_state,max_depth=max_depth)     #Creating an instance of Decision Tree Classifier 
    model=clf.fit(X_train,y_train)                   #Training the model 
    
    print "Predicting Values...."
    y_pred=model.predict(X_test) #Predictions for test Data
    Accuracy=accuracy_score(y_pred,y_test)*100.0 
   
    print("Decision Tree model (random_state=%f, max_depth=%f):" % (random_state, max_depth))
    print("  Accuracy: %s" % Accuracy)
    
    #Logging the details 
    mlflow.log_param("random_state", random_state)   
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", Accuracy)
    mlflow.sklearn.log_model(model,"DTModel")   #Model can be logged as an Artifact 
    print "Parameters logged and model saved..."





