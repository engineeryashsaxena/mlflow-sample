# mlflow-sample

I am using the titanic data from kaggle  . I am splitting the data into 80 % and 20 %  as training and test respectively . I am just assuming some predictors on my own and training the model using Scikit Decision Tree Classifier . I performed training multiple times by providing different values to hyperparameters and achieving different accuracy scores. The algorithm used is Decision Tree and hyperparameters are : 'random_State' and 'max_depth' .The results are observed on UI page now. 


1. TitanicMLFLOW.py file : It contains the complete code ( Comments are inline)
2. titanic data.csv file : This is the dataset 
3. A screenshot png file : This contains screenshot of mlflow ui running on local host . It shows information of all instances with different parameter values and accuracy achieved when I ran the TitanicMLFLOW.py file with different hyperparameters..
4. mlruns directory :  Containing log info 


Note :

Running the TitanicMLFLOW.py file ,there are 2 options : 

Option 1 :  
python TitanicMLFLOW.py    ( it runs with default hyper parameters having random_state as 0 and max_depth as 3 ) 

Option 2 : 
python TitanicMLFLOW.py  <randon_state> <max_depth>

