# COMP3314 Assignment 1 

## 1. Logistic Regression
### __Overview of the implement:__

  Logistic Regression is kind of linear classification algorithm. For 

### __Class:__
#### __LogisticRegression__

    Logistic Regression for binary classification. 
    
 * ___Construntion___    
 
    __parameters:__   
        eta: learning rate  
        n_iter: training times  
        random_state: random functionn parameter  
        C: parameter for regularizationn. To avoid weight bias.  
    __e.g.__  lgr = LogisticRegression(eta=0.01, n_iter=50, random_state=1, C=10000)
    
 * ___fit(X, y)___    Main function for training. 
    
    __parameters:__     
      X: training feature set     
      y: training label set   
    __e.g.__  lgr.fit(X_train, y_train)
    
 * ___probability(X)___    Give the probability of being in current class. 
    
    __parameters:__     
      X: testing feature set  
    __e.g.__  prob = lgr.probability(X_test)
    
 * ___predict(X)___    Give a prediction for given feature set. 
    
    __parameters:__     
      X: testing feature set  
    __e.g.__  y_pred = lgr.predict(X_test)
  
#### __MultiLogisticRegression__

    Logistic Regression for multiclass classification.      
    Train binary classifier for each class, and come up with the predistion with highest probability. 
    
 * ___Construntion___    
 
    __parameters:__   
        eta: learning rate  
        n_iter: training times  
        random_state: random functionn parameter  
        C: parameter for regularizationn. To avoid weight bias.  
    __e.g.__  mlr = MultiLogisticRegression(eta=0.01, n_iter=50, random_state=1, C=10000)
    
  * ___fit(X, y)___    Main function for training. 
    
    __parameters:__     
      X: training feature set     
      y: training label set   
    __e.g.__  mlr.fit(X_train, y_train)
    
 * ___probability(X)___    Give a list of probability of being in current class for each sample. 
    
    __parameters:__     
      X: testing feature set  
    __e.g.__  prob = mlr.probability(X_test)
    
 * ___predict(X)___    Give a prediction for given feature set for each sample. 
    
    __parameters:__     
      X: testing feature set  
    __e.g.__  y_pred = mlr.predict(X_test)
    
## 2. Random Forest
### __Overview of the implement:__

  The Random Forest classification algorithm is developed based on Decision Tree algorithm. In this algorithm, sample function is used to randomly sample from the origional data set, in order to build a forest of random decision trees. 

### __Class:__
