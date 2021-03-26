# COMP3314 Assignment 1 

## 1. Logistic Regression
### __Overview of the implement:__

  Logistic Regression is kind of linear classification algorithm.        
  For ML training use, only need to visit functions in MultiLogisticRegression class.  

### __Class:__
#### __\[internal\]LogisticRegression__

    Logistic Regression for binary classification. 
    
 * _Variables_      
        
        eta: learing rate
        n_iter: training times
        random_state: random functionn parameter
        C: parameter for regularizationn. To avoid weight bias.
        w_: weight vector for features
        cost_: cost record for the whole training process
 
 * _\[internal\]LogisticRegression(eta, n_iter, random_state, C)_    
 
        eta: learning rate  
        n_iter: training times  
        random_state: random functionn parameter  
        C: parameter for regularizationn.  
    __e.g.__  lgr = LogisticRegression(eta=0.01, n_iter=50, random_state=1, C=10000)
    
 * _\[internal\]fit(X, y)_    Main function for training. 
    
        X: training feature set     
        y: training label set   
    __e.g.__  lgr.fit(X_train, y_train)
    
 * _\[internal\]probability(X)_    Give the probability of being in current class. 
    
        X: testing feature set  
    __e.g.__  prob = lgr.probability(X_test)
    
 * _\[internal\]predict(X)_    Give a prediction for given feature set. 
     
        X: testing feature set  
    __e.g.__  y_pred = lgr.predict(X_test)
  
#### __MultiLogisticRegression__

    Logistic Regression for multiclass classification.      
    Train binary classifier for each class, and come up with the predistion with highest probability. 
    
  * ___Variables___      
        
        eta: learing rate
        n_iter: training times
        random_state: random functionn parameter
        C: parameter for regularizationn. To avoid weight bias.
        lgrs: list of logistic regression classifiers
        classes: list of class corresponding to each classifier
 
 * ___MultiLogisticRegression(eta, n_iter, random_state, C)___    
 
        eta: learning rate  
        n_iter: training times  
        random_state: random functionn parameter  
        C: parameter for regularizationn. To avoid weight bias.  
    __e.g.__  mlr = MultiLogisticRegression(eta=0.01, n_iter=50, random_state=1, C=10000)
    
  * ___fit(X, y)___    Main function for training. 
      
        X: training feature set     
        y: training label set   
    __e.g.__  mlr.fit(X_train, y_train)
    
 * ___probability(X)___    Give a list of probability of being in current class for each sample. 
       
        X: testing feature set  
    __e.g.__  prob = mlr.probability(X_test)
    
 * ___predict(X)___    Give a prediction for given feature set for each sample. 
       
        X: testing feature set  
    __e.g.__  y_pred = mlr.predict(X_test)
    
## 2. Random Forest
### __Overview of the implement:__

  The Random Forest classification algorithm is developed based on Decision Tree algorithm. In this algorithm, sample function is used to randomly sample from the origional data set, in order to build a forest of random decision trees. 
  For ML training use, only need to use functions in RandomForest class. 

### __Class:__
#### __\[internal\]TreeNode__

    To describe a single node in decision tree.  
    
 * _Variables_      
        
        X, y: features and labels of samples contained in current node
        isLeaf: whether current node is leaf node or not
        split_feature, split_value: feature and critical value used to devide subnodes
        childs: list of subnodes. usually have 2. 
        info: further information discribing current node. Format is (node depth, current impurity, (size of 1st childnode, size of 2nd child node)
        classes: list of sample classes contained in current node
        class_: Mode of classes. The overall class of current node
        
 * _\[internal\]TreeNode(X, y, childs, info, isLeaf, split_feature, split_value)_    
 
        X: feature matrix for sample in current node
        y: label array for sample in current node
        info: further information discribing current node.
        isLeaf: whether current node is leaf node or not
        split_feature: feature index used to devide subnodes
        split_value: critical value used to devide subnodes
        
    __e.g.__  tn = TreeNode(X=X, y=y, childs=childs, info=info, isLeaf=False, split_feature=feature_best, split_value=value_best)
    
 * _\[internal\]predict_single(X)_    Presict the class for single sample 
    
        X: feature array for the single sample   
    __e.g.__  y_pred = tn.predict_single(x_test)
    
* _\[internal\]predict(X)_    Predict the classes for an nparray of samples

        X: feature matrix for the group of samples
 
    __e.g.__  y_pred = tn.predict(X_test)

#### __\[internal\]DecisionTree__

    To build a decision tree based on a set of training data.   
    
 * _Variables_      
        
        max_depth: maximum depth for the decision tree
        min_gain: minimum gain for each division
        impurity_mode: impurity calculate mode.
        root: tree root node.
        feature index: to pick features to develop current tree
        
 * _\[internal\]DecisionTree(impurity_mode, max_depth, min_gain)_    
 
        max_depth: maximum depth for the decision tree. default=5
        min_gain: minimum gain for each division. default=0.0
        impurity_mode: impurity calculate mode. Can choose from 'gini', 'entropy', 'ce'. default='gini'
        
    __e.g.__  tree = DecisionTree(impurity_mode='gini', max_depth=10, min_gain=0.0)
    
 * _\[internal\]impurity(y):_    Calculate the impurity of a set of samples based on impurity mode
    
        y: label array of samples
        
    __e.g.__  imp_currnt = tree.impurity(y)
 
 * _\[internal\]split(X, y, feature, value):_    Binary split the samples. 
    
        X: feature matrix of samples
        y: label array of samples
        feature: feature index used in the division
        value: boundary value. If it is number, the split according to >= and <. Else split according to whether equal. 
        
        Return two lists. One has two arrays of Xs, another has two arrays of ys accordingly. 
        
    __e.g.__  childnode_X, childnode_y = tree.split(X, y, feature=1, value='yellow')
    
 * _\[internal\]build_decision_tree(X, y, depth):_    Main function used to develop decision tree. 
    
        X: training feature set     
        y: training label set 
        depth: to record the current depth. No need to set when developing a new tree outside. 
        
        After built, can use tree.root to start visiting the whole tree
        
    __e.g.__  tree.build_decision_tree(X_train, y_train)
    
* _\[internal\]set_feature_index(feature_index)_    Record the feature index after sampling

        feature_index: list of index for features used in current tree
 
    __e.g.__  tree.set_feature_index(sample_col_index)

* _\[internal\]predict_single(X)_    Presict the class for single sample 

        X: feature array for the single sample   
 
    __e.g.__  y_pred = tree.predict_single(x_test)

* _\[internal\]predict(X)_    Predict the classes for an nparray of samples

         X: feature matrix for the group of samples
 
    __e.g.__  y_pred = tree.predict(X_test)
   
#### __RandomForest__

    To implement random forest algorithm.   
    
 * ___Variables___      
        
        n_estimation: number of decision trees in the system. 
        feature_percentage: proportion of features chosed for each tree. 
        sample_percentage: proportion of training samples number and total sample number. 
        random_state: random function parameter
        impurity_mode: impurity calculate mode
        max_depth: maximum depth for the decision tree
        min_gain: minimum gain for each division
        showProcess: whether print sample number and feature number in each training to show the process
        trees: list of trees in the forset
        
 * ___RandomForest(n_estimation, feature_percentage, sample_percentage, random_state, impurity_mode=, max_depth, min_gain, showProcess)___    
 
        n_estimation: number of decision trees in the system. default=100
        feature_percentage: proportion of features chosed for each tree. Usually low. defalut=0.3
        sample_percentage: proportion of training samples number and total sample number. Usually close to 1. defalut=0.95
        random_state: random function parameter. defalut=1
        impurity_mode: impurity calculate mode. defalut='gini'
        max_depth: maximum depth for the decision tree. defalut=8
        min_gain: minimum gain for each division. defalut=0.0
        showProcess: whether print sample number and feature number in each training to show the process. defalut=False
        
    __e.g.__  rf = RandomForest(n_estimation=100, sample_percentage=0.95, feature_percentage=0.25, max_depth=10, showProcess=False)
    
 * _\[internal\]sample(X, y):_    Function for random sampling. 
    
        X: training feature set     
        y: training label set  
        
        Return an array of sampled X, an array for corresponding y, a list of feature index which should be set to the tree developed based on these data. 
        
    __e.g.__  X_sample, y_sample, feature_index = rf.sample(X, y)
    
* ___fit(X, y)___    Main function for training the Random forest

        X: training feature set     
        y: training label set 
 
    __e.g.__  rf.fit(X_train, y_train)

* ___predict_single(X)___    Presict the class for single sample 

        X: feature array for the single sample   
 
    __e.g.__  y_pred = rf.predict_single(x_test)

* ___predict(X)___    Predict the classes for an nparray of samples

         X: feature matrix for the group of samples
 
    __e.g.__  y_pred = rf.predict(X_test)
