
# coding: utf-8

# In[1]:

# get_ipython().run_line_magic('pylab', 'inline')

import os
import gc
import psutil

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold, StratifiedKFold # KFold as RegKFold, StratifiedKFold
from sklearn.feature_selection import RFECV, chi2, f_regression, f_classif, mutual_info_classif, mutual_info_regression, SelectKBest, SelectFpr, SelectFdr

import itertools

import copy as cp


# ### Memory Cleaner

# In[2]:

def memory_cleaner():
    proc = psutil.Process(os.getpid())
    gc.collect()
    mem0 = proc.memory_info().rss
    
    # create approx. 10**7 int objects and pointers
    foo = ['abc' for x in range(10**7)]
    mem1 = proc.memory_info().rss
    
    # unreference, including x == 9999999
    del foo
    mem2 = proc.memory_info().rss

    # collect() calls PyInt_ClearFreeList()
    # or use ctypes: pythonapi.PyInt_ClearFreeList()
    gc.collect()
    mem3 = proc.memory_info().rss

    pd = lambda x2, x1: 100.0 * (x2 - x1) / mem0
    print ("Allocation: %0.2f%%" % pd(mem1, mem0))
    print ("Unreference: %0.2f%%" % pd(mem2, mem1))
    print ("Collect: %0.2f%%" % pd(mem3, mem2))
    print ("Overall: %0.2f%%" % pd(mem3, mem0))


# ### Confusion Matrix

# In[3]:

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# ### Model Testers

# #### Classifier

# In[4]:

def classification_model(model, data, predictors, outcome):
    memory_cleaner()
    print(model)
    
    #Fit the model:
    model.fit(data[predictors], data[outcome])
    
    #Make predictions on training set:
    predictions = model.predict(data[predictors])
    
    #Print accuracy
    accuracy = metrics.accuracy_score(data[outcome], predictions)
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    
    #Perform k-fold cross-validation with 5 folds
    cv = StratifiedKFold(n_splits=5)
    results = []
    
    #Generate Confusion Matrix and Classification Report
    classes = [str(x) for x in list(data[outcome].unique())]
    cnf_matrix = metrics.confusion_matrix(data[outcome], predictions)
    report = metrics.classification_report(data[outcome], predictions, digits=4, target_names=classes)
    
    X = data[predictors]
    y =  data[outcome]
    for train_index, test_index in cv.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Training the algorithm using the predictors and target.
        model.fit(X_train, y_train)
        #Record error from each cross-validation run
        results.append(model.score(X_test, y_test))
    
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(results)))
    print()
    print()
    
    #Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors],data[outcome])
    
    return (results, cnf_matrix, report)


# In[5]:

def test_classifiers(modelNameList, modelList, data, predictors, outcome):
    
    if len(modelNameList) != len(modelList):
        raise Exception('Length of the lists modelNameList and modelList should match')
    
    kfold_results = []
    cnf_matrixs = []
    reports = []
    
    for model in modelList:
        result = classification_model(model, data, predictors, outcome)
        kfold_results.append(result[0])
        cnf_matrixs.append(result[1])
        reports.append(result[2])
    
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(kfold_results)
    ax.set_xticklabels(modelNameList)
    plt.show()
    
    classes = list(data[outcome].unique())
    
    i = 0
    for cm in cnf_matrixs:
        plot_confusion_matrix(cm, classes=classes, title=modelNameList[i])
        print('------------ REPORT ------------')
        print(reports[i])
        print()
        
        i+=1


# In[12]:

def classification_roc_auc_test(model, data, predictors, outcome):
    memory_cleaner()
    print(model)
    
    #Fit the model:
    model.fit(data[predictors], data[outcome])
    
    #Prediction probability on training set:
    pred_prob = model.predict_proba(data[predictors])
    
    #Print roc_auc_score
    score = metrics.roc_auc_score(data[outcome], pred_prob[:,1])
    print("ROC AUC Score : %s" % "{0:.4}".format(score))
    
    #Perform k-fold cross-validation with 5 folds
    cv = StratifiedKFold(n_splits=5)
    results = []
    
    X = data[predictors]
    y =  data[outcome]
    for train_index, test_index in cv.split(X, y):
        # Filter training data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		
        # Training the algorithm using the predictors and target.
        model.fit(X_train, y_train)
        pred_prob_fold = model.predict_proba(X_test)
        score_fold = metrics.roc_auc_score(y_test, pred_prob_fold[:,1])
        #Record error from each cross-validation run
        results.append(score_fold)
    
    print("Cross-Validation ROC AUC Score : %s" % "{0:.4}".format(np.mean(results)))
    print()
    print()
    
    return results


# In[8]:

def test_classifiers_roc_auc_score(modelNameList, modelList, data, predictors, outcome):
    
    if len(modelNameList) != len(modelList):
        raise Exception('Length of the lists modelNameList and modelList should match')
    
    kfold_results = []
    
    for model in modelList:
        result = classification_roc_auc_test(model, data, predictors, outcome)
        kfold_results.append(result)
    
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(kfold_results)
    ax.set_xticklabels(modelNameList)
    plt.show()


# #### Regressor

# In[7]:

def regression_model(model, data, predictors, outcome):
    memory_cleaner()
    print(model)
    
    #Fit the model:
    model.fit(data[predictors], data[outcome])
    
    #Make predictions on training set:
    predictions = model.predict(data[predictors])
    
    #Print accuracy
    accuracy = np.sqrt(metrics.mean_squared_error(data[outcome], predictions))
    print("RMSE : " + str(accuracy))
    
    #Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
   
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train,:])
        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]
        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
        #Record error from each cross-validation run
        predictions = model.predict(data[predictors].iloc[test,:])
        accuracy = np.sqrt(metrics.mean_squared_error(data[outcome].iloc[test], predictions))
        error.append(accuracy)
    
    print("Cross-Validation RMSE : " + str(np.mean(error)))
    print()
    print()
    
    #Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors],data[outcome])
    
    return error


# In[9]:

def test_regressors(modelNameList, modelList, data, predictors, outcome):
    
    if len(modelNameList) != len(modelList):
        raise Exception('Length of the lists modelNameList and modelList should match')
    
    kfold_results = []

    for model in modelList:
        result = regression_model(model, data, predictors, outcome)
        kfold_results.append(result)
    
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(kfold_results)
    ax.set_xticklabels(modelNameList)
    plt.show()


# ### Recursive Feature Elimination CV

# In[6]:

def rec_feature_elimination(data, predictors, outcome, model, problem_type='classification', model_scoring='accuracy'):
    # X, y
    X = data[predictors]
    y = data[outcome]
    
    rfecv = None
    
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    if problem_type == 'classification':
        rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring=model_scoring)
    elif problem_type == 'regression':
        rfecv = RFECV(estimator=model, step=1, cv=KFold(5), scoring=model_scoring) # RegKFold(5), scoring=model_scoring)
    else:
        raise Exception(str(problem_type) + ' is not a valid problem_type. Valid problem_type is one of [classification, regression]')
    
    rfecv.fit(X, y)
    
    # Optimal Feature Selection
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    feature_index = [zero_based_index for zero_based_index in list(rfecv.get_support(indices=True))]
    optimal_predictors = []
    for i in feature_index:
        optimal_predictors.append(predictors[i])
    
    print(optimal_predictors)
    print()
    
    high_score = 0.0
    low_score = 999999999.99
    for score in rfecv.grid_scores_:
        if score > high_score:
            high_score = score
        if score < low_score:
            low_score = score
    
    print('Lowest Score: ' + str(low_score))
    print('Highest Score: ' + str(high_score))
    print()
    
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    return optimal_predictors


# ### K-Feature Selection

# In[6]:

def select_features(data, features, target, feature_selector='SelectKBest', k=10, alpha=0.05, score_func='f_classif'):
    X = data[features]
    y = data[target]
    
    if score_func == 'f_classif':
        score_func = f_classif
    elif score_func == 'f_regression':
        score_func = f_regression
    elif score_func == 'chi2':
        score_func = chi2
    elif score_func == 'mutual_info_classif':
        score_func = mutual_info_classif
    elif score_func == 'mutual_info_regression':
        score_func = mutual_info_regression
    else:
        raise Exception('Undefined score_func')
    
    
    if feature_selector == 'SelectKBest':
        feature_selector = SelectKBest(score_func=score_func, k=k)
    elif feature_selector == 'SelectFpr':
        feature_selector = SelectFpr(score_func=score_func, alpha=alpha)
    elif feature_selector == 'SelectFdr':
        feature_selector = SelectFdr(score_func=score_func, alpha=alpha)
    else:
        raise Exception('Undefined score_func')
    
    feature_selector.fit_transform(X, y)
    feature_index = [zero_based_index for zero_based_index in list(feature_selector.get_support(indices=True))]
    
    best_features = []
    for i in feature_index:
        best_features.append(features[i])
    
    print('Best features selected are: ' + str(best_features))
    
    return best_features


# ### Recursive Model Learner

# In[2]:

def recursive_train(data, predictors, outcome, prediction_model, learning_model, learning_depth=2):
    
    m = []
    pred = [None] * learning_depth
    missmatch_pred = [None] * learning_depth
    lm = []
    
    for i in range(0, learning_depth):
        m.append(cp.copy(prediction_model))
        lm.append(cp.copy(learning_model))
    
    for i in range(0, learning_depth):
        m[i].fit(data[predictors], data[target])
        
        pred[i] = m[i].predict(data[predictors])
        print('Prediction Accuracy Score after iteration ' + str(i) + " is: " 
              + str(metrics.accuracy_score(data[outcome], pred[i])))
        
        data['pred' + str(i)] = pred[i]
        data['missmatch' + str(i)] = np.where(data[outcome] == data['pred' + str(i)], 0, 1)
        
        lm[i].fit(data[predictors], data['missmatch' + str(i)])
        
        missmatch_pred[i] = lm[i].predict(data[predictors])
        print('Learning Accuracy Score after iteration ' + str(i) + " is: " 
              + str(metrics.accuracy_score(data['missmatch' + str(i)], missmatch_pred[i])))
        
        data = data[(data['missmatch' + str(i)] == 1)]
        data.reset_index(drop=True, inplace=True)
        
    return (m, lm)


# In[3]:

def recursive_predict(data, predictors, outcome, pred_model_list, learning_model_list, is_test=False):
    
    final_prediction = None
    
    depth = len(pred_model_list)
    
    m = pred_model_list[0]
    lm = learning_model_list[0]
    
    data['pred' + str(0)] = m.predict(data[predictors])
    data['final_pred' + str(0)] = data['pred' + str(0)]
    data['missmatch' + str(0)] = lm.predict(data[predictors])
    
    print('Potential missmatch learned after iteration 0: ' + str(data['missmatch' + str(0)].sum()))
    
    if is_test:
        print('Accuracy after iteration 0: ' + str(metrics.accuracy_score(data[outcome], data['final_pred' + str(0)])))
    
    for i in range(1, depth):
        m = pred_model_list[i]
        lm = learning_model_list[i]
        
        data['pred' + str(i)] = m.predict(data[predictors])  
        data['missmatch' + str(i)] = lm.predict(data[predictors])
        
        print('Potential missmatch learned after iteration ' + str(i) + ': ' + str(data['missmatch' + str(i)].sum()))
        
    for i in range(1, depth):
        where_missmatch = []
        for j in range(0, i):
            where_missmatch.append('missmatch' + str(j))
        
        truth_values = []
        for m_col in where_missmatch:
            if len(truth_values) == 0:
                truth_values = np.array(data[m_col] == 1)
            else:
                truth_values = truth_values & np.array(data[m_col] == 1)
        
        data['final_pred' + str(i)] = np.where(truth_values, data['pred' + str(i)], data['final_pred' + str(i-1)])
        # print(truth_values)
        # print()
        # np.where(test_new['MissMatch'] == 1, test_new['Pred2'], test_new['Pred1'])
            
        if is_test:
            print('Accuracy after iteration ' + str(i) + ': ' + 
                 str(metrics.accuracy_score(data[outcome], data['final_pred' + str(i)])))
            
        final_prediction = data['final_pred' + str(i)]
    
    return final_prediction


# In[ ]:



