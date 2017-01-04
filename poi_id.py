# coding: utf-8

# In[1]:

#!/usr/bin/python

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[3]:

def maxPrinter(data_dict, sub_field):
    sub_dict = {key:data_dict[key][sub_field] for key in data_dict if data_dict[key][sub_field] != 'NaN'}
    max_key = max(sub_dict, key=sub_dict.get)
    print "Max", sub_field, max_key, sub_dict[ max_key ]

def removeMax(data_dict, sub_field):
    sub_dict = {key:data_dict[key][sub_field] for key in data_dict if data_dict[key][sub_field] != 'NaN'}
    max_val = max(sub_dict, key=sub_dict.get) 
    data_dict.pop(max_val, 0)

maxPrinter(data_dict, 'salary')
print 'remove max'
removeMax(data_dict, 'salary')
maxPrinter(data_dict, 'salary')


# In[4]:

### Task 3: Create new feature(s)
#email_to_poi_ratio
#email_from_poi_ratio
for key in data_dict.keys():
    email_to = data_dict[key]['to_messages']
    to_poi = data_dict[key]['from_this_person_to_poi']
    email_from = data_dict[key]['from_messages']
    from_poi = data_dict[key]['from_poi_to_this_person']
    exer_stock_opt = data_dict[key]['exercised_stock_options']
    total_stock = data_dict[key]['total_stock_value']
    
    if 'NaN' not in (email_to, to_poi):
        data_dict[key]['email_to_poi_ratio'] = float(to_poi)/float(email_to)
    else:
        data_dict[key]['email_to_poi_ratio'] = 'NaN'
    if 'NaN' not in (email_from, from_poi):
        data_dict[key]['email_from_poi_ratio'] = float(from_poi)/float(email_from)
    else:
        data_dict[key]['email_from_poi_ratio'] = 'NaN'
    if 'NaN' not in (exer_stock_opt, total_stock):
        data_dict[key]['exer_stock_ratio'] = float(exer_stock_opt)/float(total_stock)
    else:
        data_dict[key]['exer_stock_ratio'] = 'NaN'


# In[5]:

def my_k_fold_test_short (classifier, features, labels, kval=10):
    from sklearn.model_selection import KFold
    k_fold = KFold(kval)

    #print "Iteration: precision, recall, f1"
    precision = []
    recall = []
    f1 = []

    for k, (train, test) in enumerate(k_fold.split(features, labels)):
        scores = scoreClassifier (classifier,
                                     [features[ii] for ii in train], [features[ii] for ii in test],
                                     [labels[ii] for ii in train], [labels[ii] for ii in test]) 
        #print "#", k,":", scores
        precision.append(scores[0])
        recall.append(scores[1])
        f1.append(scores[2])
    avPrecision = sum(precision)/kval
    avRecall = sum(recall)/kval
    avF1 = sum(f1)/kval
    print 'Precision', avPrecision, ', Recall', avRecall, ', F1', avF1

def scoreClassifier(classifier, features_train, features_test, labels_train, labels_test):
    from sklearn import metrics
    classifier.fit(features_train, labels_train)
    test_pred = classifier.predict(features_test)
    precision = metrics.precision_score(labels_test, test_pred)
    recall = metrics.recall_score(labels_test, test_pred)  
    f1 = metrics.f1_score(labels_test, test_pred)
    return (precision, recall, f1)


# In[9]:

warnings.filterwarnings("ignore", category=DeprecationWarning) 
def get_best_decision_tree(data_dict, features_list):
    print "+++++++++++++++++++++++++++++++++++++"
    print "Returning best Grid Search Decision Tree Classifier"
    print 'Features:', features_list
    my_dataset = data_dict
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)    
    ### base test/train split for simple 
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test =         train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

    # Set up cross validator (will be used for tuning all classifiers)
    from sklearn import cross_validation
    cv = cross_validation.StratifiedShuffleSplit(labels_train, test_size=.1, random_state = 42)

    ### Test Decision Trees 
    from sklearn import tree
    parameters = {'max_depth':[2,3,5,8,10,15], 'min_samples_split':[2,3,5], 'criterion' : ['gini','entropy'],
                  'class_weight': [{True: 12, False: 1}, {True: 4, False: 1}, 'auto', None]}
    tempDTClf = tree.DecisionTreeClassifier()
    from sklearn.model_selection import GridSearchCV
    gridClf = GridSearchCV(tempDTClf, parameters, scoring='f1', cv=cv)
    best_dt = gridClf.fit(features_train, labels_train).best_estimator_
    my_k_fold_test_short(best_dt, features, labels)
    return best_dt

features_list_importance = ['poi', 'other', 'expenses', 'total_stock_value', 'exercised_stock_options',
                            'long_term_incentive', 'from_this_person_to_poi', 'from_messages', 'restricted_stock']    
best_dt = get_best_decision_tree(data_dict, features_list_importance)


# In[10]:

dump_classifier_and_data(best_dt, data_dict, features_list_importance)
