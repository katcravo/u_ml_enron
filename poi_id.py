
# coding: utf-8

# In[1]:

#!/usr/bin/python

#get_ipython().magic(u'matplotlib notebook')
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import my_classifier_utils
import my_data_utils

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus', 'total_payments','exercised_stock_options','shared_receipt_with_poi','expenses',
                'email_to_poi_ratio', 'email_from_poi_ratio', 'exer_stock_ratio'] 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[2]:

### Task 2: Remove outliers
import my_data_utils


# In[3]:

### TOTAL entry existed in data set.  This must be removed.
my_data_utils.maxPrinter(data_dict, 'salary')
print 'remove max'
my_data_utils.removeMax(data_dict, 'salary')
my_data_utils.maxPrinter(data_dict, 'salary')
### Further exploration of the data didn't warrant removal of any other entries

# In[4]:

### Task 3: Create new feature(s)

### Build ratios for poi emails and exercised stock options
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


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# In[6]:

### Split the data and a print summary

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)
print "Train:"
my_classifier_utils.count_true(labels_train)
my_classifier_utils.count_false(labels_train)
print "Test:"
my_classifier_utils.count_true(labels_test)
my_classifier_utils.count_false(labels_test)


# In[7]:

### Try a default decision tree

from sklearn import tree
dtClf = tree.DecisionTreeClassifier()
dt_pred = my_classifier_utils.trainAndTestClassifier (dtClf, features_train, features_test, labels_train, labels_test)
print dt_pred


# In[8]:

print 'DT Importance:'
for i in range (0,len(features_list)-1):
    print features_list[i+1], ":", dtClf.feature_importances_[i]


# In[9]:

### Try a simple decision tree in adaboost

from sklearn.ensemble import AdaBoostClassifier
abcDT = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME")
abcDT_pred = my_classifier_utils.trainAndTestClassifier (abcDT, features_train, features_test, labels_train, labels_test)
print abcDT_pred


# In[10]:

print 'AB Importance:'
for i in range (0,len(features_list)-1):
    print features_list[i+1], ":", abcDT.feature_importances_[i]


# In[11]:


### Try decision tree with different parameters using grid search

parameters = {'max_depth':[2,3,5,8,10,15], 'min_samples_split':[2,3,5], 'criterion' : ['gini','entropy']}
tempDTClf = tree.DecisionTreeClassifier()
from sklearn.model_selection import GridSearchCV
gridClf = GridSearchCV(tempDTClf, parameters, scoring='f1')
grid_pred = my_classifier_utils.trainAndTestClassifier (gridClf, features_train, features_test, labels_train, labels_test)
print grid_pred
print gridClf.best_params_


# In[12]:

print 'Grid DT Importance:'
for i in range (0,len(features_list)-1):
    print features_list[i+1], ":", gridClf.best_estimator_.feature_importances_[i]


# In[13]:


### Fit and test the classifier with other folds of training and test data
my_classifier_utils.my_k_fold_test(gridClf, features, labels)


# In[14]:


### Save the details for tester
dump_classifier_and_data(gridClf, my_dataset, features_list)


# In[ ]:



