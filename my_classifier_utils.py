from sklearn import metrics
import sys
from time import time
from time import sleep


def classifierAccuracy(classifier, features_test, labels_test):
    """ compute the accuracy of your classifier """
    t0 = time()
    pred = classifier.predict(features_test)
    print "predict time:", round(time()-t0, 3), "s"
    
    accuracy = metrics.accuracy_score(labels_test, pred, normalize=True, sample_weight=None)
    print "accuracy:", accuracy

    return pred

def count_true(binaryList):
    trueCount = sum(1 for item in binaryList if item==1.)
    print "True Count:" + str(trueCount)
    return trueCount
def count_false(binaryList):
    falseCount = sum(1 for item in binaryList if item==0.)
    print "False Count:" + str(falseCount)
    return falseCount
def wrong_predictions(pred, labels):    
    ids = [(i,labels[i]) for i in range(0,len(pred)) if pred[i]!=labels[i]]
    print str(len(ids)), "Wrong ones:", ids
    return ids
def true_positives(pred, labels):
    ids = [i for i in range(0,len(pred)) if pred[i]==1. and labels[i]==1.]
    print str(len(ids)), "True Positives:", ids
    return ids
def true_negatives(pred, labels):
    ids = [i for i in range(0,len(pred)) if pred[i]==0. and labels[i]==0.]
    print str(len(ids)), "True Negatives:", ids
    return ids
def false_positives(pred, labels):
    ids = [i for i in range(0,len(pred)) if pred[i]==1. and labels[i]==0.]
    print str(len(ids)), "False Positives:", ids
    return ids
def false_negatives(pred, labels):
    ids = [i for i in range(0,len(pred)) if pred[i]==0. and labels[i]==1.]
    print str(len(ids)), "False Negatives:", ids
    return ids
    
def audit_classifier_results(pred, labels):
    wrong_predictions(pred, labels)
    true_positives(pred, labels)
    true_negatives(pred, labels)
    false_positives(pred, labels)
    false_negatives(pred, labels)
    print "Precision:", metrics.precision_score(labels, pred)
    print "Recall:", metrics.recall_score(labels, pred)    

def trainAndTestClassifier(classifier, features_train, 
                    features_test, labels_train, labels_test):
    print len(labels_train), "Training Points"

    t0 = time()
    classifier.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"
    
    train_pred = classifierAccuracy(classifier, features_train, labels_train)
    print "Train Precision:", metrics.precision_score(labels_train, train_pred)
    print "Train Recall:", metrics.recall_score(labels_train, train_pred)   
    
    print ''    
    print len(labels_test), "Test Points"
    pred = classifierAccuracy(classifier, features_test, labels_test)
    
    audit_classifier_results(pred, labels_test)
    print ''
    
    return pred
    
from sklearn.preprocessing import MinMaxScaler
def trainAndTestWithScale(classifier, features_train, 
                    features_test, labels_train, labels_test):
    scaler = MinMaxScaler()
    rescaled_features_train = scaler.fit_transform(features_train)
    rescaled_features_test = scaler.transform(features_test)
    return trainAndTestClassifier (classifier, rescaled_features_train, rescaled_features_test, labels_train, labels_test)
    

def my_k_fold_test_abc(classifier, features, labels):
    from sklearn.model_selection import KFold
    k_fold = KFold(3)

    for k, (train, test) in enumerate(k_fold.split(features, labels)):
        print "K ", k, "ABC"
        from sklearn.ensemble import AdaBoostClassifier
        k_abc = AdaBoostClassifier(classifier,
                                 algorithm="SAMME")
        k_abc_pred = trainAndTestClassifier (
            k_abc, 
            [features[ii] for ii in train], [features[ii] for ii in test],
            [labels[ii] for ii in train], [labels[ii] for ii in test]) 
        print k_abc_pred  
        

def my_k_fold_test (classifier, features, labels):
    from sklearn.model_selection import KFold
    k_fold = KFold(3)

    for k, (train, test) in enumerate(k_fold.split(features, labels)):
        print "K ", k, "ABC"
        k_pred = trainAndTestClassifier (
            classifier,
            [features[ii] for ii in train], [features[ii] for ii in test],
            [labels[ii] for ii in train], [labels[ii] for ii in test]) 
        print k_pred  