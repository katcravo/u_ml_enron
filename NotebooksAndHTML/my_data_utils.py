
import math

def maxPrinter(data_dict, sub_field):
    sub_dict = {key:data_dict[key][sub_field] for key in data_dict if data_dict[key][sub_field] != 'NaN'}
    max_key = max(sub_dict, key=sub_dict.get)
    print "Max", sub_field, max_key, sub_dict[ max_key ]

def max10pctPrinter(data_dict, sub_field):
    sub_dict = {key:data_dict[key][sub_field] for key in data_dict if data_dict[key][sub_field] != 'NaN'}
    #print sub_dict
    #identify the tops
    new_length = int(math.ceil(len(sub_dict)*.9))   
    print "Top 10%", sub_field, ":"
    while len(sub_dict) > new_length:
        max_key = max(sub_dict, key=sub_dict.get)
        print max_key, sub_dict[ max_key ]
        sub_dict.pop(max_key, 0)

def myOutlierCleaner(predictions, features, targets):

    #get indices of the differences
    diff_dict = { i:abs(predictions[i][0]-targets[i][0])
                  for i in range(0,len(features)) }
    
    #remove the top differences
    new_length = int(math.ceil(len(features)*.9))    
    while len(diff_dict) > new_length:
        del diff_dict[ max(diff_dict, key=diff_dict.get) ]
    
    #rebuild the features list 
    cleaned_data = [(features[i],targets[i],diff_dict[i]) for i in diff_dict.keys()]
    return cleaned_data

def removeMax(data_dict, sub_field):
    sub_dict = {key:data_dict[key][sub_field] for key in data_dict if data_dict[key][sub_field] != 'NaN'}
    max_val = max(sub_dict, key=sub_dict.get) 
    data_dict.pop(max_val, 0)
    
    
import matplotlib.pyplot as plt
def plot_two(data_dict, field1, field2):    
    plt.close()
    colors = ['b','r']
    for key in data_dict.keys():
        x = data_dict[key][field1]
        y = data_dict[key][field2]
        plt.scatter( x, y, color=colors[int(data_dict[key]['poi'])] )
    plt.xlabel(field1)
    plt.ylabel(field2)
    plt.show()
    
def Draw(pred, features, poi, mark_poi=False, name="image.png", 
         features_list=["poi", "feature 1", "feature 2"],
         f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """
    plt.close()
    f1_idx = int(features_list.index(f1_name) - 1)
    f2_idx = int(features_list.index(f2_name) - 1)
    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "g", "k", "c", "m"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][f1_idx], features[ii][f2_idx], color = colors[int(pred[ii])])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][f1_idx], features[ii][f2_idx], color="r", marker="+")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    #plt.savefig(name)
    plt.show()