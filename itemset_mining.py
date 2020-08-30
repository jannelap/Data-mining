import numpy as np
import csv
'''
Purpose of this program is to find frequent itemsets and association rules from dataset.
Algorithms were tested with market basket dataset from https://www.kaggle.com/irfanasrullah/groceries?select=groceries.csv
'''


def read_file(filename):
    '''
    Algorithm that reads and handles the dataset

    Parameters
    ----------
    filename : Name of the source file

    Returns
    -------
    tracts : List of transactions
    U : List of all the items

    '''
    tracts = list()
    total_trans = set()
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            total_trans = total_trans.union(set(row))
            tracts.append(row)
    U = list(total_trans)
    U.sort()
    i = 0
    while i<len(tracts):
        j = 0
        while j<len(tracts[i]):
            tracts[i][j] = U.index(tracts[i][j])
            j = j+1
        tracts[i] = set(tracts[i])
        i = i+1
    return(tracts, U)
    
def apriori(tracts, U, sigma):
    '''
    Level-wise algorithm that finds frequent itemsets with given minimum support

    Parameters
    ----------
    tracts : List of transactions
    U : List of all the items
    sigma : Minimum support

    Returns
    -------
    final: List of frequent itemsets
    freq: List of supports of the frequent itemsets

    '''
    total=[]
    final=[]
    freq=[]
    K=set()
    for i in range(len(U)):
        K=set()
        K.add(i)
        total.append(K)
    p_level=[]
    #Finding frequent singletons
    for i in total:
        count=0
        for j in range(len(tracts)):
            if i.issubset(tracts[j]):
                count=count+1
        if (count/len(tracts))>=sigma:
            final.append(i)
            freq.append(round(count/len(tracts),3))
            p_level.append(i)
    #Finding rest of the frequent itemsets
    while len(p_level)>0:
        #Generating candidates
        candidates=[]
        for i in range(len(p_level)):
            j=0
            while j<len(p_level):
                for r in p_level[j]:
                    e=set()
                    e.add(r)
                    if len(p_level[i].union(e))==(len(p_level[i])+1):
                        if candidates.count(p_level[i].union(e))==0:
                            candidates.append(p_level[i].union(e))
                j=j+1
        p_level=[]
        #Calculating support
        for i in candidates:
            supp_i = support(tracts, U, i)
            if supp_i>sigma:
                final.append(i)
                freq.append(round(supp_i,3))
                p_level.append(i)
    return(final, freq)

def support(tracts, U, item):
    '''
    Calculates support of itemset

    Parameters
    ----------
    tracts : List of transactions
    U : List of all the items
    item : Itemset whose support needs to be found

    Returns
    -------
    support: Support of given itemset

    '''
    count = 0
    i = 0
    while i < len(tracts):
        if len(tracts[i].union(item)) == len(tracts[i]):
            count = count+1
        i = i+1
    support = count/len(tracts)
    return(support)
    
def association_rule(tracts, U, sigma, conf):
    '''
    Finds association rules with given minimum support and confidence

    Parameters
    ----------
    tracts : List of transactions
    U : List of all the items
    sigma : Minimum support for the rule
    conf : Minimum confidence

    Returns
    -------
    rules: List of association rules
    conf_list: List of confidences of the rules
    sup_list: List of supports of the rules

    '''
    #Generating list of possible rules
    frequents, freq = apriori(tracts, U, sigma)
    frequents1 = frequents.copy()
    list_x=[]
    list_y=[]
    i = 0
    while i < len(frequents):
        j = 0
        while j < len(frequents):
            if j != i:
                if frequents[i].union(frequents[j]) == frequents[j] or len(frequents[i])==1:
                    frequents1.remove(frequents[i])
                    break
            j = j+1 
        i = i+1
    i = 0
    while i < len(frequents1):
        for j in frequents1[i]:
            temp_set=frequents1[i].copy()
            temp_set.remove(j)
            list_x.append(temp_set)
            list_y.append(set({j}))
            if len(frequents1[i])>2:
                list_y.append(temp_set)
                list_x.append(set({j}))
        i = i+1
    i = 0
    true_list_x = []
    true_list_y = []
    conf_list = []
    sup_list = []
    #Calculating confidences of the rules
    while i < len(list_x):
        support_xy = support(tracts, U, list_x[i].union(list_y[i]))
        real_conf = support_xy/support(tracts, U, list_x[i])
        if real_conf>conf:
            true_list_x.append(list_x[i])
            true_list_y.append(list_y[i])
            conf_list.append(round(real_conf, 3))
            sup_list.append(round(support_xy, 3))
        i = i+1
    rules = np.column_stack((true_list_x, true_list_y))
    return(rules, conf_list, sup_list)

    
    
    
if __name__ == "__main__":
    filename = 'groceries.csv'
    tracts, U = read_file(filename)
    sigma = 0.03
    confidence = 0.40
    
    #Mine and print all frequent itemsets with given support (sigma) 
    items, freq = apriori(tracts, U, sigma)
    total = np.column_stack((items, freq))
    print(total)
    
    #Mine and print all association rules with given support and confidence
    rules, confs, supp = association_rule(tracts, U, sigma, confidence)
    list1 = np.column_stack((rules, supp))
    list2 = np.column_stack((list1, confs))
    print(list2)
    
    
