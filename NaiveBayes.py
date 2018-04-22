from __future__ import division
import os
import re
import math

clean_pat = re.compile(r'\w{2,}')
pos_path = os.getcwd() + "\pos"
pos_test_path = os.getcwd() + "\\test_pos"
neg_test_path = os.getcwd() + "\\test_neg"
neg_path = os.getcwd() + "\\neg"
pos_dict = {}
neg_dict = {}
count_pos = 0
count_neg = 0

def getWordCount(dir):
    temp_count = {}
    count = 0
    for filename in os.listdir(dir):
        with open(dir+ "\\" +filename,'r') as files:
            count += 1
            for i in files.readlines():
                for word in re.findall(clean_pat,i):
                    if not word in temp_count:
                        temp_count[word] = 1
                    else:
                        temp_count[word]+= 1
    return temp_count,count

def NB(doc):
    tp = 0
    fp = 0
    for filename in os.listdir(doc):
        with open(doc+ "\\" +filename, 'r') as files:
            tempPos = P_pos
            tempNeg = P_neg
            for i in files.readlines():
                for word in re.findall(clean_pat,i):
                    tempCPos = (pos_dict.get(word, 1) + 1)/(count_pos + vocab)
                    tempPos += math.log(tempCPos)
                    tempCNeg = (neg_dict.get(word, 1) + 1)/(count_neg + vocab)
                    tempNeg += math.log(tempCNeg)
            if tempNeg > tempPos:
                fp += 1
            else:
                tp += 1
    return tp, fp

pos_dict, total_pos_docs = getWordCount(pos_path)
neg_dict, total_neg_docs = getWordCount(neg_path)

for i in pos_dict.keys():
    count_pos += pos_dict[i]

for i in neg_dict.keys():
    count_neg += neg_dict[i]

total_docs = total_pos_docs + total_neg_docs
P_pos = total_pos_docs / total_docs
P_neg = total_neg_docs / total_docs

vocab = len(set(pos_dict.keys() + neg_dict.keys()))
tp, fp = NB(pos_test_path)
fn, tn = NB(neg_test_path)

precision_pos = tp/ (tp + fp)
precision_neg = tn/ (tn + fn)
recall_pos = tp/ (tp + fn)
recall_neg = tp/ (tp + fn)
F1_pos = (2*precision_pos*recall_pos)/(precision_pos+recall_pos)
F1_neg = (2*precision_neg*recall_neg)/(precision_neg+recall_neg)

print ("PrecisionPos:" ,precision_pos)
print ("PrecisionNeg:",precision_neg)
print ("RecallPos:",recall_pos)
print ("RecallNeg:",recall_neg)
print ("F1Pos:",F1_pos)
print ("F1Neg:",F1_neg)

