#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:20:32 2017

@author: stephanosarampatzes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
data=pd.read_csv('auto-mpg-nameless.csv')

# number of cars per origin
data.origin.value_counts()

# percentages per origin
for i in range(1,4):
    x=round((len(data.loc[(data['origin']==i)])/len(data.origin)*100),2)
    print('origin',i,'is',x,'% of total number of cars')
    
# discrete numerical plots
f, ax = plt.subplots(figsize=(10, 7))
plt.subplot(2,1,1)
sns.countplot(x="cyls",hue="origin", data=data);
plt.title("Cylinder count per origin")
plt.subplot(2,1,2)
sns.countplot(x='year',hue="origin",data=data);
plt.title("Year count per origin")
plt.tight_layout()
plt.show()

# stripplots
plot=sns.stripplot(x='weight',y='mpg',hue='origin',data=data,size=5.0)
plot.set(xticklabels=[])
plt.xticks(rotation=90)
plt.show()
plot=sns.stripplot(x='displacement',y='mpg',hue='origin',data=data)
plt.xticks(rotation=90,fontsize=7)
plt.show()
plot=sns.stripplot(x='cyls',y='mpg',hue='origin',data=data,jitter=True)
plt.show()
sns.stripplot(x='displacement',y='hp',hue='origin',data=data)
plt.xticks(rotation=90,fontsize=6.5)
plt.show()
sns.stripplot(x='year',y='acc',hue='origin',data=data, jitter =True)
plt.show()    


# import packages as usual

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from sklearn.neighbors import KNeighborsClassifier 

# train & test set
X=data.iloc[:,0:7]
y=data.iloc[:,7]

data['cc']=np.array(data.displacement)*16.38706
X['cc']=data.cc

# cross validation
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.25 ,random_state=150)

RF=RandomForestClassifier(n_estimators=100,random_state=150)
RF.fit(X_train,y_train)
scores = cross_val_score(RF, X, y, cv=10, scoring='accuracy')
y_pred=RF.predict(X_test)
accuracy_RF=accuracy_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred)

print(scores.mean())
print(accuracy_RF)
print(conf_matrix)
print(classification_report(y_test,y_pred))


# feature importances

importances = RF.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances,color="r",align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout
plt.xticks(rotation=45)
plt.show()

# confusion matrix analysis
print(conf_matrix)
print()
for i in range(1,3):
    if i==1:
        print('| TP','FN |')
    else:
        print('| FP','TN |')
        
# useful groups with sorted values for feature engineering
group1=data.sort_values(['hp','origin'], ascending=[False,False])
group2=data.sort_values(['displacement','origin'], ascending=[False,False])
group3=data.sort_values(['cyls','origin'], ascending=[False,False])
group4=data.sort_values(['acc','origin'], ascending=[False,False])
group5=data.sort_values(['year','origin'], ascending=[False,False])
group6=data.sort_values(['weight','origin'], ascending=[False,False]) 

group_max = data[['displacement','origin']].groupby(['origin']).max()
print(group_max)

# 20 new features
### We can change data with X to append directly features into train set !!!
# us cars new features
data['us1']=np.where(((data['cyls']==4)&((data['displacement']==135)|(data['displacement']==140))),1,0)
data['us2']=np.where(((data['acc']>16.7)&(data['acc']<19.6)&(data['cyls']==6)),1,0)
data['us3']=np.where(((data['displacement']>183)&(data['cyls']==8)),1,0)
data['us4']=np.where(((data['cyls']==8)&((data['year']==70)|(data['year']==73)|(data['year']==77)|(data['year']==79))),1,0)
data['us5']=np.where(((data['acc']<12.5)&(data['year']<76)),1,0)

# eu cars new features
data['eu1']=np.where(((data['displacement']>=114)&(data['displacement']<=121)&(data['mpg']<23)&(data['cyls']==4)),1,0)
data['eu2']=np.where((((data['hp']<50)|(data['hp']>102))&(data['cyls']==4)),1,0)
data['eu3']=np.where((((data['acc']>=22.5)|(data['acc']==15.5)|(data['acc']==14)|((data['acc']>=21.5)&(data['acc']<=21.9)))&(data['cyls']==4)),1,0)
data['eu4']=np.where(((data['weight']>3100)&((data['cyls']==4)|(data['cyls']==5))),1,0)
data['eu5']=np.where(((data['hp']==71)&(data['cyls']==4)),1,0)
data['eu6']=np.where((((data['cyls']==4)&(data['weight']<=2190)&(data['weight']>=2188))),1,0)
data['eu7']=np.where(((data['mpg']<30)&(data['weight']<1955)),1,0)

# japanese cars new features
data['jp1']=np.where((((data['displacement']>=70)&(data['displacement']<=78)&(data['hp']>50))|(data['cyls']==3)),1,0)
data['jp2']=np.where(((((data['displacement']>=81)&(data['displacement']<=85))|(data['displacement']==107)|(data['displacement']==108))&(data['mpg']>30)),1,0)
data['jp3']=np.where(((data['hp']==67)&((data['displacement']==91)|(data['displacement']==97))),1,0)
data['jp4']=np.where(((data['hp']==75)&(data['displacement']>=97)&(data['displacement']<116)),1,0)
data['jp5']=np.where(((data['hp']==88)&(data['displacement']==97)),1,0)
data['jp6']=np.where(((data['hp']==97)&(data['mpg']>18)),1,0)
data['jp7']=np.where(((data['hp']>=92)&(data['hp']<=94)&(data['displacement']<140)),1,0)
data['jp8']=np.where(((data['cyls']==4)&(data['hp']<66)&(data['weight']<=1800)),1,0)  

# always check our features

def check_us(column1):
    for i in data.index:
        if data.origin[i] == data[column1][i]:
            print('us',data.index[i])
        elif data.origin[i] != 1 & data[column1][i]==1:
            print('not us',data.index[i])
    return(column1)

for i in ('us1','us2','us3','us4','us5'):
    print(i)
    print()
    check_us(i)
    print()
    
def check_eu(column2):
    for i in data.index:
        if data.origin[i]==2 and data[column2][i]==1:
            print('eu',data.index[i])
        elif data.origin[i] != 2 & data[column2][i]==1:
            print('not eu',data.index[i])
    return(column2)

for i in ('eu1','eu2','eu3','eu4','eu5','eu6','eu7'):
    print(i)
    print()
    check_eu(i)
    print()
  
def check_jp(column3):
    for i in data.index:
        if data.origin[i]==3 and data[column3][i]==1:
            print('jp',data.index[i])
        elif data.origin[i] != 3 & data[column3][i]==1:
            print('not jp',data.index[i])

for i in ('jp1','jp2','jp3','jp4','jp5','jp6','jp7','jp8'):
    print(i)
    print()
    check_jp(i)     