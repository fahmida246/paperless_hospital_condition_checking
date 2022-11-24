# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 17:44:29 2022

@author: nus34
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
#%%
df = pd.read_csv('dhaka.csv')
print(df)
print(df.head())
#%%

print(df.shape)

#%%
# is there any missing value?
print(df.isna().sum())
#%%
X = df.drop(columns=['Name','class'], axis=1)
y = df['class']
print('shape of X and y respectively :', X.shape, y.shape)
#%%
#shape of x and y for train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('shape of X and y for training :', X_train.shape, y_train.shape)
print('shape of X and y for testing :', X_test.shape, y_test.shape)


#%% 3
#************logistic regresssion*************************
print('Logistic Regression')
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
Y_pred = model1.predict(X_test)

score1 = model1.score(X_train, y_train)
print('Training accuracy:', score1)

score = model1.score(X_test, y_test)
print('Testing accuracy:', score)


a1=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",a1)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for logistic regression')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()




#%% 1
#************decision tree classifier *************************
print('DecisionTreeClassifier')
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(max_depth=5) 
model2.fit(X_train, y_train)  
Y_pred = model2.predict(X_test)

score1 = model2.score(X_train, y_train)
print('Training accuracy:', score1)

score = model2.score(X_test, y_test)
print('Testing accuracy:', score)

a2=metrics.accuracy_score(y_test, Y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Blues', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for decision tree')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


#%% 2
#************random forest tree classifier *************************
print('RandomForestClassifier')
from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=100) 
model3.fit(X_train, y_train)
Y_pred = model3.predict(X_test)
score1 = model3.score(X_train, y_train)

print('Training accuracy:', score1)
score = model3.score(X_test, y_test)

print('Testing accuracy:', score)

a3=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Reds', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for random forest')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#%% 4
#************kneighbours classifier *************************
print('KNeighborsClassifier')
from sklearn.neighbors import KNeighborsClassifier
model4 = KNeighborsClassifier()
model4.fit(X_train, y_train)
Y_pred = model4.predict(X_test)
score1 = model4.score(X_train, y_train)

print('Training accuracy:', score1)
score = model4.score(X_test, y_test)

print('Testing accuracy:', score)
a4=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for knc')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#%%  5
#*********************naive bayes*************
print('naive bayes')
from sklearn.naive_bayes import GaussianNB
model5 = GaussianNB()
model5.fit(X_train, y_train)
Y_pred = model5.predict(X_test)
score1 = model5.score(X_train, y_train)

print('Training accuracy:', score1)
score = model5.score(X_test, y_test)

print('Testing accuracy:', score)
a5=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Blues', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for navie bayes')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


#%%
#support vector classifier using 3 different tecniques linear,rbf and poly

from sklearn.svm import SVC
print('using linear kernel')
model6 = SVC( kernel='linear')
model6.fit(X_train, y_train)
Y_pred = model6.predict(X_test)
score = model6.score(X_train, y_train)

print('Training accuracy:', score)
score = model6.score(X_test, y_test)

print('Testing accuracy:', score)
a6=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Reds', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for svm linear')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
#%%
#************************
print('using poly kernel')
model7 = SVC( kernel='poly')
model7.fit(X_train, y_train)
Y_pred = model7.predict(X_test)
score = model7.score(X_train, y_train)

print('Training accuracy:', score)
score = model7.score(X_test, y_test)

print('Testing accuracy:', score)
a7=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for svm poly')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show() 
#%%
#************************
print('using rbf kernel')
model8 = SVC( kernel='rbf')
model8.fit(X_train, y_train)
Y_pred = model8.predict(X_test)

score = model8.score(X_train, y_train)
print('Training accuracy:', score)

score = model8.score(X_test, y_test)
print('Testing accuracy:', score)

a8=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Blues', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for svm rbf')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
#%%
#************************linear discriminante
print('using linear discriminante')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model9 = LinearDiscriminantAnalysis()
model9.fit(X_train, y_train)
Y_pred = model9.predict(X_test)
score = model9.score(X_train, y_train)

print('Training accuracy:', score)
score = model9.score(X_test, y_test)

print('Testing accuracy:', score)

a9=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Reds', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for linear discriminante analysis')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


#%% 8
#using adaboost
print('adaboost')
from sklearn.ensemble import AdaBoostClassifier
model10 = AdaBoostClassifier()
model10.fit(X_train, y_train)
Y_pred = model10.predict(X_test)
score = model10.score(X_train, y_train)

print('Training accuracy:', score)
score = model10.score(X_test, y_test)

print('Testing accuracy:', score)


a10=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for adaboost')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


#%%
#using xgboost
from xgboost import XGBClassifier
model11 = XGBClassifier(n_jobs=-1,n_estimators=12, random_state=4)
model11.fit(X_train, y_train)
Y_pred = model11.predict(X_test)
score = model11.score(X_train, y_train)

print('Training accuracy:', score)
score = model11.score(X_test, y_test)

print('Testing accuracy:', score)


a11=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for xgboost')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()





#%%
results=pd.DataFrame(columns=['accuracy'])
results.loc['Logistic Regression']=[a1]
results.loc['Decision Tree Classifier']=[a2]
results.loc['Random Forest Classifier']=[a3]
results.loc['k nearest neighours']=[a4]
results.loc['naive bayes']=[a5]
results.loc['(svm)linear']=[a6]
results.loc['(svm)poly']=[a7]
results.loc['(svm)rbf']=[a8]
results.loc['Linear discriminant analysis']=[a9]
results.loc['adaboost']=[a10]
results.loc['xgboost']=[a11]
print('accuracy list')
b=results.sort_values('accuracy',ascending=False)
print(b)
#%%
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
classifiers = ['Logistic', 'DT','RF','KNN','naive ','ld','poly','rbf','disc','ada','xgb']
accuracies = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]
ax.bar(classifiers,accuracies,align='center', width=0.4,color=['red','green','yellow','blue','purple'])
plt.ylim()
plt.show()

#%%
class_name = ('Logistic', 'DT','RF','KNN','naive ','ld','poly','rbf','disc','ada','xgb')
class_score = (a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11)
y_pos= np.arange(len(class_score))
colors = ("red","gray","purple","green","orange","blue")
plt.figure(figsize=(20,12))
plt.bar(y_pos,class_score,color=colors)
plt.xticks(y_pos,class_name,fontsize=20)
plt.yticks(np.arange(0.00, 1.05, step=0.2))
plt.ylabel('Accuracy')
plt.grid()
plt.title(" Accuracy Comparision of the Classes",fontsize=15)
plt.show()


#%%
#correlation matrix
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize = (15, 10))

ax = sns.heatmap(corr_matrix, 
                annot=True,
                linewidths= 0.5,
                annot_kws={"size":30},
                fmt="0.2f",
                cmap="coolwarm");


#%%
