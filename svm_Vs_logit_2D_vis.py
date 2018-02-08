# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:10:15 2017

@author: Anurag_d
"""

from sklearn import svm
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def frange(start, stop, step):
    myList=[]
    i=start
    while i<stop:
        myList.append(i)
        i=i+step
    return myList
        
#Import data
#xlFile = pd.ExcelFile("data_imbalance.xlsx")
#xlFile = pd.ExcelFile("data.xlsx")
xlFile = pd.ExcelFile("data_3.xlsx")
data=xlFile.parse("Sheet1")

#print(data.ix[0:,2])
X_train = data.ix[:,0:2]
y_train = data.ix[:,2]

#visualise the plot
fig = plt.figure()
plt.ylim(ymin=-2, ymax=2)
plt.xlim(xmin=-6, xmax=15)
#ax = fig.add_subplot(111, projection='3d')
for i in range(0,len(X_train)):
    if y_train.ix[i] == 1:
        plt.plot(X_train.ix[i,0], X_train.ix[i,1], lw=2, color="black", marker="+")
        #ax.scatter(X_train.ix[i,0], X_train.ix[i,1], 1, lw=2, color="black", marker="+")
    else:
        plt.plot(X_train.ix[i,0], X_train.ix[i,1], lw=2, color="black", marker="o")
        #ax.scatter(X_train.ix[i,0], X_train.ix[i,1], 1, lw=2, color="black", marker="o")
#plt.show()

#X_trainRS = X_train.reshape(-1, 1)
clf = svm.LinearSVC(verbose=1, fit_intercept = True)
clfK = svm.SVC(verbose=1, kernel='linear')
clf.fit(X_train, y_train)
clfK.fit(X_train, y_train)

W=clf.coef_[0]
WK = clfK.coef_[0]
intL=clf.intercept_ #we expect this to be 0 if fit_intercept=False.
intK=clfK.intercept_

print("LinearSVC", clf.decision_function(X_train))
print("Weights:", W, " InterceptL:", intL)

print("SVC with linear kernel", clfK.decision_function(X_train))
print("WeightsK:", WK, " InterceptK:", intK)

#Let's plot W
#plt.plot(W[0], int, lw=2, color="red", marker="*")
#plt.plot(WK[0], intK, lw=2, color="blue", marker="1")
#plt.plot(WK[0],WK[1], lw=2, color="blue", marker="1")

#let's draw the weight vector
xp=[0, W[0]]
#yp=[0, int]
yp=[0, W[1]]
#plt.plot(xp, yp, 'r')

#plotting the decision boundary
# W.X + b = 0
#w0x0 + w1x1 + b = 0
#let's vary x1 and record the value of x2.
x1=[]
x2=[]
x1R=[]
x2R=[]

x1R=frange(-4,4,0.1)

if(WK[1] >0):
    for x1Val in x1R:
        x1.append(x1Val)
        x2.append((-intK - WK[0]*x1Val)/WK[1])
    plt.plot(x1, x2, 'r')

x2R=frange(-4,4,0.1)
if(WK[0] > 0):
    for x2Val in x2R:
        x1.append((-intK - WK[1]*x2Val)/WK[0])
        x2.append(x2Val)
    plt.plot(x1, x2, 'r')
    
#plt.show()

#Let's implement the logistic regression on this data.

def sig(x):
  return (1/(1 + math.exp(-x)))
  
def hyp(w, b, x):
  return sig(w[0]*x[0] + w[1]*x[1] + b)

def computeLoss(w, b):
  #y \in (0, 1). Use the NG's formulation: -y_i log (prob of class) - (1-y_i)(log (1 - prob))
  J=0
  for i in range(0,len(X_train)):
    J += (-y_train.ix[i] * (math.log(hyp(w,b,X_train.ix[i]) + err))) - (1 - y_train.ix[i])*(math.log( 1 -  hyp(w,b, X_train.ix[i]) + err) )
  J += (regFactor/2)*(math.pow(w[0],2) + math.pow(w[1], 2)) #regularising	
  return (J/len(X_train))

def computeGradient(w, b, wDim ):
  #hard code to 2 dimensions.
  dw=0
  for i in range(0,len(X_train)):
    if wDim == 0:
      dw += ((hyp(w, b, X_train.ix[i]) - y_train.ix[i])*(X_train.ix[i, 0]) + regFactor*w[0])
    elif wDim == 1:
      dw += ((hyp(w, b, X_train.ix[i]) - y_train.ix[i])*(X_train.ix[i, 1]) + regFactor*w[1])
    else:
      dw += ((hyp(w, b, X_train.ix[i]) - y_train.ix[i])*(1)) #We don't regularise for the bias term,  thus no contribution to it's gradient.
  return (dw/len(X_train))	  

  
#Main
w=[]
w.append(1)
w.append(1) #some initial values of w
b=1
dw0=0
dw1=0
db=0

lr=0.1 #learning rate
regFactor = 1 #regularisation constant
numIter = 1000 #how many iterations of gradient descent to do
err=0#.000001 #to prevent log(0)

for _ in range(0, numIter):
  #print("hyp", hyp(w, b, X_train.ix[0]))
  #print("hyp_2", hyp(w, b, X_train.ix[1]))
  #print("Current Loss:", computeLoss(w, b))
  dw0 = computeGradient(w, b, 0)
  dw1 = computeGradient(w, b, 1)
  db = computeGradient(w, b, 2)
  #print("Computed gradients: dw0=", dw0, " dw1=", dw1, "db=", db)
  w[0] -= lr * dw0
  w[1] -= lr * dw1
  b -= lr * db

print("Final weights:", w, b)
print("Final Loss:",computeLoss(w, b) )  
  
x1=[]
x2=[]
x1R=[]
x2R=[]
x1R=frange(-4,4,0.1)

if(w[1] >0):
    for x1Val in x1R:
        x1.append(x1Val)
        x2.append((-b - w[0]*x1Val)/w[1])
    plt.plot(x1, x2, 'g')

x2R=frange(-4,4,0.1)
if(w[0] > 0):
    for x2Val in x2R:
        x1.append((-b - w[1]*x2Val)/w[0])
        x2.append(x2Val)
    plt.plot(x1, x2, 'g')

plt.show()	

#that's all folks!