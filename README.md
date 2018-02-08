# SVM_vs_LogisticRegression
A python program to depict the decision boundaries (in 2D) for the SVM (linear kernel) and the logistic regression.

SVM decision boundary is obtained by the scikit-learn SVC API which internally uses the LIBSVM library. The kernel used is linear.

Logistic regression has been implemented from scratch. It uses the cross entropy loss and L2 regularisation. The optimisation is done by gradient descent. 

The hyper-paramters have been chosen such that the decision boundary of the balanced dataset (data_3.xlsx) for both are the same.

--------------------------
How to run:

Run the python script with data from 'Data_3.xlsx'. This learns the decision boundary for 2 data points and plots the decision boundaries for both. Since the decision boundaries are exactly the same, they overlap.

Change the data file to 'Data_imbalance.xlsx'. This data has a largely skewed data towards the positive class. However, the decision boundary for the SVM does not change (since the support vectors do not change), but the decision boundary of logistic regression shifts away from the positive class (which has more datapoints). The decision boundary from logistic regression is no longer a max-margin. Thus, we cannot consider the logistic regression as a max margin classifier.

Play around with the logistic regression hyper-paramters, such as regularisation constant, to see some fun decision boundaries that are learnt.

Comments, bugs - report to anurag.bms at gmail.com