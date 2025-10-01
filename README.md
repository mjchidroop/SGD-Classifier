# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Linear SVM (hinge).
2. Logistic Regression.
3. Modified Huber.
4. Perceptron.

## Program:
```python
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: CHIDROOP M J
RegisterNumber:  25018548
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# Generate a toy binary classification dataset with 2 features
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SGDClassifier as Logistic Regression
clf = SGDClassifier(loss='log_loss', max_iter=1000, alpha=0.01, random_state=42)
clf.fit(X_train, y_train)

# Plot data points
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='coolwarm', edgecolors='k', alpha=0.6, label='Train')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', marker='*', edgecolors='k', label='Test')

# Plot decision boundary
coef = clf.coef_[0]
intercept = clf.intercept_[0]
x_vals = np.linspace(X[:,0].min(), X[:,0].max())
y_vals = -(coef[0]*x_vals + intercept)/coef[1]
plt.plot(x_vals, y_vals, color='black', linewidth=2, label='Decision boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression (SGDClassifier) Decision Boundary')
plt.legend()
plt.show()

```

## Output:
![prediction of iris species using SGD Classifier](sam.png)
<img width="617" height="432" alt="image" src="https://github.com/user-attachments/assets/1e239c2c-7566-43ef-8cfd-c0710f16d749" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
