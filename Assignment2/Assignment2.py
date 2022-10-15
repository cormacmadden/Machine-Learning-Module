# %%
%matplotlib widget
import numpy as np
import math
from matplotlib import cm
import pandas as pd
import scipy as sp
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as matlib
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('week3.csv')
df.columns = ["X1","X2","y"]
X1=df.iloc[:,0]
X2=df.iloc[:,1]
y=df.iloc[:,2]
X=np.column_stack((X1,X2))

# %% [markdown]
# a i)

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1,X2,y)
ax.set_xlabel('Feature X1')
ax.set_ylabel('Feature X2')
ax.set_zlabel('Target y')
plt.title('Dataset Scatterplot')
plt.show()

# %% [markdown]
# a) ii)

# %%
Xpoly = PolynomialFeatures(degree = 5).fit_transform(X)

baseline = DummyRegressor(strategy="mean").fit(Xpoly, y)
print("J(θ_baseline) = %f\n"%mean_squared_error(y, baseline.predict(Xpoly)))

C = [1,10,100,1000,10000]
lassos = []
for Ci in C:
    model = Lasso(alpha=1/(2*Ci)).fit(Xpoly,y)
    lassos.append(model)
    print("\nC value: "+str(Ci))
    print("Lasso coef: "+str(model.coef_))
    print("Lasso intercept: "+str(model.intercept_))
    print("J(θ) = %f\n"%mean_squared_error(y, model.predict(Xpoly)))


# %%
### i) c)
Xtest = []
grid = np.linspace(-2,2)
for i in grid:
    for j in grid:
        Xtest.append([i,j])

Xtest = np.array(Xtest)
Xtest = PolynomialFeatures(5).fit_transform(Xtest)

from matplotlib import cm
Xpoly = PolynomialFeatures(5).fit_transform(X)

C_range = [1, 10, 100, 1000, 10000]
for Ci in C_range:
    model = Lasso(alpha=1/(2*Ci))
    model.fit(Xpoly, y)
    y_pred=model.predict(Xtest)
    
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X1, X2, y, c='g', label="Training data")
    surf = ax.plot_trisurf(Xtest[:,1], Xtest[:,2], y_pred, cmap=cm.YlGn, alpha=0.8, linewidth=0, antialiased=True)
    ax.set_title('Lasso prediction surface for C = %.0f'%Ci)
    ax.set(xlabel='X1', ylabel='X2', zlabel='Target Y')
    ax.legend(loc='upper left') 
    
    plt.xlabel("Feature X1"); plt.ylabel("Feature X2")

    plt.show()


# %%
# (i)(e)

baseline = DummyRegressor(strategy="mean").fit(Xpoly, y)
print("J(θ_baseline) = %f\n"%mean_squared_error(y, baseline.predict(Xpoly)))

ridges = []
C_ridge = [0.0001,0.001,0.01,0.1,1,10]
for Ci in C_ridge:
    model = Ridge(alpha=1/(2*Ci)).fit(Xpoly, y)
    theta = np.insert(model.coef_, 0, model.intercept_)
    ridges.append(model)
    
    print("\nC value: "+str(Ci))
    print("θ =", theta)
    print("J(θ) = %f\n"%mean_squared_error(y, model.predict(Xpoly)))
        
    fig = plt.figure(dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:,0], X[:,1], y, c='g', label="Training data")
    surf = ax.plot_trisurf(Xtest[:,1], Xtest[:,2], model.predict(Xtest), cmap=cm.YlGn, alpha=0.8, linewidth=0, antialiased=True)

    ax.set_title('Ridge prediction surface for C = ' + str(Ci))
    ax.set(xlabel='Feature X1', ylabel='Feature X2', zlabel='Target Y')
    ax.legend()

    plt.show()

print("exit")


# %%
### ii a)

kf = KFold(n_splits=5)
mean_error=[]
std_error=[]
Cs2 = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]
for Ci in Cs2:
    model = Lasso(alpha=1/(2*Ci))
    print("\nC: "+str(Ci))
    kf = KFold(n_splits=5)
    temp = []
    for train, test in kf.split(X):
        model.fit(X[train],y[train])
        ypred = model.predict(X[test])
        print("Lasso coeff: "+str(model.coef_))

        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(y[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

plt.figure(dpi=100)
plt.errorbar(Cs2,mean_error,yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('Mean error and standard deviation')
plt.title('Lasso Regression Cross-Validation of C values')
plt.show()

### ii c)
mean_error=[]
std_error=[]
C_ridge2 = [0.0001,0.001,0.01,0.015,0.02,0.025,0.03,0.05,0.1,0.2]
for Ci in C_ridge2:
    model = Ridge(alpha=1/(2*Ci))

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    temp = []
    print("\nC: "+str(Ci))

    for train, test in kf.split(X):
        model.fit(X[train],y[train])
        ypred = model.predict(X[test])
        print("Ridge coeff: "+str(model.coef_))
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(y[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

plt.figure(dpi=100)
plt.errorbar(C_ridge2,mean_error,yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('Mean error and standard deviation')
plt.title('Ridge Regression Cross-Validation of C values')
plt.show()


# %%



