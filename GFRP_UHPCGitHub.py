#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , Normalizer 
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
import sklearn.decomposition as dec
from sklearn.linear_model import SGDRegressor , Ridge , LinearRegression , Lasso , LassoLars ,RANSACRegressor, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor,XGBClassifier
from sklearn.ensemble import AdaBoostRegressor , RandomForestRegressor , GradientBoostingRegressor , ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from matplotlib.cm import get_cmap
from sklearn.metrics import mean_squared_error    
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.font_manager as font_manager
import random
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from catboost import CatBoostRegressor


# In[17]:


# Splitting the dataset
df = pd.read_excel(r"D:\Articles\GFRP_UHPC\Dataset.xlsx", sheet_name='Dataset' ,header = 0 )
y = df.iloc[:, 8].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0,1,2,3,4,5,6,7]].to_numpy()
Xtr , Xte , ytr , yte = train_test_split(X,y, train_size = 0.7 ,random_state=42 )
df.head(5)


# In[10]:


Xtr , Xte , ytr , yte = train_test_split(X, y, train_size=0.7 ,random_state=42)
model=AdaBoostRegressor(random_state=0)
model.fit(Xtr, ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)
r2tr = round(r2_score(ytr , yprtr), 2)
r2te = round(r2_score(yte , yprte), 2)
msetr = round(mean_squared_error(ytr , yprtr)**0.5, 2)
msete = round(mean_squared_error(yte , yprte)**0.5, 2)
maetr = round(mean_absolute_error(ytr , yprtr), 2)
maete = round(mean_absolute_error(yte , yprte), 2)
a = 0
b = 55

# 'learning_rate': 0.5, 'loss': 'exponential', 'n_estimators': 100
Xtr1 , Xte1 , ytr1 , yte1 = train_test_split(X, y, train_size=0.7, random_state=42)
model1=AdaBoostRegressor(random_state=0, n_estimators=100, learning_rate=0.5, loss='exponential')
model1.fit(Xtr1 , ytr1)
yprtr1 = model1.predict(Xtr1)
yprte1 = model1.predict(Xte1)
r2tr1 = round(r2_score(ytr1 , yprtr1), 2)
r2te1 = round(r2_score(yte1 , yprte1), 2)
msetr1 = round(mean_squared_error(ytr1 , yprtr1)**0.5, 2)
msete1 = round(mean_squared_error(yte1 , yprte1)**0.5, 2)
maetr1 = round(mean_absolute_error(ytr1 , yprtr1), 2)
maete1 = round(mean_absolute_error(yte1 , yprte1), 2)
a1 = 0
b1 = 55

# Plotting the figures
plt.figure(figsize=(12, 6))
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)
plt.subplot(1, 2, 1)
plt.scatter(ytr, yprtr, s=100,marker='*', facecolors='cornflowerblue', edgecolors='black',
            label=f'\n Train \n R² = {r2tr}  \nRMSE = {msetr}\nMAE = {maetr}')
plt.scatter(yte , yprte, s=100, marker='*',facecolors='pink', edgecolors='black',
            label=f'\n Test \n R² = {r2te} \nRMSE = {msete}\nMAE = {maete}')
plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title(f'ADAboost Before Hyperparameter tuning', fontsize=14)
plt.xlabel('Bond Strength (MPa)_Experimental', fontsize=15)
plt.ylabel('Bond Strength (MPa)_Predicted', fontsize=15)
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)
plt.legend(loc=4)
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.scatter(ytr1, yprtr1, s=100,marker='*', facecolors='slateblue', edgecolors='black',
            label=f'\n Train \n R² = {r2tr1}  \nRMSE = {msetr1}\nMAE = {maetr1}')
plt.scatter(yte1 , yprte1, s=100, marker='*',facecolors='deeppink', edgecolors='black',
            label=f'\n Test \n R² = {r2te1} \nRMSE = {msete1}\nMAE = {maete1}')
plt.plot([a1, b1], [a1, b1], c='black', lw=1.4, label='y = x')
plt.title(f'ADAboost After Hyperparameter tuning', fontsize=14)
plt.xlabel('Bond Strength (MPa)_Experimental', fontsize=15)
plt.ylabel('Bond Strength (MPa)_Predicted', fontsize=15)
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)
plt.legend(loc=4)
plt.tight_layout()


plt.savefig(r"D:\Articles\GFRP_UHPC\Figs\ADA.png", format='png', dpi=2000, bbox_inches='tight')
plt.show()


# In[21]:


import matplotlib.pyplot as plt
import numpy as np

# Calculate metrics for each model
metrics = {
    'ADA': {
        'R2': round(r2_score(yte, pred1), 2),
        'RMSE': round(mean_squared_error(yte, pred1) ** 0.5, 2),
        'MAE': round(mean_absolute_error(yte, pred1), 2)
    },
    'CAT': {
        'R2': round(r2_score(yte, pred2), 2),
        'RMSE': round(mean_squared_error(yte, pred2) ** 0.5, 2),
        'MAE': round(mean_absolute_error(yte, pred2), 2)
    },
    'GB': {
        'R2': round(r2_score(yte, pred3), 2),
        'RMSE': round(mean_squared_error(yte, pred3) ** 0.5, 2),
        'MAE': round(mean_absolute_error(yte, pred3), 2)
    },
    'XGB': {
        'R2': round(r2_score(yte, pred4), 2),
        'RMSE': round(mean_squared_error(yte, pred4) ** 0.5, 2),
        'MAE': round(mean_absolute_error(yte, pred4), 2)
    },
    'HGB': {
        'R2': round(r2_score(yte, pred5), 2),
        'RMSE': round(mean_squared_error(yte, pred5) ** 0.5, 2),
        'MAE': round(mean_absolute_error(yte, pred5), 2)
    },
    'VotingRegressor': {
        'R2': round(r2_score(yte, pred6), 2),
        'RMSE': round(mean_squared_error(yte, pred6) ** 0.5, 2),
        'MAE': round(mean_absolute_error(yte, pred6), 2)
    }
}

# Prepare data for plotting
models = list(metrics.keys())
r2_values = [metrics[model]['R2'] for model in models]
rmse_values = [metrics[model]['RMSE'] for model in models]
mae_values = [metrics[model]['MAE'] for model in models]

# Create a figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Histogram for R2
axes[0].bar(models, r2_values, color='skyblue')
axes[0].set_title('R² Values for Each Model')
axes[0].set_ylabel('R² Value')
axes[0].set_ylim(0, 1)  # R² ranges from 0 to 1

# Histogram for RMSE
axes[1].bar(models, rmse_values, color='lightgreen')
axes[1].set_title('RMSE Values for Each Model')
axes[1].set_ylabel('RMSE Value')

# Histogram for MAE
axes[2].bar(models, mae_values, color='salmon')
axes[2].set_title('MAE Values for Each Model')
axes[2].set_ylabel('MAE Value')

# Adjust layout to avoid overlap
plt.tight_layout()

# Save or display the figures
# plt.savefig(r"D:\Articles\GFRP_UHPC\Figs\model_metrics_histograms.png", format='png', dpi=2000, bbox_inches='tight')
plt.show()

