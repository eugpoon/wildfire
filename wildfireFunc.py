# %matplotlib inline
import os
import os.path
import random
import warnings
import zipfile
from itertools import chain, product

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from IPython.display import display
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, mapping
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

random.seed(100)
# ## Split
S = 100
W = 'balanced'  # deals with class imbalance

# clip using cropped sierra nevada boundary
def clip(f, res, bounds, boundaries, x_dim='lon', y_dim='lat'):
    # include 1 more pixel on all sides
    minLon, maxLon = bounds.iloc[0].minx - res, bounds.iloc[0].maxx + res
    maxLat, minLat = bounds.iloc[0].maxy + res, bounds.iloc[0].miny - res
    d = xr.open_dataset(f)
    var = list(d.keys())[0]
    d = d[var].sel(lon=slice(minLon, maxLon), lat=slice(maxLat, minLat))
    d.rio.write_crs('epsg:4326', inplace=True)
    d.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
    d = d.rio.clip(boundaries.geometry.apply(mapping),
                   boundaries.crs, drop=False)
    return d


def getNC(path, bounds, boundaries, years):
    res = (2/24)  # add 2 pixels to all sides
    vars = ['bi_', 'rmin_', 'tmmx_', 'vs_']
    files = [i[0] + i[1] + '.nc' for i in list(product(vars, years))]
    ds = []
    for f in files:
        f = 'https://www.northwestknowledge.net/metdata/data/'+f+'#mode=bytes'
        ds.append(clip(f, res, bounds, boundaries).drop('crs'))
    ds = xr.merge(ds)  # merge all gridmet vars
    ds.to_netcdf(path)
    z = zipfile.ZipFile(path+'.zip', 'w', zipfile.ZIP_DEFLATED)
    z.write(path, os.path.basename(path))
    z.close()
    # !rm data/gm.nc


def getSquare(lon, lat, res):
    return Polygon([(lon+res/2, lat+res/2), (lon+res/2, lat-res/2),
                    (lon-res/2, lat-res/2), (lon-res/2, lat+res/2)])


def getFireRange(firePer):
    fireDate = []
    for a, c in zip(firePer.ALARM_DATE, firePer.CONT_DATE):
        fireDate.append(pd.date_range(start=a, end=c).to_list())
    return fireDate


def getDF(path, res, dd, firePer):
    df = dd.to_dataframe().dropna().reset_index()

    # expand coordinate points to include area around it
    df['point'] = df.apply(lambda x: getSquare(x.lon, x.lat, res), axis=1)

    # check if dates had fire (true, false)
    fireDate = list(set(chain(*getFireRange(firePer))))
    df['fireDate'] = df.isin({'day': fireDate}).day

    # check if polygon had fire on a date
    ind = []
    for date in fireDate:
        fire = firePer[firePer.ALARM_DATE <= date][date <= firePer.CONT_DATE]
        pp = gpd.GeoSeries(df[df.day == date].point)
        inp, res = fire.geometry.sindex.query_bulk(pp, predicate='intersects')
        ind.extend(pp[np.isin(np.arange(0, len(pp)), inp)].index)
    df['fire'] = 0
    df['fire'][ind] = 1

    # 0 spring: 3/20 - 6/20         1 summer: 6/21 - 9/22
    # 2 autumn: 9/23 - 12/20        3 winter: 12/21 - 3/19
    df['season'] = (df['day'].dt.month*100 + df['day'].dt.day - 320) % 1300
    df.season = pd.cut(df['season'], bins=[0, 300, 602, 900, 1300],
                       labels=[0, 1, 2, 3],
                       include_lowest=True)
    # convert data type
    df = df.apply(pd.to_numeric, errors='ignore')
    df.day = pd.to_datetime(df.day)
    df = df.drop(columns=['point', 'fireDate'])
    # save dataframe
    df.to_pickle(path)
    z = zipfile.ZipFile(path+'.zip', 'w', zipfile.ZIP_DEFLATED)
    z.write(path, os.path.basename(path))
    z.close()
    # !rm data/gm.pkl

def xy(df, target):
    X = df.drop(columns=target + ['day'])
    y = df[target]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def split(df, target):
    X, X_test, y, y_test = train_test_split(xy(df, target)[0], xy(df, target)[1],
                                            test_size=0.4, random_state=S)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                      random_state=S)
    return X, X_test, y, y_test, X_train, X_val, y_train, y_val

def score(y_true, y_pred, method, p=False):
    if method == 'c':
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        TPR = tp / (tp+fn)
        FPR = fp / (fp+tn)
        PRE = tp / (tp+fp)
        f1 = f1_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        if p == True:
            print(
                f'f1: {f1:.4f}, TPR: {TPR:.4f}, FPR: {FPR:.4f}, Precision: {PRE:.4f}')
        return f1, TPR, FPR, PRE, logloss
    elif method == 'r':
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        if p == True:
            print(f'rmse: {rmse:.4f}, mae: {mae:.4f}')
        return rmse, mae


def model(X_train, y_train, X_val, y_val, method):
    # baseline model comparison
    if method == 'r':
        mm = {'Linear Regression': [LinearRegression()],
              'Decision Tree': [DecisionTreeRegressor(random_state=S)],
              # 'Random Forest': [RandomForestRegressor(random_state=S)], # really slow
              'LightGBM': [LGBMRegressor(random_state=S, verbose=-1)],
              'XGBoost': [XGBRegressor(random_state=S)],
              # 'CatBoost': [CatBoostRegressor(cat_features=['coor', 'season'], random_state=S, verbose=0)]
              }
        metrics = ['rmse', 'mae']
        ascending = True
    elif method == 'c':
        mm = {'Logistic Regression': [LogisticRegression(class_weight=W, random_state=S)],
              # slow
              'Decision Tree': [DecisionTreeClassifier(class_weight=W, random_state=S)],
              # 'Random Forest': [RandomForestClassifier(class_weight=W, random_state=S)],
              'LDA': [LinearDiscriminantAnalysis()],
              'LightGBM': [LGBMClassifier(random_state=S, verbose=-1)],
              'XGBoost': [XGBClassifier(random_state=S)],
              # 'CatBoost': [CatBoostClassifier(cat_features=['coor', 'season'], random_state=S, verbose=0)]
              }
        metrics = ['F1', 'TPR', 'FPR', 'Precision', 'logloss']
        ascending = False

    for name, m in mm.items():
        m[0].fit(X_train, y_train)
        print(name + ' - ', end=' ')
        m.append(score(y_val, m[0].predict(X_val), method))

    comp = dict()
    for k, v in mm.items():
        comp[k] = v[1]

    cc = pd.DataFrame(data=comp, index=metrics).transpose()
    display(cc.sort_values(metrics[0], ascending=ascending))
    return mm


def plotRes(y_true, y_pred):
    #     fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    #     sns.regplot(x=y_pred, y=y_true, ax=axs[0], marker='.', color='#592693',
    #             scatter_kws={'s': 10, 'alpha': 0.4},
    #             line_kws={'lw': 1, 'color': 'black'})
    #     axs[0].set(title='Predicted vs Actual Burning Index',
    #               xlabel='Predicted', ylabel='Actual')

    sns.residplot(x=y_true, y=y_pred,  # ax=axs[1],
                  scatter_kws={'s': 8, 'alpha': 0.4},)
    plt.title('Residual Plot')
    plt.xlabel('Actual')
    plt.ylabel('Residual')
    plt.show()
#     axs[1].set(title='Residual Plot', xlabel='Actual', ylabel='Residual')
    plt.show()


def plot_loss(history):
    plt.plot(np.sqrt(history.history['loss']), label='loss')
    plt.plot(np.sqrt(history.history['val_loss']), label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
