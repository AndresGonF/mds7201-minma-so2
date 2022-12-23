import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import numpy as np

def series_plot(data_df,ylabel_list, dates=None):
    axes = data_df.plot(figsize=(20,5*data_df.shape[1]),subplots=True, sharex=True)
    for idx, ax in enumerate(axes):
        ax.set_ylabel(ylabel_list[idx])
        if dates != None:
            # print(dates[idx])
            # date_range = pd.date_range(dates[idx],  pd.to_datetime(dates[idx])+pd.Timedelta('23H'), freq='1H')
            ax.vlines(x=dates, ymin=data_df.iloc[:,idx].min(), ymax=data_df.iloc[:,idx].max(), alpha=0.2, color='red')
    plt.suptitle('Visualización histórica de datos',fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def lag_plot(data_df, lag, unit):
    n_rows = data_df.shape[1] // 3
    if n_rows == 0:
        n_rows = 1
    fig, ax = plt.subplots(n_rows, 3, figsize=(20,10))

    for idx, col in enumerate(data_df.columns):
        if n_rows == 1:
            axes = ax[idx%3]
        else:
            axes = ax[idx//3, idx%3]
        pd.plotting.lag_plot(data_df[col], lag=lag, ax=axes)
        axes.set_title(col)

    fig.suptitle(f'Lag plot - {lag} {unit}', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()    

def conf_matrix(data_df, ax=None, abs=True):
    if ax is None:
        ax = plt.gca()
    data_corr = data_df.corr(method='pearson')
    # .loc[['SO2']].T.sort_values('SO2', ascending=False)
    if abs:
        data_corr = data_corr.abs()
    heat_map = sns.heatmap(data_corr.loc[['SO2']].T.sort_values('SO2', ascending=False), vmin=0, vmax=1, center=0, linewidths=.1,# square=True, cbar_kws={"shrink": .5},
                            annot=True, fmt='.2f', cmap='viridis')
    # heat_map.figure.set_size_inches(10,10)
    

def time_describe(data_df, col, res, from_date, to_date, highlights=False):
    df = data_df.copy()
    df[res] = df.index.strftime(f'%{res[0]}')
    stats_df = df[from_date:to_date][[res,col]].groupby(res).describe()
    if highlights:
        display(
            stats_df.style\
                .format('{:.2f}')\
                .highlight_max(color = 'green', axis = 0)\
                .highlight_min(color = 'red', axis = 0)
        )
    else:
        display(stats_df)
    