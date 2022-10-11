import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def series_plot(data_df,ylabel_list):
    axes = data_df.plot(figsize=(20,5*data_df.shape[1]),subplots=True, sharex=True)
    for idx, ax in enumerate(axes):
        ax.set_ylabel(ylabel_list[idx])
    plt.suptitle('Visualización histórica de datos',fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def hist_plot(data_df, xlabel_list, n_rows=2, **kwargs):
    axes = data_df.hist(figsize=(20,3*data_df.shape[1]), bins=30)
    for idx in range(data_df.shape[1]):
        ax = axes[idx // n_rows, idx % n_rows]
        ax.set_xlabel(xlabel_list[idx])
        if kwargs != None:
            ax.set(**kwargs)
        ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    plt.suptitle('Visualización de la distribución de datos', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def lag_plot(data_df, lag, unit):
    n_rows = data_df.shape[1] // 3 + 1
    fig, ax = plt.subplots(n_rows, 3, figsize=(20,10))

    for idx, col in enumerate(data_df.columns):
        pd.plotting.lag_plot(data_df[col], lag=lag, ax=ax[idx//3, idx%3])
        ax[idx//3, idx%3].set_title(col)

    fig.suptitle(f'Lag plot - {lag} {unit}', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()    

def conf_matrix(data_df):
    data_corr = data_df.corr(method='pearson')
    
    heat_map = sns.heatmap(data_corr,  vmax=.8, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='viridis')
    heat_map.figure.set_size_inches(10,10)
    
    plt.show()

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
    