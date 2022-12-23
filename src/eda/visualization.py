
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
from src.eda.processing import to_season, daily_stats
import numpy as np
from math import ceil


def hist_plot(data_df, xlabel_list, **kwargs):
    """Recibe un DataFrame y grafica histogramas de todas sus columnas.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame con columnas para hacer histogramas.
    xlabel_list : list
        Lista de nombres para utilizar en el eje x de acuerdo a
        cada columna

    Returns
    -------
    None
    """
    n_rows = ceil(data_df.shape[1] / 2)
    axes = data_df.hist(figsize=(20,3*data_df.shape[1]), bins=30, layout=(n_rows, 2))
    print()
    for idx in range(data_df.shape[1]):
        ax = axes[idx % 2, idx % 2]
        ax.set_xlabel(xlabel_list[idx])
        if kwargs != None:
            ax.set(**kwargs)
        ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    plt.suptitle('Visualización de la distribución de datos')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def conf_matrix(data_df, SO2_only=False, abs=True, ax=None):
    """Recibe un DataFrame y grafica una matriz de correlaciones entre
    sus variables.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame en estudio para calcular correlaciones.
    SO2_only : boolean
        Condición para determinar si se grafican solo las correlaciones
        hacia 'SO2', ordenadas de manera descendente.
    abs : boolean
        Condición para utilizar los valores absolutos de las
        correlaciones al graficar.
    ax : axes (optional)
        Axes en donde se posicionará el gráfico.

    Returns
    -------
    None
    """       
    if ax is None:
        ax = plt.gca()
    data_corr = data_df.corr(method='pearson')
    square = True
    if SO2_only:
        data_corr = data_corr.loc[['SO2']].T.sort_values('SO2', ascending=False)
        square = False
    if abs:
        data_corr = data_corr.abs()
    heat_map = sns.heatmap(data_corr, vmin=0, vmax=1, center=0, linewidths=.1, cbar_kws={"shrink": .5}, square=square,
                            annot=True, fmt='.2f', cmap='viridis', ax=ax)

def lag_plot(data_df, lag, unit):
    """Recibe un DataFrame y para dado desfase en el tiempo grafica un 
    lag plot para cada columna.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame en estudio para graficar.
    lag : int
        Cantidad de desfase a considerar en los gráficos. Depende 
        de la frecuencia de los datos.
    unit : string
        String que indica la frecuencia de los datos para indicar
        a cuánto equivale cada desfase. Solo influye en el título.

    Returns
    -------
    None
    """    
    n_rows = ceil(data_df.shape[1] / 3)
    if n_rows == 0:
        n_rows = 1
    fig, ax = plt.subplots(n_rows, 3, figsize=(20,5*n_rows))

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


def plot_peak_counts(data_peaks_df, data_df, estacion, include_days=True, include_weekday=True, include_seasons=True):
    """Recibe una lista de DataFrames con días de peaks, una lista de DataFrames
    con días normales y grafica el conteo de variables comparadas, según la estación.
    
    Parameters
    ----------
    data_peaks_df : list
        Lista de DataFrames con días de peaks.
    data_df : list
        Lista de DataFrames con días sin peaks.
    estacion : string
        String que indica la estación a graficar. Solo para el título del
        gráfico.
    include_days : boolean
        Condición para incluir el conteo de días normales y con peaks.
    include_weekday : boolean
        Condición para incluir el conteo de peaks por día de semana.
    include_seasons : boolean
        Condición para incluir el conteo de peaks según la estación del
        año.

    Returns
    -------
    None
    """        
    peak_days = [date.index[0].date() for date in data_peaks_df]
    
    n_cols = include_days + include_weekday + include_seasons
    fig, axes = plt.subplots(1, n_cols, figsize=(15,5))

    if n_cols == 1:
        axes = np.array(axes)

    for idx, ax in enumerate(axes.ravel()):
        if include_days:
            include_days = False
            bar = sns.barplot(x=['Días de peak', 'Días normales'], y=[len(data_peaks_df), len(data_df)], ax=ax)
            ax.set_ylabel('N')
            ax.xaxis.set_ticklabels(ax.get_xticklabels(), rotation=20)
            for i in bar.containers:
                bar.bar_label(i,)
        elif include_weekday:
            include_weekday = False
            bar = sns.countplot(x=list(map(lambda x: x.weekday(), peak_days)), ax=ax)
            ax.set_ylabel('N')
            ax.set_xlabel('Día de la semana')
            ax.xaxis.set_ticklabels(['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'], rotation=20)
            for i in bar.containers:
                bar.bar_label(i,)
        elif include_seasons:
            include_seasons = False
            bar = sns.countplot(x=to_season(peak_days), ax=ax)
            ax.set_ylabel('N')
            ax.set_xlabel('Estación del año')
            ax.xaxis.set_ticklabels(ax.get_xticklabels(), rotation=20)
            for i in bar.containers:
                bar.bar_label(i,)

    fig.suptitle(f'Conteo de peaks de SO2 - {estacion}')
    plt.tight_layout()
    plt.show()

def plot_temp(df_daily, station, type='cumdistr', figsize=(20,80)):
    """Recibe una lista de DataFrames y grafica la distribución acumulada o el
    histograma de las estadísticas diarias de cada columna, para una 
    determinada estación.
    
    Parameters
    ----------
    df_daily : list
        Lista de listas DataFrames separados por días a considerar para comparar.
    station : string
        String que indica la estación a graficar. Solo para el título del
        gráfico.
    type : string
        String que indica el tipo de gráfico a implementar, puede ser
        'cumdistr' para la distribución acumulada y 'hist' para histogramas.
    figsize : tuple
        Tupla para controlar el tamaño del gráfico.

    Returns
    -------
    None
    """      
    metrics = ['min', 'max', 'mean', 'std']
    columns = df_daily[0][0].columns
    n_cols = columns.shape[0]
    stats_daily = [daily_stats(df) for df in df_daily]

    fig = plt.figure(constrained_layout=True, figsize=(20, 5*n_cols))

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=n_cols, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle('- '*20 + f' {columns[row]} - ' + station + '- '*20 , fontsize=16)

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=len(metrics))
        for col, ax in enumerate(axs):
            for df in stats_daily:
                if type=='cumdistr':
                    sns.ecdfplot(df[:, col, row], ax=ax)
                elif type=='hist':
                    ax.hist(df[:, col, row], alpha=0.5, density=True, bins=30)
            # ax.hist(stats_daily2[:, col, row], alpha=0.5, density=True, bins=30)
            ax.legend(['Peak days', 'Normal days'])
            ax.set_xlabel(metrics[col])
    plt.tight_layout()
    plt.show()

def cumdistr_comparison(df_daily, station, columns, x_log=False):
    fig, axes = plt.subplots(2,2,figsize=(20, 10))

    stats_daily = [daily_stats(df) for df in df_daily]
    style = ['-','--']
    colors = sns.color_palette('Paired')
    metrics = ['min', 'max', 'mean', 'std']
    col_idxs = [df_daily[0][0].columns.tolist().index(col) for col in columns]

    for jdx, ax in enumerate(axes.ravel()):
        for idx, df in enumerate(stats_daily):
            for j, i in enumerate(col_idxs):
                temp = df[:, jdx, i]
                temp_plot = sns.ecdfplot(temp, ax=ax, linestyle=style[idx], color=colors[j])
                if x_log:
                    ax.set_xscale('log')
                ax.set_xlabel(metrics[jdx])
        first_legend = ax.legend(df_daily[0][0].columns[col_idxs], loc='upper right')
        ax.add_artist(first_legend)
        line1 = mlines.Line2D([], [], color='black', linestyle='-', label='Peak days')
        line2 = mlines.Line2D([], [], color='black', linestyle='--', label='Normal days')                          
        ax.legend(handles=[line1,line2], loc='lower right')
    fig.suptitle(f'Distribución acumulada de métricas diarias - {station}')
    plt.tight_layout()
    plt.show()

def cumdistr_comparacion_horaria(df1, df2, df1_label, df2_label, column):
    fig, axes = plt.subplots(6,4,figsize=(20,30))

    for idx, ax in enumerate(axes.ravel()):
        sns.ecdfplot(df1.loc[df1.hour==idx, column], ax=ax)
        sns.ecdfplot(df2.loc[df2.hour==idx, column], ax=ax)
        ax.set_title(f'Distribución acumulada \na las {idx}:00 hrs')
        ax.legend([df1_label, df2_label])
    fig.suptitle(f'Comparación de distribución acumulada por hora - {column}', y=0.99)
    plt.tight_layout()

def plot_estabilidad_TDiff(data_df):
    df_days = np.array([g for n, g in data_df[['T_diff']].groupby(pd.Grouper(freq='D')) if not g.empty], dtype=object)
    stats_daily = daily_stats(df_days)

    fig, ax = plt.subplots(figsize=(15,8))

    ax.hist(stats_daily[:,0,0],bins=40, alpha=0.7)
    A = ax.vlines(-1.9, 0, 280, linestyles='dashed', color='red')
    B= ax.vlines(-1.7, 0, 280, linestyles='dashed', color='blue')
    C=ax.vlines(-1.5, 0, 280, linestyles='dashed', color='green')
    D=ax.vlines(-0.5, 0, 280, linestyles='dashed', color='yellow')
    E=ax.vlines(1.5, 0, 280, linestyles='dashed', color='brown')
    F=ax.vlines(4.0, 0, 280, linestyles='dashed', color='black')
    ax.legend([A,B,C,D,E,F], ['-1.9','-1.7','-1.5','-0.5','1.5','4.0'])
    ax.set_ylim(0,200)
    ax.set_xlabel('T_diff mean')
    ax.set_title('Distribución de promedios diarios de T_diff')

def plot_estabilidad_SigDir(data_df):
    df_days = np.array([g for n, g in data_df[['SigDir_10', 'SigDir_20', 'SigDir_40']].groupby(pd.Grouper(freq='D')) if not g.empty], dtype=object)
    fig, ax = plt.subplots(figsize=(15,8))

    stats_daily = daily_stats(df_days) 

    ax.hist(stats_daily[:,0,0],bins=40, alpha=0.5)
    ax.hist(stats_daily[:,0,1],bins=40, alpha=0.5)
    ax.hist(stats_daily[:,0,2],bins=40, alpha=0.5)
    ax.legend(['SigDir_10', 'SigDir_20', 'SigDir_40'])
    A = ax.vlines(25, 0, 280, linestyles='dashed', color='red')
    B= ax.vlines(20, 0, 280, linestyles='dashed', color='blue')
    C=ax.vlines(15, 0, 280, linestyles='dashed', color='green')
    D=ax.vlines(10, 0, 280, linestyles='dashed', color='yellow')
    E=ax.vlines(5, 0, 280, linestyles='dashed', color='brown')
    F=ax.vlines(2.5, 0, 280, linestyles='dashed', color='black')
    G=ax.vlines(1.7, 0, 280, linestyles='dashed', color='lightblue')
    ax.set_ylim(0,200)
    ax.set_xlabel('SigDir mean')
    ax.set_title('Distribución de promedios diarios\nde SigDir a distintas alturas') 