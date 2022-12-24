import numpy as np
import pandas as pd
import copy

def get_SO2_peaks(df, SO2_col, peak_level):
    """Encuentra los días de peaks de SO2 determinados por un 'peak_level'
    y devuelve los datos filtrados junto con sus días.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con una columna de SO2 y datos a filtrar.
    SO2_col : string
        Columna respectiva del SO2 dentro del DataFrame.
    peak_level : int
        Número entero que determina desde qué concentracións se
        considera un peak.        

    Returns
    -------
    tuple
        tuple con un DataFrame de datos filtrados y con una lista de los
        días filtrados.            
    """    
    SO2_daily = np.array([g for n, g in df[SO2_col].groupby(pd.Grouper(freq='D')) if not g.empty], dtype=object)
    peak_mask = [(day_df > peak_level).any() for day_df in SO2_daily]
    peak_days = [day.index[0].date() for day in SO2_daily[peak_mask]]
    return SO2_daily[peak_mask], peak_days

def get_SO2_limit(df, SO2_col, peak_level):
    """Encuentra los días con concentraciones menores a 'peak_level'
    y devuelve los datos filtrados junto con sus días.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con una columna de SO2 y datos a filtrar.
    SO2_col : string
        Columna respectiva del SO2 dentro del DataFrame.
    peak_level : int
        Número entero que determina desde el valor máximo a considerar
        para guardar el valor.  

    Returns
    -------
    tuple
        tuple con un DataFrame de datos filtrados y con una lista de los
        días filtrados.            
    """        
    SO2_daily = np.array([g for n, g in df[SO2_col].groupby(pd.Grouper(freq='D')) if not g.empty], dtype=object)
    peak_mask = [(day_df > peak_level).any() == False for day_df in SO2_daily]
    peak_days = [day.index[0].date() for day in SO2_daily[peak_mask]]
    return SO2_daily[peak_mask], peak_days

def filter_by_dates(df, date_list, output_format='list'):
    """Filtra los valores de un DataFrame de acuerdo a los días a considerar
    dentro de la lista 'date_list' y los devuelve como lista o DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos a filtrar.
    data_list : list
        Lista de fechas a filtrar.
    output_format : string
        Valor que condiciona la salida hacia un DataFrame o una lista..  

    Returns
    -------
    list
        Lista de los días filtrados.
    pd.DataFrame
        DataFrame de datos filtrados
    """       
    filtered_dates = [df.loc[date.strftime('%Y-%m-%d')] for date in date_list]
    if output_format == 'list':
        return filtered_dates
    elif output_format == 'DataFrame':
        return pd.concat(filtered_dates)

def dates_to_hours(df_list):
    """Recibe una lista de DataFrames indexados por fechas y devuelve la
    misma fecha indexada por horas.

    Parameters
    ----------
    df_list : list
        Lista de DataFrames indexados por fechas.

    Returns
    -------
    list
        Lista de DataFrames indexados por hora.
    """           
    daily_df = copy.deepcopy(df_list)
    for df in daily_df:
        df.index = df.index.hour
    return daily_df

def to_season(date_list):
    """Recibe una lista de fechas y devuelve una lista con la
    correspondencia de cada elemento a una estación del año.

    Parameters
    ----------
    date_list : list
        Lista de fechas.

    Returns
    -------
    list
        Lista de estaciones correspondientes a cada una de las 
        fechas.
    """          
    season_list = []
    for date in date_list:
        date = date.strftime('%m-%d') 
        if (date > '12-21') | (date < '03-20'):
            season_list.append('Verano')
        elif (date > '03-20') & (date < '06-21'):
            season_list.append('Otoño')
        elif (date > '06-21') & (date < '09-23'):
            season_list.append('Invierno')
        else:
            season_list.append('Primavera')            
    return season_list

def df_counts(data):
    """Recibe un DataFrame y devuelve un resumen con la cantidad de
    datos, cantidad de datos esperados y el tipo de datos presente.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame al cual se le extraerá la información de la estrucutra
        de sus datos.

    Returns
    -------
    pd.DataFrame
        DataFrame que muestra el resumen de los conteos respectivos.
    """   
    df = pd.concat([data.count(), data.isna().sum()], axis=1)
    expected_dates = pd.date_range(data.index[0], data.index[-1], freq='H').shape[0]
    df.columns = ['N° datos', 'N° datos nulos']
    df['N° datos esperados'] = expected_dates
    df['Datos respecto al esperado [%]'] = (100 * df['N° datos'] / df['N° datos esperados']).round(2)
    df['Tipo de datos'] = data.dtypes
    return df

def daily_stats(df_list_daily):
    """Recibe una lista de DataFrames y devuelve otro DataFrame con distintas
    estadísticas por día.

    Parameters
    ----------
    df_list_daily : list
        list de DataFrame para calcular estadísticos diarios.

    Returns
    -------
    np.array
        Arreglo de numpy en forma de matriz con los estadísticos diarios
        calculados.
    """       
    return np.array([df.describe().loc[['min', 'max', 'mean', 'std']].values for df in df_list_daily])

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