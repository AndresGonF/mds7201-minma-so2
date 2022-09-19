from src.utils import get_project_root
import pandas as pd
import os
from functools import reduce

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def adjust_index(minma_df):
    """Combina los indices fecha y hora de un df con datos de MINMA
    y los transforma en un DatetimeIndex.
    
    Parameters
    ----------
    minma_df : pd.DataFrame
        pd.DataFrame con datos de MINMA a modificar.
    """    
    padding = '0'
    date_len = 6
    time_len = 4 
    minma_df.index = minma_df.index.set_levels([s.rjust(date_len,padding) for s in minma_df.index.levels[0].astype(str)], level=0)
    minma_df.index = minma_df.index.set_levels([s.rjust(time_len,padding) for s in minma_df.index.levels[1].astype(str)], level=1)
    minma_df.index = minma_df.index.map(' '.join)
    minma_df.index = minma_df.index.map(lambda x: '20'+x if (x[0] != '9') else '19'+x) 
    minma_df.index = pd.to_datetime(minma_df.index)


def get_minma_data(param_list, station, from_last=None, to_date=-1):
    """Recibe una lista de parámetros para una estación y entrega
    un DataFrame con datos dentro de algún período.

    Parameters
    ----------
    param_list : list[str]
        list con parámetros monitoreados a utilizar.
    station : str
        str con nombre de la estación de interés.
    from_last : str
        str indicando el tiempo hacia atrás a considerar.
    to_date : str
        str indicando la fecha hacia el presente a considerar.

    Returns
    -------
    pd.DataFrame
        pd.DataFrame con datos de cada parámetro para alguna estación.       
    """        
    param_df_list = []
    for param in param_list:
        path = get_project_root() / 'data' / 'raw' / station / f'{station}_{param}.csv'
        param_df = pd.read_csv(path,
                                sep=';',
                                usecols=range(5),
                                index_col=[0,1],
                                decimal=','
                                ).add_suffix(f'_{param}')
        param_df_list.append(param_df)
    station_df = reduce(lambda  left,right: pd.merge(left,right,on=['FECHA (YYMMDD)', 'HORA (HHMM)']), param_df_list)
    adjust_index(station_df)
    if to_date == -1:
        to_date = station_df.index[-1]
    from_date = None
    if from_last != None:
        from_date = to_date - pd.Timedelta(from_last)
    return station_df.loc[from_date:to_date]

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", type=str, help="Output filename")
    parser.add_argument("station", type=str, help="Station name")
    parser.add_argument("-p", "--param", nargs='+', help="List of parameters", required=True)
    parser.add_argument("--from_last", default=None, type=str, help="Check last period")
    parser.add_argument("--to_date", default=-1, help="Check till period")
    args = vars(parser.parse_args())

    filename = args['filename']
    station = args['station']
    params = args['param']
    from_last = args['from_last']
    to_date = args['to_date']

    output_dataset = os.path.join("data", 'processed', f"{filename}.pkl")

    dataset = get_minma_data(params, station, from_last, to_date)

    dataset.to_pickle(output_dataset)

if __name__ == '__main__':
    main()
