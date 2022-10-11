import numpy
import matplotlib.pyplot as plt
import seaborn as sns

from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
seed = 0

def plot_clusters(X_train, y_pred, N_clusters, clustering_name, estimator):
    fig, ax =  plt.subplots(N_clusters, figsize = (12, N_clusters))
    for yi in range(N_clusters):
        booleano = y_pred == yi
        for i, xx in enumerate(X_train):
            if booleano[i]:
                sns.lineplot(x = range(24), y = xx.ravel(), color = "k", ax = ax[yi], alpha=.2)
        if clustering_name != "GAK KMeans":
            sns.lineplot(x = range(24), y = estimator.cluster_centers_[yi].ravel(), color = "r", ax = ax[yi], label = "Cluster {}".format(yi))
        ax[yi].set_xlim(0,24)
    ax[0].set(title=clustering_name)
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.show()
    
def test_clusterings(X_train, N_clusters, method = "Euclidean KMeans"):
    if method == "Euclidean KMeans":
        estimator = TimeSeriesKMeans(n_clusters= N_clusters, verbose=True, random_state=seed)
    elif method ==  "Soft-DTW KMeans": 
        estimator = TimeSeriesKMeans(n_clusters= N_clusters,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
    elif method == "DBA KMeans":
        estimator = TimeSeriesKMeans(n_clusters = N_clusters,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
    elif method == "KShape":
        estimator = KShape(n_clusters= N_clusters, n_init=1, random_state=seed)
    elif method == "GAK KMeans":
        estimator = KernelKMeans(n_clusters=N_clusters, kernel="gak", random_state=seed)
    print(method)
    y_pred = estimator.fit_predict(X_train)
    plot_clusters(X_train, y_pred, N_clusters, method, estimator)
    return y_pred, estimator

def generate_data_clustering(data_df, col, years, window):
    timeseries = data_df
    timeseries["week"] = timeseries.index.isocalendar().week
    timeseries["day_of_month"] = timeseries.index.day
    timeseries["month"] = timeseries.index.month
    timeseries["year"] = timeseries.index.year

    days = []
    for ano_elegido in years:
        for mes in range(12):
            for x in timeseries[[col, "day_of_month", "month", "year"]].query("month == {} and year == {}". format(mes+1, ano_elegido)).groupby("day_of_month"):
                days.append(x[1][col].interpolate().fillna(0))

    X_total = days
    fechas = dict([((s.index.day.max(), s.index.month.max(), s.index.year.max()), i) for i, s in enumerate(X_total)])
    X_train = days[fechas[window[0]]:fechas[window[1]]]
    
    return X_total, X_train, fechas
