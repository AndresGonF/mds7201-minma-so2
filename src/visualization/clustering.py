import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def weekly_cluster_distribution(y_pred, df_list):
    
    days_of_week = np.array([df.index[0].day_of_week for df in df_list])
    n_cluster, clust_counts = np.unique(y_pred, return_counts=True)
    print(f"Clusters / N° elements: {list(zip(n_cluster,clust_counts))}")
    clusters = []
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(nrows = n_cluster[-1]+1, figsize=(15,7*n_cluster[-1]+1))
    for cluster in range(n_cluster[-1]+1):
        x_cluster_days = days_of_week[y_pred == cluster]
        x_cluster, x_cluster_counts = np.unique(x_cluster_days, return_counts=True)
        x_cluster_counts = x_cluster_counts / x_cluster_counts.sum()
        day_count = np.zeros(7)
        for idx, cluster_idx in enumerate(x_cluster):
            day_count[idx] = x_cluster_counts[idx]
        ax[cluster].bar(day_names, day_count)
        ax[cluster].set_xticklabels(day_names, rotation = 45)
        ax[cluster].set_title("Cluster "+str(cluster))
        ax[cluster].set_ylabel("Frecuencia")
        plt.tight_layout()

def plot_cluster_centers(clust_centers, names, ylabel):
    n_clusters, _, n_dim = clust_centers.shape

    fig, ax = plt.subplots(n_dim, figsize=(18,5*n_dim),sharex=True)

    if n_dim == 1:
        ax = [ax]

    lines = ["-","--","-.",":","-*","-o","-^"]
    hours = pd.date_range(start = "00:00",freq="H", periods=24).strftime("%H:%M")

    for idx in range(n_clusters*n_dim):
        ax[idx//n_clusters].plot(hours, clust_centers[idx%n_clusters, :, idx%n_dim], lines[idx%n_clusters])
        ax[idx // n_clusters].set_ylabel(ylabel)
    ax[-1].set_title('Visualización de centroides de clústers')
    ax[-1].legend(names)
    ax[-1].set_xlabel("Hora del día")
    ax[-1].xaxis.set_major_locator(plt.MaxNLocator(12))
    ax[-1].xaxis.set_tick_params(rotation=15)

def cluster_distribution(labels, normalize=False, ax=None):
    if ax is None:
        ax = plt.gca()    
    n_cluster, cluster_counts = np.unique(labels,return_counts=True)
    plot_ticks = [f"Cluster {cluster}" for cluster in n_cluster]
    ax.bar(n_cluster,
            cluster_counts, 
            tick_label=plot_ticks)
    ax.set_ylabel("N° días")

def plot_series_by_cluster(labels, ts_list, cluster_center, centers=False):
    n_cluster = len(set(labels))
    n_dim = ts_list[0].shape[-1]
    fig, axs = plt.subplots(nrows=n_cluster, ncols=n_dim,figsize=(18,5*n_cluster),sharex=True)
    if n_dim ==1:
        axs = [axs]
    for idx, ts in enumerate(ts_list):
        for i, col in enumerate(ts):
            ts_copy = ts[col].copy()
            ts_copy.index = ts_copy.index.strftime("%H:%M")
            axs[labels[idx]].plot(
                                ts_copy,
                                color="gray",
                                alpha=0.4,)

    for cluster in set(labels):
        for dim in range(n_dim):
            axs[cluster, dim].plot(cluster_center[cluster, :, dim],
                            c="red",
                            linewidth=3)
            axs[cluster, dim].xaxis.set_major_locator(plt.MaxNLocator(10))
            axs[cluster, dim].set_title("Cluster "+str(cluster))
            axs[cluster, dim].set_ylabel("Tiempo [min]")
            axs[cluster, dim].set_xlabel("Fecha/Hora")

            stats = '\n'.join([
                '{:<8}{:>4.2f}'.format('Mean:', np.mean(cluster_center[cluster, dim])),
                '{:<8}{:>4.2f}'.format('SD:', np.std(cluster_center[cluster, dim]))])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            axs[cluster, dim].text(0.8, 0.1, stats, transform=axs[cluster, dim].transAxes, bbox=props)

    plt.tight_layout()
    plt.show()