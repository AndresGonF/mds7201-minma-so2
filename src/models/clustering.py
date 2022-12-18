from time import time
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn import metrics


def bench_k_means(kmeans, name, data, n_clusters):
    t0 = time()
    data = TimeSeriesScalerMeanVariance().fit_transform(data)
    clustered = kmeans.fit_predict(data)
    fit_time = time() - t0
    inertia = kmeans.inertia_
    labels = kmeans.labels_

    results = [name, n_clusters, fit_time, inertia]

    # The silhouette score requires the full dataset
    if data.shape[-1] < 2:
        results += [
            metrics.silhouette_score(data.squeeze(), labels,
                                    metric="euclidean", sample_size=300)
        ]
    else:
        results += [0]

    # Show the results
    formatter_result = ("{:9s}\t\t{}\t{:.3f}\t{:.3f}\t\t{:.3f}")
    print(formatter_result.format(*results))
    return kmeans, clustered, inertia