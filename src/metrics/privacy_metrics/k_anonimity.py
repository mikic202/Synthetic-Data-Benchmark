import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from heapq import nsmallest


MAX_NUMBER_OF_CLUSTERS = 20


def calculate_k_anonimity_for_datset(
    dataset: pd.DataFrame, identifier_atributes: list[str] | None = None
) -> float:
    if not identifier_atributes:
        identifier_atributes = dataset.columns.tolist()
    dataset = dataset[identifier_atributes]
    smallest_ks = []
    for n_clusters in range(2, MAX_NUMBER_OF_CLUSTERS + 1):
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(dataset)
        silhouette_avg = silhouette_score(dataset, cluster_labels)
        smallest_ks.append(
            (silhouette_avg, np.unique(cluster_labels, return_counts=True)[1].min())
        )
    return nsmallest(
        1,
        nsmallest(MAX_NUMBER_OF_CLUSTERS // 5, smallest_ks, key=lambda x: -x[0]),
        key=lambda x: x[1],
    )[0][1]
