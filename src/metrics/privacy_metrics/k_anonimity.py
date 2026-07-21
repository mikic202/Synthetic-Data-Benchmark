import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from heapq import nsmallest
import random

MAX_NUMBER_OF_CLUSTERS = 20


def measure_k_anonimity_for_datset(
    dataset: pd.DataFrame, identifier_atributes: list[str] | None = None, random_state: int = 42
) -> float:
    if not identifier_atributes:
        identifier_atributes = dataset.columns.tolist()
    dataset = dataset[identifier_atributes]
    smallest_ks = []
    for n_clusters in range(2, MAX_NUMBER_OF_CLUSTERS + 1):
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = clusterer.fit_predict(dataset)
        silhouette_avg = silhouette_score(dataset, cluster_labels)
        smallest_ks.append(
            (silhouette_avg, np.unique(cluster_labels, return_counts=True)[1].min())
        )
    return nsmallest(
        1,
        nsmallest(MAX_NUMBER_OF_CLUSTERS // 4, smallest_ks, key=lambda x: -x[0]),
        key=lambda x: x[1],
    )[0][1]


def calculate_k_anonimity_for_datset(dataset: pd.DataFrame, identifier_atributes: list[str] | None = None, number_of_repetitions: int = 5):
    results = list(map(k_anonimity_process_worker, [(dataset, identifier_atributes, random.randint(0, 100)) for _ in range(number_of_repetitions)]))
    return sum(results) / len(results)

def k_anonimity_process_worker(args):
    return measure_k_anonimity_for_datset(*args)


def measure_relative_k_anonimity(real_dataset: pd.DataFrame, synthetic_dataset: pd.DataFrame, identifier_atributes: list[str] | None = None, random_state: int = 42):
    if not identifier_atributes:
        identifier_atributes = real_dataset.columns.tolist()
    real_dataset = real_dataset[identifier_atributes]
    synthetic_dataset = synthetic_dataset[identifier_atributes]
    smallest_ks = []
    for n_clusters in range(2, MAX_NUMBER_OF_CLUSTERS + 1):
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusterer.fit(synthetic_dataset)
        cluster_labels = clusterer.predict(real_dataset)
        silhouette_avg = silhouette_score(real_dataset, cluster_labels)
        smallest_ks.append(
            (silhouette_avg, np.unique(cluster_labels, return_counts=True)[1].min())
        )
    return nsmallest(
        1,
        nsmallest(MAX_NUMBER_OF_CLUSTERS // 4, smallest_ks, key=lambda x: -x[0]),
        key=lambda x: x[1],
    )[0][1]


def calculate_relative_k_anonimity_for_dataset(real_dataset: pd.DataFrame, synthetic_dataset: pd.DataFrame, identifier_atributes: list[str] | None = None, number_of_repetitions: int = 5):
    results = list(map(relative_k_anonimity_process_worker, [(real_dataset.astype(float), synthetic_dataset.astype(float), identifier_atributes, random.randint(0, 100)) for _ in range(number_of_repetitions)]))
    return sum(results) / len(results)

def relative_k_anonimity_process_worker(args):
    return measure_relative_k_anonimity(*args)