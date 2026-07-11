import numpy as np
import pandas as pd
from feature_order.corelation_based_order import (
    generate_correlation_based_order_of_features,
)
import networkx as nx
import matplotlib.pyplot as plt


def generate_graph_form_adjecency_matrix(
    adjacency_matrix: np.ndarray,
    mylabels: list[str],
    connection_cutoff: float = 0.3,
    draw: bool = False,
) -> nx.Graph:
    rows, cols = np.where(abs(adjacency_matrix) >= connection_cutoff)
    graph = nx.Graph()
    all_rows = range(0, adjacency_matrix.shape[0])
    for n in all_rows:
        graph.add_node(mylabels[n])
    for row, column in zip(rows, cols):
        weight = adjacency_matrix[row, column]
        graph.add_edge(mylabels[row], mylabels[column], weight=weight)
    if draw:
        nx.draw(
            graph,
            node_size=900,
            labels={label: label for label in mylabels},
            with_labels=True,
        )
        plt.show()
    return graph


def generate_graph_based_order_of_features(dataset: pd.DataFrame) -> list[str]:
    correlation_matrix = dataset.corr().to_numpy()
    np.fill_diagonal(correlation_matrix, 0.0)
    graph = generate_graph_form_adjecency_matrix(
        correlation_matrix,
        mylabels=dataset.columns.tolist(),
        connection_cutoff=0.2,
    )
    correlation_matrix_stats = list(
        reversed(
            generate_correlation_based_order_of_features(dataset.corr()).index.tolist()
        )
    )
    start_of_search = correlation_matrix_stats.pop()
    visited_nodes = []
    while len(visited_nodes) != len(graph.nodes):
        if start_of_search in visited_nodes:
            start_of_search = correlation_matrix_stats.pop()
            continue
        visited_nodes.append(start_of_search)
        if len(graph[start_of_search]) == 0:
            if not correlation_matrix_stats:
                break
            start_of_search = correlation_matrix_stats.pop()
            continue

        not_visited = set(graph[start_of_search].keys()) - set(visited_nodes)

        sorted_importance = sorted(
            list(not_visited),
            key=lambda x: abs(graph[start_of_search][x]["weight"]),
            reverse=True,
        )
        if not sorted_importance:
            start_of_search = correlation_matrix_stats.pop()
            continue
        start_of_search = sorted_importance[0]
    return visited_nodes
