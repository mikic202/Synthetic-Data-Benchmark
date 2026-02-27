import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
from sklearn.decomposition import PCA
import numpy as np


def calculate_convex_hull_coverage(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
):
    real_data = encode_data_to_contionous_space(real_data)
    synthetic_data = encode_data_to_contionous_space(synthetic_data)

    pca = PCA(n_components=min(7, real_data.shape[1] - 1))
    convex_hull = ConvexHull(transformed_points)
    transformed_points = pca.fit_transform(real_data)

    transformed_synthetic_points = pca.transform(synthetic_data)
    delaunay = Delaunay(convex_hull.points[convex_hull.vertices])
    inside_count = np.sum(delaunay.find_simplex(transformed_synthetic_points) >= 0)
    return inside_count / len(synthetic_data) if len(synthetic_data) > 0 else 0.0


def encode_data_to_contionous_space(data: pd.DataFrame):
    # TODO
    return data
