import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
from sklearn.decomposition import PCA
import numpy as np
from tabicl import TabICLClassifier, InferenceConfig
import torch


def calculate_convex_hull_coverage(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
):
    real_data = encode_data_to_contionous_space(real_data)
    synthetic_data = encode_data_to_contionous_space(synthetic_data)

    pca = PCA(n_components=min(7, real_data.shape[1] - 1))
    transformed_points = pca.fit_transform(real_data)
    convex_hull = ConvexHull(transformed_points)

    transformed_synthetic_points = pca.transform(synthetic_data)
    delaunay = Delaunay(convex_hull.points[convex_hull.vertices])
    inside_count = np.sum(delaunay.find_simplex(transformed_synthetic_points) >= 0)
    return inside_count / len(synthetic_data) if len(synthetic_data) > 0 else 0.0


def encode_data_to_contionous_space(data: pd.DataFrame):
    model = TabICLClassifier().fit(data, [0] * len(data))
    inference_config = InferenceConfig()
    data_tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)
    representations = model.model_.row_interactor(
        model.model_.col_embedder(
            data_tensor,
            train_size=len(data_tensor),
            feature_shuffles=None,
            mgr_config=inference_config.COL_CONFIG,
        ),
        mgr_config=inference_config.ROW_CONFIG,
    )
    return representations.squeeze(0).cpu().detach().numpy()
