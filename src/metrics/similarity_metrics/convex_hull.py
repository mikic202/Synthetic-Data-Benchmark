import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
from sklearn.decomposition import PCA
import numpy as np
from tabicl import TabICLClassifier, InferenceConfig
import torch
from sklearn.preprocessing import QuantileTransformer


def calculate_convex_hull_coverage(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
):
    real_data, synthetic_data = encode_data_to_contionous_space(
        real_data, synthetic_data
    )

    pca = PCA(n_components=min(7, real_data.shape[1] - 1))
    transformed_points = pca.fit_transform(real_data)
    convex_hull = ConvexHull(transformed_points)

    transformed_synthetic_points = pca.transform(synthetic_data)
    delaunay = Delaunay(convex_hull.points[convex_hull.vertices])
    inside_count = np.sum(delaunay.find_simplex(transformed_synthetic_points) >= 0)
    return inside_count / len(synthetic_data) if len(synthetic_data) > 0 else 0.0


def encode_data_to_contionous_space(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
):
    synthetic_data = synthetic_data[real_data.columns]
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    model = TabICLClassifier().fit(real_data, [0] * len(real_data))
    real_data = qt.fit_transform(real_data)
    synthetic_data = qt.transform(synthetic_data)
    inference_config = InferenceConfig()
    real_data_tensor = torch.tensor(real_data, dtype=torch.float32).unsqueeze(0)
    synthetic_data_tensor = torch.tensor(synthetic_data, dtype=torch.float32).unsqueeze(
        0
    )
    real_representations = model.model_.row_interactor(
        model.model_.col_embedder(
            real_data_tensor,
            train_size=len(real_data_tensor),
            feature_shuffles=None,
            mgr_config=inference_config.COL_CONFIG,
        ),
        mgr_config=inference_config.ROW_CONFIG,
    )
    synthetic_representation = model.model_.row_interactor(
        model.model_.col_embedder(
            synthetic_data_tensor,
            train_size=len(synthetic_data_tensor),
            feature_shuffles=None,
            mgr_config=inference_config.COL_CONFIG,
        ),
        mgr_config=inference_config.ROW_CONFIG,
    )
    return (
        real_representations.squeeze(0).cpu().detach().numpy(),
        synthetic_representation.squeeze(0).cpu().detach().numpy(),
    )
