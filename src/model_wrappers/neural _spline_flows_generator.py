import torch
import pandas as pd
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows import distributions, flows


# not ever parameter from the original TabPFNGen code is used here
class NeuralSplineFlowsGenerator:
    def __init__(
        self,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 256,
        num_bins: int = 8,
        tail_bound: int = 3,
        num_transform_blocks: int = 1,
        dropout: float = 0.1,
        batch_norm: bool = False,
    ) -> None:
        self._n_layers_hidden = n_layers_hidden
        self._n_units_hidden = n_units_hidden
        self._num_bins = num_bins
        self._tail_bound = tail_bound
        self._num_transform_blocks = num_transform_blocks
        self._dropout = dropout
        self._batch_norm = batch_norm
        self._flow = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_flow(self, n_features: int):
        transforms_list = []
        transforms_list.append(RandomPermutation(features=n_features))

        transforms_list.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=n_features,
                hidden_features=self._n_units_hidden,
                num_bins=self._num_bins,
                tails="linear",
                tail_bound=self._tail_bound,
                num_blocks=self._num_transform_blocks,
                use_residual_blocks=True,
                random_mask=False,
                dropout_probability=self._dropout,
                use_batch_norm=self._batch_norm,
            )
        )
        transform = CompositeTransform(transforms_list)
        base_dist = distributions.StandardNormal(shape=[n_features])

        self._flow = flows.Flow(transform, base_dist).to(self._device)

    def fit(
        self,
        X: np.ndarray,
        sampling_patience: int = 500,
        learning_rate: float = 5e-4,
        n_iter_min: int = 100,
        batch_size: int = 1000,
    ) -> "NeuralSplineFlowsGenerator":
        self._build_flow(n_features=X.shape[1])
        optimizer = torch.optim.Adam(self._flow.parameters(), lr=learning_rate)

        best_loss = float("inf")
        patience_counter = 0

        X_tensor = torch.tensor(X, dtype=torch.float32)
        train_loader = torch.utils.data.DataLoader(
            X_tensor, batch_size=batch_size, shuffle=True
        )

        for _ in range(n_iter_min):
            self._flow.train()
            epoch_loss = 0.0

            for batch in train_loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()
                loss = -self._flow.log_prob(batch).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                best_model_state = self._flow.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= sampling_patience:
                    break

        self._flow.load_state_dict(best_model_state)
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        if self._flow is None:
            raise ValueError("The flow model has not been built. Call fit() first.")

        with torch.no_grad():
            samples = self._flow.sample(n_samples).cpu().numpy()
        return samples

    def __call__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        n_samples: int,
        balance_classes: bool,
        indices: list[int] | None = None,
        **kwargs
    ):
        data = X_train.copy()
        data["target"] = y_train
        if indices is not None:
            data = data.iloc[indices]
        self.fit(data.to_numpy(), **kwargs)
        samples = self.sample(n_samples)
        samples = pd.DataFrame(samples, columns=data.columns)
        return samples.drop(columns=["target"], axis=1), samples["target"]
