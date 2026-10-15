"""Gradient boosting forecaster (LightGBM, native API).

The model predicts the RATIO load / scale_kwh from normalised features, and the scale converts the
ratio back to kWh, so a single model serves buildings of very different size. One booster gives
the point forecast (mean); one booster per quantile gives the probabilistic forecast (quantile
regression). Quantile forecasts are sorted per row so they can never cross.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from jardik import __version__
from jardik.core.logging import get_logger

log = get_logger("forecasting.lightgbm")

MEAN = "mean"


class LGBMParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ratio:    learn load / scale directly.
    # residual: learn (load - base) / scale, i.e. a correction of the base forecast (the mean of
    #           the same slot in the last weeks).
    # Defaults were chosen on VALIDATION error on Building Data Genome 2 offices (see
    # docs/DECISIONS.md): ratio with a Huber loss. The plain L2 loss ("regression") is
    # not recommended: the load/scale ratio has heavy tails and boosting stopped after ~16 trees.
    target_mode: Literal["ratio", "residual"] = "ratio"
    objective: Literal["regression", "regression_l1", "huber"] = "huber"
    huber_delta: float = Field(default=0.2, gt=0)  # in units of the scale (only for huber)
    n_estimators: int = Field(default=2000, gt=0)
    n_estimators_without_validation: int = Field(default=300, gt=0)
    early_stopping_rounds: int = Field(default=50, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    num_leaves: int = Field(default=63, gt=1)
    min_child_samples: int = Field(default=100, gt=0)
    feature_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_freq: int = Field(default=1, ge=0)
    lambda_l2: float = Field(default=1.0, ge=0)
    # Fixed so results do not depend on the number of cores of the machine.
    num_threads: int = Field(default=4, gt=0)


def _label(quantile: float) -> str:
    return f"q{quantile:g}"


class LightGBMForecaster:
    name = "lightgbm"
    supports_quantiles = True

    def __init__(
        self,
        features: list[str],
        params: LGBMParams | None = None,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        seed: int = 42,
        name: str = "lightgbm",
    ) -> None:
        self.name = name
        self.features = list(features)
        self.params = params or LGBMParams()
        self.quantiles = tuple(sorted(quantiles))
        self.seed = seed
        self.boosters: dict[str, lgb.Booster] = {}
        self.best_iterations: dict[str, int] = {}
        self.n_train_rows = 0

    # --- training -------------------------------------------------------------------------
    def _lgb_params(self, objective: str, alpha: float | None = None) -> dict[str, Any]:
        p = self.params
        out: dict[str, Any] = {
            "objective": objective,
            "learning_rate": p.learning_rate,
            "num_leaves": p.num_leaves,
            "min_data_in_leaf": p.min_child_samples,
            "feature_fraction": p.feature_fraction,
            "bagging_fraction": p.bagging_fraction,
            "bagging_freq": p.bagging_freq,
            "lambda_l2": p.lambda_l2,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": p.num_threads,
            "verbose": -1,
        }
        if alpha is not None:
            out["alpha"] = alpha
        elif objective == "huber":
            out["alpha"] = p.huber_delta
        return out

    @property
    def _residual(self) -> bool:
        return self.params.target_mode == "residual"

    def _extra(self, *names: str) -> tuple[str, ...]:
        return (*names, "base_kwh") if self._residual else names

    def _require(self, frame: pd.DataFrame, extra: tuple[str, ...]) -> None:
        missing = [c for c in [*self.features, *extra] if c not in frame.columns]
        if missing:
            raise KeyError(f"frame lacks required column(s): {missing}")

    def _xy(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        self._require(frame, self._extra("target_kwh", "scale_kwh"))
        usable = frame["target_kwh"].notna() & frame["scale_kwh"].notna() & (frame["scale_kwh"] > 0)
        if self._residual:
            usable &= frame["base_kwh"].notna()
        rows = frame[usable]
        target = rows["target_kwh"] - (rows["base_kwh"] if self._residual else 0.0)
        return rows[self.features], (target / rows["scale_kwh"]).to_numpy(float)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> None:
        x, y = self._xy(train)
        if len(x) == 0:
            raise ValueError("no usable training rows (target and scale are needed)")
        self.n_train_rows = len(x)
        dataset = lgb.Dataset(x, y, free_raw_data=False)
        valid_sets: list[lgb.Dataset] = []
        if valid is not None:
            xv, yv = self._xy(valid)
            if len(xv):
                valid_sets = [lgb.Dataset(xv, yv, reference=dataset)]

        rounds = (
            self.params.n_estimators
            if valid_sets
            else min(self.params.n_estimators, self.params.n_estimators_without_validation)
        )
        callbacks = (
            [lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)]
            if valid_sets
            else []
        )
        specs = [(MEAN, self._lgb_params(self.params.objective))] + [
            (_label(q), self._lgb_params("quantile", alpha=q)) for q in self.quantiles
        ]
        for key, lgb_params in specs:
            booster = lgb.train(
                lgb_params, dataset, rounds, valid_sets=valid_sets or None, callbacks=callbacks
            )
            self.boosters[key] = booster
            self.best_iterations[key] = (
                booster.best_iteration if valid_sets and booster.best_iteration > 0 else rounds
            )
            log.debug("%s: %d trees on %d rows", key, self.best_iterations[key], len(x))

    # --- prediction -----------------------------------------------------------------------
    def _ratio(self, key: str, frame: pd.DataFrame) -> np.ndarray:
        booster = self.boosters[key]
        return booster.predict(frame[self.features], num_iteration=self.best_iterations[key])  # type: ignore[no-any-return]

    def _check_fitted(self, frame: pd.DataFrame) -> None:
        if MEAN not in self.boosters:
            raise RuntimeError(f"{self.name} is not fitted yet")
        self._require(frame, self._extra("scale_kwh"))

    def _to_kwh(self, ratio: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        """Turn model output (ratios, or residual ratios) into kWh per hour."""
        scale = frame["scale_kwh"].to_numpy(float)
        base = frame["base_kwh"].to_numpy(float) if self._residual else 0.0
        if ratio.ndim == 2:
            scale = scale[:, None]
            base = base[:, None] if self._residual else 0.0
        return ratio * scale + base  # type: ignore[no-any-return]

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Point forecast in kWh per hour, indexed like `features`."""
        self._check_fitted(features)
        return pd.Series(self._to_kwh(self._ratio(MEAN, features), features), index=features.index)

    def predict_quantiles(self, features: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
        """Quantile forecasts in kWh per hour; one column per requested quantile."""
        self._check_fitted(features)
        missing = [q for q in quantiles if _label(q) not in self.boosters]
        if missing:
            raise ValueError(
                f"quantile(s) {missing} were not trained; trained quantiles: {list(self.quantiles)}"
            )
        all_ratios = np.column_stack([self._ratio(_label(q), features) for q in self.quantiles])
        # Sorted per row so quantiles never cross, whatever the individual boosters said.
        ordered = self._to_kwh(np.sort(all_ratios, axis=1), features)
        frame = pd.DataFrame(ordered, index=features.index, columns=list(self.quantiles))
        return frame[sorted(quantiles)]

    # --- inspection and persistence ---------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Total gain of each feature in the mean model, best first."""
        booster = self.boosters[MEAN]
        gain = booster.feature_importance(importance_type="gain")
        table = pd.DataFrame({"feature": booster.feature_name(), "gain": gain})
        return table.sort_values("gain", ascending=False, kind="stable").reset_index(drop=True)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for key, booster in self.boosters.items():
            booster.save_model(str(directory / f"{key}.txt"))
        meta = {
            "name": self.name,
            "features": self.features,
            "quantiles": list(self.quantiles),
            "params": self.params.model_dump(),
            "seed": self.seed,
            "best_iterations": self.best_iterations,
            "n_train_rows": self.n_train_rows,
            "jardik": __version__,
            "lightgbm": lgb.__version__,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> LightGBMForecaster:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        model = cls(
            meta["features"],
            LGBMParams(**meta["params"]),
            tuple(meta["quantiles"]),
            meta["seed"],
            meta["name"],
        )
        model.best_iterations = {k: int(v) for k, v in meta["best_iterations"].items()}
        model.n_train_rows = int(meta["n_train_rows"])
        for key in model.best_iterations:
            model.boosters[key] = lgb.Booster(model_file=str(directory / f"{key}.txt"))
        return model
