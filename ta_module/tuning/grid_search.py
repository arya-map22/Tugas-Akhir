from typing import Callable, Sequence

import optuna
from optuna import Trial

from ta_module.utils import get_current_run_datetime_str
from .tuning_result import TuningResult


def reg_coef_grid_search(
    objective_fn: Callable[[Trial], float],
    reg_coef_candidates: Sequence[float],
    storage: str,
    seed: int = 42,
) -> TuningResult:
    assert all(x >= 0 for x in reg_coef_candidates)

    run_datetime = get_current_run_datetime_str()
    study_name = f"reg_coef_grid_search_{run_datetime}"
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.GridSampler(
            search_space={"reg_coef": reg_coef_candidates}, seed=seed
        ),
        storage=storage,
        study_name=study_name,
    )

    study.optimize(
        objective_fn,
        n_trials=len(reg_coef_candidates),
        show_progress_bar=True,
    )

    result = TuningResult.model_validate(
        {
            "study_name": study_name,
            "trials": study.trials_dataframe().to_dict(orient="list"),
            "best_params": study.best_params,
        }
    )

    return result
