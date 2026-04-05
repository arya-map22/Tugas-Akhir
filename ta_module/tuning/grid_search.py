import optuna

from typing import Callable, Sequence
from optuna import Trial

from ta_module.utils import get_current_run_datetime


def l1_eta_grid_search(
    objective_fn: Callable[[Trial], float],
    eta_candidates: Sequence[float],
    storage: str,
    seed: int = 42,
):
    assert all(x >= 0 for x in eta_candidates)

    run_datetime = get_current_run_datetime()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.GridSampler(
            search_space={"eta": eta_candidates}, seed=seed
        ),
        storage=storage,
        study_name=f"eta_grid_search_{run_datetime}",
    )

    study.optimize(
        objective_fn,
        n_trials=len(eta_candidates),
        show_progress_bar=True,
    )

    trials = study.trials_dataframe()
    best_params = study.best_params

    return {
        "trials": trials,
        "best_eta": best_params["eta"],
    }
