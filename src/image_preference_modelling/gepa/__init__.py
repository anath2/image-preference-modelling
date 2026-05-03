from image_preference_modelling.gepa.scoring import score_rollout_feedback

__all__ = ["run_gepa_optimization", "score_rollout_feedback"]


def __getattr__(name: str):
    if name == "run_gepa_optimization":
        from image_preference_modelling.gepa.optimizer import run_gepa_optimization

        return run_gepa_optimization
    raise AttributeError(name)
