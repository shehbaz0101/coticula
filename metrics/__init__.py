"""VU-Bench metrics: predict, conserve, counterfactual, explain, OOD/trust."""
from .conserve import audit_burgers, audit_heat
from .counterfactual import (
    burgers_counterfactual,
    grade_burgers_cf_prediction,
    grade_heat_cf_prediction,
    heat_counterfactual,
)
from .explain import batch_score, score_explanation
from .ood import (
    generate_burgers_cases,
    generate_heat_cases,
    make_burgers_ic,
    make_heat_ic,
    resample_1d_periodic,
    resample_2d_dirichlet,
    score_burgers_labeler,
    score_burgers_surrogate,
    score_heat_labeler,
    score_heat_surrogate,
)
from .predict import batch_relative_l2, nmse, relative_l2
from .trust import (
    BURGERS_TRAIN_SUPPORT,
    HEAT_TRAIN_SUPPORT,
    TRUST_POLICY,
    attach_trust,
    decide_trust,
    param_in_range,
    refuse_silent_heatmap,
)

__all__ = [
    "relative_l2",
    "nmse",
    "batch_relative_l2",
    "audit_burgers",
    "audit_heat",
    "burgers_counterfactual",
    "heat_counterfactual",
    "grade_burgers_cf_prediction",
    "grade_heat_cf_prediction",
    "score_explanation",
    "batch_score",
    "decide_trust",
    "param_in_range",
    "attach_trust",
    "refuse_silent_heatmap",
    "TRUST_POLICY",
    "BURGERS_TRAIN_SUPPORT",
    "HEAT_TRAIN_SUPPORT",
    "make_burgers_ic",
    "make_heat_ic",
    "resample_1d_periodic",
    "resample_2d_dirichlet",
    "generate_burgers_cases",
    "generate_heat_cases",
    "score_burgers_labeler",
    "score_heat_labeler",
    "score_burgers_surrogate",
    "score_heat_surrogate",
]
