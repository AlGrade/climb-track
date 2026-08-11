"""Lightweight manual ground-truth workflow for difficult climbing frames."""

from climbtrack.annotation.evaluation import compare_pose_session, evaluate_session
from climbtrack.annotation.session import load_session, prepare_session, save_session

__all__ = [
    "compare_pose_session",
    "evaluate_session",
    "load_session",
    "prepare_session",
    "save_session",
]
