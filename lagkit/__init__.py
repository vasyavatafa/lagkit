"""lagkit — обработка лагов публикации факторов.

Превращает витрину, где значение фактора стоит на отчётной дате, в таблицу, где оно
стоит на дате, когда реально стало известно (без утечки будущего).

    from lagkit import LagProcessor, LagConfig, build_views
"""
import logging

from .config import DiagnosticsConfig, LagConfig, LagkitError
from .processor import LagProcessor, build_views

__all__ = ["LagProcessor", "LagConfig", "DiagnosticsConfig", "LagkitError", "build_views"]
__version__ = "1.0.0"

logging.getLogger("lagkit").addHandler(logging.NullHandler())
