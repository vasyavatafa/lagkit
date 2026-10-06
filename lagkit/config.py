"""Конфигурация модуля lagkit: dataclass ``LagConfig`` и его загрузка из dict/JSON/YAML."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import pandas as pd


class LagkitError(ValueError):
    """Ошибка входных данных, конфигурации или нарушение инварианта модуля."""


#: Допустимые обозначения частотности и их нормализованный вид.
FREQ_ALIASES: Dict[str, str] = {"м": "м", "m": "м", "н": "н", "w": "н", "д": "д", "d": "д"}
FREQUENCIES = ("м", "н", "д")
DEFAULT_LAGS: Dict[str, int] = {"м": 30, "н": 7, "д": 1}
DEFAULT_PERIODS: Dict[str, int] = {"м": 30, "н": 7, "д": 1}

VIEWS = ("raw", "journal", "known")
LAYOUTS = ("wide", "long")
CALENDARS = ("vitrine", "daily")
OUTPUT_FORMATS = ("parquet", "csv", "xlsx")


def normalize_frequency(value: Any) -> Optional[str]:
    """Приводит обозначение частотности к ``м``/``н``/``д``; для неизвестного значения возвращает None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return FREQ_ALIASES.get(str(value).strip().lower())


def parse_date_scalar(value: Any, date_format: Optional[str] = None) -> pd.Timestamp:
    """Разбирает одиночную дату: ISO (ГГГГ-ММ-ДД) или российский формат ДД.ММ.ГГГГ."""
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    try:
        if date_format:
            ts = pd.to_datetime(str(value), format=date_format)
        elif isinstance(value, str) and re.match(r"^\s*\d{4}-\d{1,2}-\d{1,2}", value):
            ts = pd.Timestamp(value.strip())
        else:
            ts = pd.to_datetime(value, dayfirst=True)
    except (ValueError, TypeError) as exc:
        raise LagkitError(f"Не удалось разобрать дату {value!r}: {exc}") from exc
    return pd.Timestamp(ts).normalize()


def _check_keys(cls: type, data: Mapping[str, Any], where: str) -> None:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise LagkitError(
            f"Неизвестные ключи в {where}: {unknown}. Допустимые ключи: {sorted(known)}"
        )


@dataclass
class DiagnosticsConfig:
    """Пороги эвристических проверок раздела 9 ТЗ."""

    gap_factor: float = 1.5
    """Интервал между отчётами больше ``gap_factor`` периодов считается пропуском отчёта."""
    stale_periods: float = 3
    """Ряд «устарел», если последний отчёт старше снимка более чем на ``stale_periods`` периодов + лаг."""
    min_interval_share: float = 0.5
    """Для м/н: медианный интервал между отчётами меньше этой доли периода — подозрение на неверную частотность."""
    daily_change_share: float = 1 / 7
    """Для д: доля дней со сменой значения ниже порога — фактор, вероятно, не дневной."""
    daily_min_obs: int = 14
    """Минимум непустых наблюдений дневного фактора для проверки ``daily_change_share``."""
    mixed_share_diff: float = 0.5
    """Раскладка ``mixed``: доли одиночных значений в половинах истории различаются больше чем на порог."""
    mixed_min_cells: int = 3
    """Минимум непустых ячеек в каждой половине истории для вывода о ``mixed``."""
    max_gaps_in_message: int = 5
    """Сколько пропусков перечислять в тексте сообщения (полный список — в ``details``)."""

    def __post_init__(self) -> None:
        if self.gap_factor <= 1:
            raise LagkitError("diagnostics.gap_factor должен быть больше 1")
        if self.stale_periods < 0:
            raise LagkitError("diagnostics.stale_periods не может быть отрицательным")


@dataclass
class LagConfig:
    """Параметры обработки лагов. Значения по умолчанию соответствуют ТЗ.

    Создаётся в коде, из словаря (:meth:`from_dict`) или из файла JSON/YAML
    (:meth:`from_file`). Неизвестный ключ — ошибка, а не молчаливое игнорирование.
    """

    date_col: str = "DATE"
    date_format: Optional[str] = None
    layout: str = "wide"
    default_lags: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LAGS))
    periods: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_PERIODS))
    value_tol: float = 0.0
    snapshot_date: Optional[Any] = None
    calendar: str = "vitrine"
    ffill_limit: Optional[int] = None
    drop_leading_empty: bool = False
    known_date_col: str = "DATE"
    date_suffix: str = "__available_date"
    fill_flag_suffix: str = "__is_filled"
    journal_fill_flag: bool = False
    keep_empty: bool = False
    strict: bool = False
    lags_override: Optional[str] = None
    output_dir: str = "./lag_output"
    output_format: str = "parquet"
    master_sheet: Union[str, int] = 0
    data_sheet: Union[str, int] = 0
    csv_sep: Optional[str] = None
    csv_encoding: Optional[str] = None
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    def __post_init__(self) -> None:
        if isinstance(self.diagnostics, Mapping):
            _check_keys(DiagnosticsConfig, self.diagnostics, "секции diagnostics")
            self.diagnostics = DiagnosticsConfig(**self.diagnostics)
        elif not isinstance(self.diagnostics, DiagnosticsConfig):
            raise LagkitError("diagnostics должен быть словарём или DiagnosticsConfig")

        self.default_lags = self._freq_table(self.default_lags, DEFAULT_LAGS, "default_lags")
        self.periods = self._freq_table(self.periods, DEFAULT_PERIODS, "periods")
        if any(v <= 0 for v in self.periods.values()):
            raise LagkitError("periods: период частотности должен быть положительным")

        self._choice("layout", LAYOUTS)
        self._choice("calendar", CALENDARS)
        self.output_format = str(self.output_format).lower().lstrip(".")
        self._choice("output_format", OUTPUT_FORMATS)

        if self.value_tol is None or float(self.value_tol) < 0:
            raise LagkitError("value_tol должен быть неотрицательным числом")
        self.value_tol = float(self.value_tol)
        if self.ffill_limit is not None:
            if int(self.ffill_limit) != self.ffill_limit or self.ffill_limit < 0:
                raise LagkitError("ffill_limit должен быть неотрицательным целым числом дней или null")
            self.ffill_limit = int(self.ffill_limit)
        if self.snapshot_date is not None:
            self.snapshot_date = parse_date_scalar(self.snapshot_date, self.date_format)
        for name in ("date_col", "known_date_col"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise LagkitError(f"{name} должен быть непустой строкой")
        if not self.date_suffix or not self.fill_flag_suffix:
            raise LagkitError("date_suffix и fill_flag_suffix не могут быть пустыми")

    def _choice(self, name: str, allowed: tuple) -> None:
        if getattr(self, name) not in allowed:
            raise LagkitError(f"{name}={getattr(self, name)!r}; допустимые значения: {list(allowed)}")

    @staticmethod
    def _freq_table(table: Optional[Mapping[str, Any]], defaults: Mapping[str, int], name: str) -> Dict[str, int]:
        result = dict(defaults)
        for key, value in (table or {}).items():
            freq = normalize_frequency(key)
            if freq is None:
                raise LagkitError(f"{name}: неизвестная частотность {key!r}; допустимы м/н/д (m/w/d)")
            try:
                ivalue = int(value)
            except (TypeError, ValueError) as exc:
                raise LagkitError(f"{name}[{key!r}] должен быть целым числом дней") from exc
            if ivalue != value or ivalue < 0:
                raise LagkitError(f"{name}[{key!r}] должен быть неотрицательным целым числом дней")
            result[freq] = ivalue
        return result

    # --- загрузка -------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LagConfig":
        """Создаёт конфиг из словаря; неизвестные ключи — ошибка."""
        if not isinstance(data, Mapping):
            raise LagkitError("Конфиг должен быть словарём")
        _check_keys(cls, data, "конфиге")
        return cls(**dict(data))

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "LagConfig":
        """Читает конфиг из JSON или YAML (YAML требует установленного pyyaml)."""
        path = Path(path)
        if not path.exists():
            raise LagkitError(f"Файл конфига не найден: {path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(text)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise LagkitError(
                    "Для чтения YAML-конфига нужен пакет pyyaml (pip install pyyaml); "
                    "либо используйте конфиг в формате JSON"
                ) from exc
            data = yaml.safe_load(text) or {}
        else:
            raise LagkitError(f"Неподдерживаемый формат конфига {suffix!r}: ожидается .json, .yaml или .yml")
        return cls.from_dict(data)

    @classmethod
    def load(cls, config: Union[None, str, Path, Mapping[str, Any], "LagConfig"]) -> "LagConfig":
        """Принимает путь, словарь, готовый ``LagConfig`` или None (значения по умолчанию)."""
        if config is None:
            return cls()
        if isinstance(config, LagConfig):
            return config
        if isinstance(config, Mapping):
            return cls.from_dict(config)
        if isinstance(config, (str, Path)):
            return cls.from_file(config)
        raise LagkitError(f"Неподдерживаемый тип конфига: {type(config).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        """Словарь параметров (даты — в ISO-строках), пригодный для JSON."""
        data = asdict(self)
        if self.snapshot_date is not None:
            data["snapshot_date"] = self.snapshot_date.strftime("%Y-%m-%d")
        return data
