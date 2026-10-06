"""CLI: python -m lagkit --master master.xlsx --data vitrina.csv [--config lag_config.yaml]."""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .config import LAYOUTS, OUTPUT_FORMATS, VIEWS, LagkitError
from .processor import LagProcessor


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа командной строки; возвращает код завершения."""
    parser = argparse.ArgumentParser(
        prog="python -m lagkit",
        description="Сдвиг факторов витрины на дату доступности (lagkit)",
    )
    parser.add_argument("--master", required=True, help="путь к master-файлу (.csv/.xlsx)")
    parser.add_argument("--data", required=True, help="путь к витрине (.csv/.xlsx)")
    parser.add_argument("--config", help="путь к конфигу (.json/.yaml)")
    parser.add_argument("--view", default="known", choices=list(VIEWS) + ["all"],
                        help="какое представление сохранить (по умолчанию known)")
    parser.add_argument("--layout", choices=LAYOUTS, help="wide | long (по умолчанию из конфига)")
    parser.add_argument("--output-dir", help="каталог результатов (по умолчанию из конфига)")
    parser.add_argument("--format", choices=OUTPUT_FORMATS, help="формат файлов (по умолчанию из конфига)")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог (уровень INFO)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        lp = LagProcessor(args.master, args.data, args.config)
        paths = lp.save(view=args.view, layout=args.layout, output_dir=args.output_dir,
                        output_format=args.format)
    except LagkitError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    diag = lp.diagnostics()
    counts = diag["severity"].value_counts().to_dict()
    print(f"Факторов: {len(lp.factors)}, снимок: {lp.snapshot_date:%Y-%m-%d}, "
          f"диагностика: {counts or 'замечаний нет'}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
