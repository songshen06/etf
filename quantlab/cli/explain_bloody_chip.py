import argparse
import sys
from pathlib import Path

def cmd_explain_bloody_chip(ns: argparse.Namespace) -> int:
    from core.data_loader import load_etf_sqlite
    from core.indicators import add_analyzer_indicators
    from core.paths import resolve_db_path
    from quantlab.scoring.bloody_chip.explain import run_explain
    from quantlab.reports.bloody_chip_report import generate_explain_artifacts
    
    db_path = resolve_db_path(ns.db_path)
    try:
        df, _ = load_etf_sqlite(db_path, ns.etf_code)
    except Exception as e:
        print(f"Error loading data for {ns.etf_code}: {e}", file=sys.stderr)
        return 1
        
    df = add_analyzer_indicators(df)
    
    res = run_explain(ns.etf_code, df, getattr(ns, "config", None))
    score_result = res["result"]
    
    out_dir = Path(ns.output_dir)
    artifacts = generate_explain_artifacts(score_result, out_dir)
    
    print(f"Artifacts generated at {out_dir}:")
    for k, path in artifacts.items():
        print(f"  {k}: {path}")
        
    return 0

def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("explain-bloody-chip", help="Generate explanation artifacts for an ETF's bloody chip score")
    p.add_argument("--code", "--etf-code", dest="etf_code", required=True, help="ETF code")
    p.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p.add_argument("--config", type=Path, default=None, help="Path to bloody_chip_etf.yaml")
    p.add_argument("--output", "-o", dest="output_dir", type=Path, default=Path("artifacts/bloody_chip"), help="Output directory")
    p.set_defaults(_run=cmd_explain_bloody_chip)
