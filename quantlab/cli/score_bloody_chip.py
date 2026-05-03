import argparse
import sys
from pathlib import Path
import json

def cmd_score_bloody_chip(ns: argparse.Namespace) -> int:
    from core.data_loader import load_etf_sqlite
    from core.indicators import add_analyzer_indicators
    from core.paths import resolve_db_path
    from quantlab.scoring.bloody_chip.explain import run_explain
    
    db_path = resolve_db_path(ns.db_path)
    try:
        df, _ = load_etf_sqlite(db_path, ns.etf_code)
    except Exception as e:
        print(f"Error loading data for {ns.etf_code}: {e}", file=sys.stderr)
        return 1
        
    df = add_analyzer_indicators(df)
    
    res = run_explain(ns.etf_code, df, getattr(ns, "config", None))
    score_result = res["result"]
    
    if ns.json:
        print(score_result.model_dump_json(indent=2))
    else:
        narrative = res.get("narrative", {})
        print(f"ETF: {score_result.etf_code} | Date: {score_result.snapshot_date}")
        print(f"Category: {score_result.category}")
        print(f"Total Score: {score_result.total_score:.2f} / 10.0")
        if narrative:
            print(f"Summary: {narrative.get('summary', '')}")
            print(f"Drawdown: {narrative.get('drawdown', '')}")
            print(f"Reversal: {narrative.get('reversal', '')}")
        for dim in ("drawdown_damage", "chip_structure", "reversal_potential"):
            if dim in score_result.dimension_scores:
                ds = score_result.dimension_scores[dim]
                print(f"  - {dim}: {ds.score:.2f} ({', '.join(ds.reason_codes)})")
            
    return 0

def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("score-bloody-chip", help="Score an ETF for 'bloody chip' state")
    p.add_argument("--code", "--etf-code", dest="etf_code", required=True, help="ETF code")
    p.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p.add_argument("--config", type=Path, default=None, help="Path to bloody_chip_etf.yaml")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.set_defaults(_run=cmd_score_bloody_chip)
