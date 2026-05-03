import argparse
import sys
from pathlib import Path
import json

GROUP_KEY_MAP = {
    "159209": "DIVIDEND",
    "515080": "DIVIDEND",
    "510300": "LARGE_CAP",
    "159361": "LARGE_CAP",
    "510500": "MID_CAP",
    "159531": "SMALL_CAP",
    "588000": "GROWTH_TECH",
    "159740": "HK_TECH",
    "513050": "HK_TECH",
    "512880": "BROKER",
    "510150": "CONSUMER",
    "159992": "HEALTHCARE",
    "510410": "RESOURCES",
    "515880": "TELECOM",
    "511130": "BOND",
    "518880": "GOLD",
    "513500": "US_TECH",
    "159501": "THEMATIC_MISC",
    "562060": "THEMATIC_MISC",
}


def group_key_for_code(code: str) -> str:
    return GROUP_KEY_MAP.get(str(code), f"CODE_{code}")


def cmd_rank_bloody_chip(ns: argparse.Namespace) -> int:
    from core.data_loader import list_etf_options, load_etf_sqlite
    from core.indicators import add_analyzer_indicators
    from core.paths import resolve_db_path
    from quantlab.scoring.bloody_chip.explain import run_explain
    
    db_path = resolve_db_path(ns.db_path)
    etfs = list_etf_options(db_path)
    
    if not etfs:
        print("No ETFs found in DB.", file=sys.stderr)
        return 1
        
    results = []
    
    for code, name in etfs:
        try:
            df, _ = load_etf_sqlite(db_path, code)
            df = add_analyzer_indicators(df)
            res = run_explain(code, df, getattr(ns, "config", None))
            score_result = res["result"]
            category = res.get("category", getattr(score_result, "category", "NOT_CANDIDATE"))
            narrative = res.get("narrative", {})
            results.append({
                "code": code,
                "name": name,
                "score": score_result.total_score,
                "category": category,
                "group_key": group_key_for_code(code),
                "narrative": narrative,
                "result_obj": score_result
            })
        except Exception as e:
            # Skip errors silently for ranking
            continue
            
    results = [r for r in results if r.get("category") != "NOT_CANDIDATE"]

    # Sort descending by score
    results.sort(key=lambda x: x["score"], reverse=True)

    if getattr(ns, "dedup_by", "group_key") == "group_key":
        best_by_group = {}
        for r in results:
            g = r["group_key"]
            if g not in best_by_group:
                best_by_group[g] = r
        results = list(best_by_group.values())
    
    # limit top K
    top_k = ns.top_k
    if top_k > 0:
        results = results[:top_k]
        
    if ns.json:
        # Avoid serializing the complex object directly to JSON
        json_results = [{k: v for k, v in r.items() if k != "result_obj"} for r in results]
        print(json.dumps(json_results, indent=2, ensure_ascii=False))
    else:
        print("Top Opportunities:\n")
        for r in results:
            print(f"[{r['group_key']}]")
            print(f"  - {r['code']} ({r['score']:.2f}, {r['category']})")
            nar = r.get("narrative") or {}
            if nar:
                print(f"    {nar.get('summary', '')}")
                print(f"    {nar.get('drawdown', '')}")
                print(f"    {nar.get('reversal', '')}")

            raw_tags = []
            for _, ds in r["result_obj"].dimension_scores.items():
                if ds.reason_codes:
                    raw_tags.extend(ds.reason_codes)
            if raw_tags:
                print(f"    [reason_codes]: {' | '.join(raw_tags)}")
            print("")
            
    return 0

def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("rank-bloody-chip", help="Rank all ETFs by bloody chip score")
    p.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p.add_argument("--top-k", type=int, default=10, help="Number of ETFs to display (0 for all)")
    p.add_argument("--config", type=Path, default=None, help="Path to bloody_chip_etf.yaml")
    p.add_argument("--dedup-by", default="group_key", choices=["group_key", "none"], help="Dedup opportunities by group key (default: group_key)")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.set_defaults(_run=cmd_rank_bloody_chip)
