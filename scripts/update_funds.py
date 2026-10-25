#!/usr/bin/env python3
"""Builds data/funds.json for the Stock Shortlist app.

Sources (both free):
  - AMFI NAVAll.txt   : list of schemes with their SEBI category
  - mfapi.in          : full NAV history per scheme

From the NAV history it works out 3Y and 5Y return, volatility, Sharpe ratio and
rolling-return consistency (how often a fund beat the median of its category over
rolling 1-year windows). Expense ratio and manager tenure are not in NAV data, so
they are read from data/manual.csv if you fill it in.

Standard library only. Run:   python scripts/update_funds.py
Check the maths offline:      python scripts/update_funds.py --selftest
"""
import argparse
import bisect
import csv
import datetime as dt
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.request

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_URL = "https://api.mfapi.in/mf/{}"
RISK_FREE = 6.5          # % a year, used for the Sharpe ratio
GRID_STEPS = 60          # rolling windows sampled every 30 days, about 5 years
MIN_WINDOWS = 24         # fewest windows needed to publish a consistency figure
MIN_PEERS = 5            # fewest funds in a category for a window to count

EQUITY_SUBCATS = {
    "Large Cap Fund", "Mid Cap Fund", "Small Cap Fund", "Flexi Cap Fund", "Multi Cap Fund",
    "Large & Mid Cap Fund", "Focused Fund", "Value Fund", "Contra Fund",
    "Dividend Yield Fund", "ELSS",
}
CAT_RE = re.compile(r"^(Open Ended|Close Ended|Interval Fund)\s*Schemes?\s*\((.+)\)\s*$", re.I)
SKIP_NAME = re.compile(r"idcw|bonus|payout|reinvest|[-–]\s*dividend\b(?!\s+yield)|\betf\b", re.I)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def http_get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "shortlist-data-bot/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


# ---------- scheme list ----------
def parse_amfi(text):
    """Returns [(code, name, category_text)] for open-ended schemes."""
    cat, out = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = CAT_RE.match(line)
        if m:
            cat = m.group(2).strip() if m.group(1).lower().startswith("open") else None
            continue
        parts = line.split(";")
        if cat and len(parts) >= 6 and parts[0].strip().isdigit():
            out.append((parts[0].strip(), parts[3].strip(), cat))
    return out


def pick_schemes(entries):
    """Direct-plan growth schemes in the equity categories we rank."""
    chosen = []
    for code, name, cat in entries:
        low = name.lower()
        if "direct" not in low or "growth" not in low or SKIP_NAME.search(name):
            continue
        label = None
        if cat.lower().startswith("equity scheme"):
            sub = cat.split("-", 1)[-1].strip()
            if sub in EQUITY_SUBCATS:
                label = sub.replace(" Fund", "")
        elif cat.lower().replace(" ", "") == "otherscheme-indexfunds":
            label = "Index Fund"
        if label:
            chosen.append({"code": code, "name": clean_name(name), "category": label})
    return chosen


def clean_name(name):
    base = re.sub(r"\s*[-–]\s*(direct plan|direct)\b.*$", "", name, flags=re.I).strip()
    return base + " (Direct)"


# ---------- NAV history ----------
class Series:
    def __init__(self, pairs):
        pairs = sorted(set(pairs))
        self.d = [p[0] for p in pairs]
        self.v = [p[1] for p in pairs]

    def at(self, day, tol=10):
        i = bisect.bisect_right(self.d, day) - 1
        if i < 0 or day - self.d[i] > tol:
            return None
        return self.v[i]


def parse_history(text):
    j = json.loads(text)
    pairs = []
    for row in j.get("data", []):
        try:
            d = dt.datetime.strptime(row["date"], "%d-%m-%Y").date().toordinal()
            v = float(row["nav"])
        except (KeyError, ValueError):
            continue
        if v > 0:
            pairs.append((d, v))
    return Series(pairs) if len(pairs) > 100 else None


# ---------- metrics ----------
def cagr(s, asof, years):
    a = s.at(asof - int(round(365.25 * years)))
    b = s.at(asof)
    if a is None or b is None:
        return None
    return ((b / a) ** (1 / years) - 1) * 100


def volatility(s, asof):
    start = asof - 1096
    i0 = bisect.bisect_left(s.d, start)
    vs, ds = s.v[i0:], s.d[i0:]
    if len(vs) < 300 or ds[0] - start > 15:
        return None
    rets = [vs[i] / vs[i - 1] - 1 for i in range(1, len(vs))]
    return statistics.stdev(rets) * math.sqrt(252) * 100


def rolling_grid(s, asof):
    grid = []
    for k in range(GRID_STEPS):
        t = asof - 30 * k
        a, b = s.at(t - 365), s.at(t)
        grid.append(None if a is None or b is None else b / a - 1)
    return grid


def fund_metrics(s, asof):
    if s.d[-1] < asof - 15:          # stale: merged or closed
        return None
    r3, r5, vol = cagr(s, asof, 3), cagr(s, asof, 5), volatility(s, asof)
    sharpe = None
    if r3 is not None and vol:
        sharpe = (r3 - RISK_FREE) / vol
    return {"ret3": r3, "ret5": r5, "vol": vol, "sharpe": sharpe, "grid": rolling_grid(s, asof)}


def add_consistency(funds):
    """funds: list of dicts with 'category' and 'grid'. Adds 'cons' (0-100) in place."""
    by_cat = {}
    for f in funds:
        by_cat.setdefault(f["category"], []).append(f)
    for peers in by_cat.values():
        beat = {id(f): 0 for f in peers}
        seen = {id(f): 0 for f in peers}
        for k in range(GRID_STEPS):
            vals = [(f, f["grid"][k]) for f in peers if f["grid"][k] is not None]
            if len(vals) < MIN_PEERS:
                continue
            med = statistics.median(v for _, v in vals)
            for f, v in vals:
                seen[id(f)] += 1
                if v >= med:
                    beat[id(f)] += 1
        for f in peers:
            n = seen[id(f)]
            f["cons"] = round(100 * beat[id(f)] / n) if n >= MIN_WINDOWS else None


def r(x, nd):
    return None if x is None else round(x, nd)


def load_manual(path):
    out = {}
    if path and os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                code = (row.get("scheme_code") or "").strip()
                if code:
                    def num(k):
                        try:
                            return float((row.get(k) or "").strip())
                        except ValueError:
                            return None
                    out[code] = (num("expense_ratio"), num("manager_tenure_years"))
    return out


def finish(funds, manual):
    add_consistency(funds)
    out = []
    for f in funds:
        exp, ten = manual.get(f["code"], (None, None))
        out.append({
            "code": f["code"], "name": f["name"], "category": f["category"],
            "ret3": r(f["ret3"], 1), "ret5": r(f["ret5"], 1), "cons": f["cons"],
            "vol": r(f["vol"], 1), "sharpe": r(f["sharpe"], 2), "exp": exp, "ten": ten,
        })
    out.sort(key=lambda x: (x["category"], x["name"]))
    return out


# ---------- offline self-test ----------
def synthetic_funds(seed=3):
    rng = random.Random(seed)
    asof = dt.date(2026, 10, 23).toordinal()
    days = [d for d in range(asof - int(365.25 * 6.2), asof + 1) if dt.date.fromordinal(d).weekday() < 5]
    funds = []
    for cat in ("Large Cap", "Mid Cap"):
        shock = [rng.gauss(0, 1) for _ in days]
        for j in range(12):
            drift = 0.06 + 0.015 * j
            nav, pairs = 10.0, []
            for d, m in zip(days, shock):
                z = 0.9 * m + 0.2 * rng.gauss(0, 1)
                nav *= math.exp((drift - 0.5 * 0.15 ** 2) / 252 + 0.15 / math.sqrt(252) * z)
                pairs.append((d, nav))
            funds.append({"code": str(1000 * len(funds) + j), "name": f"Synthetic {cat} {j} (Direct)",
                          "category": cat, "series": Series(pairs), "drift": drift})
    return funds, asof


def selftest(write_path=None):
    funds, asof = synthetic_funds()
    ready = []
    for f in funds:
        m = fund_metrics(f["series"], asof)
        assert m, "metrics missing"
        ready.append({**f, **m})
    out = finish([{k: v for k, v in f.items() if k != "series"} for f in ready], {})
    for cat in ("Large Cap", "Mid Cap"):
        rows = [o for o in out if o["category"] == cat]
        lo, hi = rows[0], rows[-1]
        names = sorted(rows, key=lambda o: int(o["name"].split()[-2]))
        lo, hi = names[0], names[-1]
        assert hi["ret5"] > lo["ret5"] + 3, (cat, lo, hi)
        assert hi["cons"] > lo["cons"] + 20, (cat, lo, hi)
        assert 10 < lo["vol"] < 25, lo
        print(cat, "lowest drift:", lo["ret5"], lo["cons"], "| highest drift:", hi["ret5"], hi["cons"])
    # parsers
    sample = ("Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date\n\n"
              "Open Ended Schemes(Equity Scheme - Large Cap Fund)\n\nSome Mutual Fund\n\n"
              "100001;INF000;-;Alpha Bluechip Fund - Direct Plan - Growth Option;10.5;23-Oct-2026\n"
              "100002;INF001;-;Alpha Bluechip Fund - Direct Plan - IDCW;10.5;23-Oct-2026\n"
              "100003;INF002;-;Alpha Bluechip Fund - Regular Plan - Growth;10.5;23-Oct-2026\n\n"
              "Open Ended Schemes(Other Scheme - Index Funds)\n\n"
              "100004;INF003;-;Beta Nifty 50 Index Fund - Direct Plan - Growth;10.5;23-Oct-2026\n"
              "Close Ended Schemes(Equity Scheme - Large Cap Fund)\n"
              "100005;INF004;-;Gamma Fixed Term - Direct Plan - Growth;10.5;23-Oct-2026\n")
    picked = pick_schemes(parse_amfi(sample))
    assert [p["code"] for p in picked] == ["100001", "100004"], picked
    assert picked[0]["category"] == "Large Cap" and picked[1]["category"] == "Index Fund", picked
    hist = parse_history(json.dumps({"data": [{"date": "%02d-01-2025" % (i % 28 + 1), "nav": "10.%d" % i} for i in range(150)]}))
    assert hist is not None
    print("selftest passed:", len(out), "synthetic funds,", len(picked), "schemes picked from sample list")
    if write_path:
        with open(write_path, "w", encoding="utf-8") as fh:
            json.dump({"updated": "2026-10-23", "funds": out}, fh)
        print("wrote", write_path)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/funds.json")
    ap.add_argument("--manual", default="data/manual.csv")
    ap.add_argument("--max", type=int, default=0, help="only fetch this many schemes (for testing)")
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between mfapi.in requests")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--synthetic", help="with --selftest, also write the synthetic result to this file")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.synthetic)
        return

    log("Fetching AMFI scheme list...")
    schemes = pick_schemes(parse_amfi(http_get(AMFI_URL)))
    if args.max:
        schemes = schemes[: args.max]
    log(f"{len(schemes)} direct growth schemes to fetch")
    if len(schemes) < 30 and not args.max:
        log("Too few schemes found: the AMFI file format may have changed. Keeping the old data.")
        sys.exit(1)

    fetched = []
    for n, sc in enumerate(schemes, 1):
        try:
            s = parse_history(http_get(MFAPI_URL.format(sc["code"])))
        except Exception as e:
            log("skip", sc["code"], sc["name"], "->", type(e).__name__)
            s = None
        if s:
            fetched.append((sc, s))
        if n % 50 == 0:
            log(f"  {n}/{len(schemes)} fetched, {len(fetched)} usable")
        time.sleep(args.delay)
    if not fetched:
        log("No NAV history could be fetched. Keeping the old data.")
        sys.exit(1)

    asof = max(s.d[-1] for _, s in fetched)
    ready = []
    for sc, s in fetched:
        m = fund_metrics(s, asof)
        if m and any(m[k] is not None for k in ("ret3", "ret5")):
            ready.append({**sc, **m})
    funds = finish(ready, load_manual(args.manual))
    if len(funds) < 30 and not args.max:
        log("Too few funds with usable numbers. Keeping the old data.")
        sys.exit(1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"updated": dt.date.fromordinal(asof).isoformat(), "funds": funds}, fh, separators=(",", ":"))
    log(f"Wrote {len(funds)} funds to {args.out}, NAV date {dt.date.fromordinal(asof)}")


if __name__ == "__main__":
    main()
