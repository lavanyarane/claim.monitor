"""export_report.py - write a self-contained report.html and findings.md from claims.db.

    python export_report.py

Open report.html in any browser (no Streamlit needed). It carries the same measures, charts,
interpretation and a backlog excerpt as the dashboard.
"""
import html
import re

import pandas as pd
import plotly.io as pio

import analytics as A
import charts as C
from config import DEFAULT_SLA_DAYS, ROOT


def md_to_html(s):
    s = html.escape(s)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)


def table_html(df, max_rows=25):
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(1)
    return d.fillna("").to_html(index=False, border=0, classes="tbl", escape=True)


def main():
    t = A.load_tables()
    asof = A.as_of(t)
    f = A.filter_tables(t)
    cs, sd, docs = f["cs"], f["sd"], f["docs"]
    bl = A.build_backlog(cs, DEFAULT_SLA_DAYS)
    k = A.kpis(cs, bl)
    st = A.stage_stats(sd, True)
    notes = A.insights(cs, sd, docs, bl, True)

    figs = [
        C.turnaround_hist(cs), C.backlog_by_stage(bl), C.aging_chart(A.aging_buckets(bl)),
        C.stage_duration_bars(st), C.time_share(st), C.heatmap(A.type_stage_matrix(sd, True)),
        C.missing_docs_bar(A.missing_docs_summary(docs, cs)),
        C.turnaround_by_doc_speed_box(A.turnaround_by_doc_speed(cs)),
        C.adjuster_bar(A.adjuster_stats(sd, cs)),
    ]
    divs = [pio.to_html(fig, full_html=False, include_plotlyjs=(i == 0), config={"displaylogo": False})
            for i, fig in enumerate(figs)]

    cards = [("Claims", f"{k['claims']:,}", f"{k['open']} open / {k['closed']} closed"),
             ("Median turnaround", f"{k['median_turnaround']:.0f} days", f"P90 {k['p90_turnaround']:.0f} days"),
             ("Docs missing", f"{k['pct_docs_missing']:.0f}%", "of all claims"),
             ("Incomplete cases", f"{k['pct_incomplete']:.0f}%", "any completeness flag"),
             ("Open past target", f"{k['open_breached']}", f"of {k['open']} open"),
             ("Oldest open", f"{k['oldest_age']:.0f} days", k["oldest_id"])]
    card_html = "".join(f'<div class="card"><div class="l">{a}</div><div class="v">{b}</div>'
                        f'<div class="s">{c}</div></div>' for a, b, c in cards)
    bl_cols = ["claim_id", "claim_type", "reconstructed_stage", "days_in_stage", "sla_days", "days_over_sla",
               "age_days", "waiting_on", "missing_docs", "adjuster_name", "stage_entry_source", "stage_entry_record"]
    grid = "".join(f'<div class="fig">{d}</div>' for d in divs)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claim Cycle-Time Report</title>
<style>
body{{margin:0;background:#fcfcfb;color:#0b0b0b;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;line-height:1.5}}
main{{max-width:1240px;margin:0 auto;padding:32px 20px 64px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:18px;margin:36px 0 12px}}
.sub{{color:#52514e;margin:0 0 20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.card{{background:#f3f2ee;border-radius:8px;padding:14px 16px}}
.l{{font-size:12px;color:#52514e}} .v{{font-size:26px;font-weight:600}} .s{{font-size:12px;color:#898781}}
ul{{padding-left:20px}} li{{margin:6px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}}
.fig{{min-width:0}}
.tbl{{border-collapse:collapse;font-size:12.5px;width:100%}} .tbl th{{text-align:left;color:#52514e;
border-bottom:1px solid #c3c2b7;padding:6px 8px}} .tbl td{{border-bottom:1px solid #e1e0d9;padding:5px 8px}}
.wrap{{overflow-x:auto}}
@media(max-width:640px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Claim Cycle-Time &amp; Documentation Monitor</h1>
<p class="sub">Snapshot {asof:%d %b %Y}. Time and completeness only - no claim is assessed for eligibility.</p>
<div class="cards">{card_html}</div>
<h2>Findings</h2><ul>{"".join(f"<li>{md_to_html(n)}</li>" for n in notes)}</ul>
<h2>Charts</h2><div class="grid">{grid}</div>
<h2>Backlog report (top 25 by days beyond target)</h2>
<div class="wrap">{table_html(bl[bl_cols])}</div>
<p class="sub">The full, filterable backlog with traceable source records is in the Streamlit dashboard.</p>
</main></body></html>"""
    (ROOT / "report.html").write_text(page, encoding="utf-8")

    md = ["# Findings", "", f"Snapshot date: {asof:%d %b %Y}. {k['claims']} synthetic claims "
          f"({k['open']} open, {k['closed']} closed).", ""]
    md += [f"- {n}" for n in notes]
    md += ["", "## Summary measures", "",
           "| Measure | Value |", "|---|---|",
           f"| Median total turnaround (closed) | {k['median_turnaround']:.1f} days |",
           f"| 90th percentile turnaround | {k['p90_turnaround']:.1f} days |",
           f"| Claims with a missing required document | {k['pct_docs_missing']:.1f}% |",
           f"| Cases with any completeness flag | {k['pct_incomplete']:.1f}% |",
           f"| Open claims past their stage target | {k['open_breached']} of {k['open']} |",
           "", "## Stage statistics (completed stages)", "",
           "| Stage | Completed | Median days | P90 days | Open now | Share of elapsed time |",
           "|---|---|---|---|---|---|"]
    for r in st.itertuples():
        md.append(f"| {r.stage} | {r.completed} | {r.median_days:.1f} | {r.p90_days:.1f} | {r.open_now} | "
                  f"{r.share_pct:.1f}% |")
    (ROOT / "findings.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Wrote report.html and findings.md")


if __name__ == "__main__":
    main()
