"""Insurance Claim Cycle-Time & Documentation Monitor - Streamlit dashboard.

    streamlit run app.py

Reads claims.db (built by pipeline.py). Measures time and completeness only; it never decides
whether a claim is eligible or payable.
"""
import re
import sqlite3

import pandas as pd
import streamlit as st

import analytics as A
import charts as C
from config import DB_PATH, DEFAULT_SLA_DAYS, STALLED_DAYS, WORK_STAGES

st.set_page_config(page_title="Claim Cycle-Time Monitor", page_icon=None, layout="wide")


def _r(df):
    """Round numeric columns only (date columns are left alone)."""
    return df.round({c: 1 for c in df.select_dtypes("number").columns})

st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1400px;}
    [data-testid="stMetricValue"] {font-size: 1.7rem;}
    .note {color:#52514e; font-size:0.9rem;}
    </style>""", unsafe_allow_html=True)

if not DB_PATH.exists():
    st.error("claims.db not found. Run `python generate_data.py` and then `python pipeline.py` first.")
    st.stop()


@st.cache_data(show_spinner=False)
def get_tables(mtime):
    return A.load_tables(DB_PATH)


T = get_tables(DB_PATH.stat().st_mtime)
AS_OF = A.as_of(T)
cs_all = T["case_summary"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Filters")
    types = st.multiselect("Claim type", sorted(cs_all["claim_type"].dropna().unique()))
    regions = st.multiselect("Region", sorted(cs_all["region"].dropna().unique()))
    dmin, dmax = cs_all["reported_at"].min().date(), cs_all["reported_at"].max().date()
    rng = st.date_input("Reported between", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    d_from, d_to = (rng if isinstance(rng, (list, tuple)) and len(rng) == 2 else (dmin, dmax))
    clean_only = st.checkbox("Exclude durations that absorb a stage with no evidence", value=True,
                             help="When a stage has no record at all, the time before the next known stage "
                                  "also contains that missing stage. Leaving these out keeps per-stage "
                                  "statistics honest.")
    with st.expander("Service targets (days per stage)"):
        sla = {s: st.number_input(s, min_value=0.0, value=float(DEFAULT_SLA_DAYS[s]), step=1.0, key=f"sla_{s}")
               for s in WORK_STAGES}
    st.caption(f"Snapshot date: {AS_OF:%d %b %Y}. Open claims are aged to this date.")

F = A.filter_tables(T, types or None, regions or None, d_from, d_to)
cs, sd, tl, docs = F["cs"], F["sd"], F["tl"], F["docs"]

st.title("Claim Cycle-Time & Documentation Monitor")
st.markdown('<p class="note">Rebuilds every claim\'s case history from status events, documents, adjuster '
            "assignments and payments. Measures time and completeness only; it does not decide eligibility.</p>",
            unsafe_allow_html=True)
if cs.empty:
    st.warning("No claims match the current filters.")
    st.stop()

bl = A.build_backlog(cs, sla)
K = A.kpis(cs, bl)
st_stats = A.stage_stats(sd, clean_only)

tabs = st.tabs(["Overview", "Stages & bottlenecks", "Incomplete cases", "Backlog report", "Case inspector",
                "Data quality", "SQL console", "Method"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    c = st.columns(6)
    c[0].metric("Claims", f"{K['claims']:,}", f"{K['open']} open / {K['closed']} closed", delta_color="off")
    c[1].metric("Median turnaround", f"{K['median_turnaround']:.0f} days" if K["closed"] else "n/a",
                f"P90 {K['p90_turnaround']:.0f} days" if K["closed"] else None, delta_color="off")
    c[2].metric("Docs missing", f"{K['pct_docs_missing']:.0f}%", "of all claims", delta_color="off")
    c[3].metric("Incomplete cases", f"{K['pct_incomplete']:.0f}%", "any completeness flag", delta_color="off")
    c[4].metric("Open past target", f"{K['open_breached']}", f"of {K['open']} open", delta_color="off")
    c[5].metric("Oldest open", f"{K['oldest_age']:.0f} days" if K["open"] else "n/a", K["oldest_id"],
                delta_color="off")

    st.subheader("What the data says")
    for line in A.insights(cs, sd, docs, bl, clean_only):
        st.markdown(f"- {line}")

    a, b = st.columns(2)
    a.plotly_chart(C.turnaround_hist(cs), width="stretch")
    b.plotly_chart(C.backlog_by_stage(bl), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(C.aging_chart(A.aging_buckets(bl)), width="stretch")
    b.plotly_chart(C.turnaround_by_type(cs), width="stretch")
    st.caption("Turnaround counts closed claims only. Open claims are shown in the backlog charts because "
               "their final duration is not known yet.")

# ------------------------------------------------------------------ stages
with tabs[1]:
    st.subheader("Where does the time go?")
    top = st_stats.sort_values("share_pct", ascending=False).iloc[0]
    st.markdown(f"**{top['stage']}** is the bottleneck: {top['share_pct']:.0f}% of all elapsed claim-days, "
                f"median {top['median_days']:.1f} days for completed cases, {int(top['open_now'])} open claims "
                "currently waiting there.")
    a, b = st.columns(2)
    a.plotly_chart(C.stage_duration_bars(st_stats), width="stretch")
    b.plotly_chart(C.time_share(st_stats), width="stretch")
    st.plotly_chart(C.heatmap(A.type_stage_matrix(sd, clean_only)), width="stretch")
    a, b = st.columns(2)
    adj = A.adjuster_stats(sd, cs, clean_only=clean_only)
    a.plotly_chart(C.adjuster_bar(adj), width="stretch")
    b.markdown("**Reading the adjuster chart**")
    b.markdown("Median days in Assessment for completed cases, by the claim's latest assigned adjuster "
               "(adjusters with fewer than 10 completed cases are hidden). Differences can reflect workload and "
               "claim mix as much as individual performance.")
    with st.expander("Table view: stage statistics"):
        show = st_stats.rename(columns={
            "stage": "Stage", "completed": "Completed", "median_days": "Median days", "mean_days": "Mean days",
            "p90_days": "P90 days", "open_now": "Open now", "open_median_days": "Open: median days so far",
            "elapsed_days": "Elapsed claim-days", "share_pct": "Share %", "bottleneck_rank": "Rank"})
        st.dataframe(_r(show), width="stretch", hide_index=True)

# ------------------------------------------------------------------ incomplete
with tabs[2]:
    st.subheader("Incomplete cases")
    inc = cs[cs["is_incomplete"]]
    reasons = {
        "Required documents missing": (~cs["docs_complete"]).sum(),
        "Progressed with documents missing": cs["progressed_with_missing_docs"].sum(),
        "History gap (stage with no evidence)": (cs["n_gap_stages"] > 0).sum(),
        "Status field disagrees with history": cs["status_mismatch"].sum(),
        "No adjuster record": (cs["incomplete_reasons"].str.contains("adjuster", na=False)).sum(),
        "No final payment record": cs["no_final_payment_record"].sum(),
    }
    c = st.columns(len(reasons))
    for col, (name, v) in zip(c, reasons.items()):
        col.metric(name, int(v))
    a, b = st.columns(2)
    a.plotly_chart(C.missing_docs_bar(A.missing_docs_summary(docs, cs)), width="stretch")
    b.plotly_chart(C.turnaround_by_doc_speed_box(A.turnaround_by_doc_speed(cs)), width="stretch")
    st.plotly_chart(C.doc_receipt_bar(A.doc_receipt_times(docs)), width="stretch")
    st.markdown(f"**{len(inc)} incomplete cases** ({100 * len(inc) / len(cs):.0f}% of claims). Pick a reason to narrow:")
    flt = st.multiselect("Show cases with", list(reasons), key="inc_filter")
    shown = inc
    needles = {"Required documents missing": "required document", "Progressed with documents missing": "progressed",
               "History gap (stage with no evidence)": "history gap", "Status field disagrees with history": "status field",
               "No adjuster record": "adjuster", "No final payment record": "final payment"}
    for f in flt:
        shown = shown[shown["incomplete_reasons"].str.contains(needles[f], case=False, na=False)]
    cols = ["claim_id", "claim_type", "reconstructed_stage", "is_open", "n_docs_missing", "missing_docs",
            "incomplete_reasons", "total_turnaround_days", "age_days"]
    st.dataframe(_r(shown[cols]), width="stretch", hide_index=True, height=360)
    st.download_button("Download incomplete cases (CSV)", shown[cols].to_csv(index=False).encode(),
                       "incomplete_cases.csv", "text/csv")

# ------------------------------------------------------------------ backlog
with tabs[3]:
    st.subheader("Backlog report")
    st.markdown('<p class="note">Every open claim, with the stage it is in, how long it has been there, what it is '
                "waiting on, and the record that proves it. Sorted by days beyond the stage target.</p>",
                unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    stage_sel = f1.multiselect("Stage", [s for s in WORK_STAGES if s in set(bl["reconstructed_stage"])])
    wait_sel = f2.multiselect("Waiting on", sorted(bl["waiting_on"].dropna().unique()))
    only_breach = f3.checkbox("Past target only")
    only_stalled = f4.checkbox(f"No activity for {STALLED_DAYS}+ days")
    v = bl.copy()
    if stage_sel:
        v = v[v["reconstructed_stage"].isin(stage_sel)]
    if wait_sel:
        v = v[v["waiting_on"].isin(wait_sel)]
    if only_breach:
        v = v[v["sla_breached"]]
    if only_stalled:
        v = v[v["stalled"]]
    cols = ["claim_id", "claim_type", "region", "reported_at", "reconstructed_stage", "stage_entered_at",
            "days_in_stage", "sla_days", "days_over_sla", "age_days", "waiting_on", "missing_docs",
            "adjuster_name", "days_since_last_activity", "stalled", "stage_entry_source", "stage_entry_record",
            "stage_entry_confidence", "incomplete_reasons"]
    st.caption(f"{len(v)} of {len(bl)} open claims shown. Pick a row's claim id in the Case inspector to see "
               "the full history and original records.")
    st.dataframe(_r(v[cols]), width="stretch", hide_index=True, height=520)
    st.download_button("Download backlog report (CSV)", v[cols].to_csv(index=False).encode(),
                       "backlog_report.csv", "text/csv")

# ------------------------------------------------------------------ inspector
with tabs[4]:
    st.subheader("Case inspector")
    default_ids = list(bl["claim_id"]) + [i for i in cs["claim_id"] if i not in set(bl["claim_id"])]
    cid = st.selectbox("Claim", default_ids, help="Open claims are listed first")
    row = cs[cs["claim_id"] == cid].iloc[0]
    c = st.columns(5)
    c[0].metric("Type / region", f"{row['claim_type']}", row["region"], delta_color="off")
    c[1].metric("Rebuilt stage", row["reconstructed_stage"], "open" if row["is_open"] else "closed",
                delta_color="off")
    c[2].metric("Turnaround" if not row["is_open"] else "Age",
                f"{(row['total_turnaround_days'] if not row['is_open'] else row['age_days']):.0f} days")
    c[3].metric("Documents", f"{row['n_required_docs'] - row['n_docs_missing']}/{row['n_required_docs']}",
                "complete" if row["docs_complete"] else f"missing: {row['missing_docs']}", delta_color="off")
    c[4].metric("Adjuster", row["adjuster_name"] or "none on record", row["adjuster_id"], delta_color="off")
    if row["is_incomplete"]:
        st.warning("Completeness flags: " + row["incomplete_reasons"])
    closed_at = tl.loc[(tl["claim_id"] == cid) & (tl["stage"] == "Closed"), "entered_at"]
    st.plotly_chart(C.case_timeline(sd[sd["claim_id"] == cid], closed_at.iloc[0] if len(closed_at) else None),
                    width="stretch")
    st.markdown("**Rebuilt timeline** - each stage entry and the record it came from")
    ct = tl[tl["claim_id"] == cid][["stage", "entered_at", "source", "source_record_id", "confidence", "adjusted",
                                    "note"]]
    st.dataframe(ct, width="stretch", hide_index=True)
    st.markdown("**Original records (before cleaning)**")
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    for label, table in [("Claim row(s)", "raw_claims"), ("Status history", "raw_status_history"),
                         ("Documents", "raw_documents"), ("Adjuster assignments", "raw_adjusters"),
                         ("Payments", "raw_payments")]:
        raw = pd.read_sql_query(f"SELECT * FROM {table} WHERE claim_key = ? ORDER BY _row", con, params=(cid,))
        with st.expander(f"{label} ({len(raw)})", expanded=table == "raw_status_history"):
            st.dataframe(raw.drop(columns=["claim_key"]), width="stretch", hide_index=True)
    con.close()
    st.caption("_row is the line number in the source CSV (header = line 1).")

# ------------------------------------------------------------------ DQ
with tabs[5]:
    st.subheader("Data-quality log")
    st.markdown("Every correction made on the way from raw files to the analysis tables.")
    dq = T["dq_log"]
    a, b = st.columns(2)
    a.metric("Issue types found", len(dq))
    b.metric("Records touched", f"{int(dq['records'].sum()):,}")
    st.dataframe(dq.sort_values(["table", "records"], ascending=[True, False]), width="stretch", hide_index=True,
                 height=560)

# ------------------------------------------------------------------ SQL
with tabs[6]:
    st.subheader("SQL console")
    st.markdown("Read-only queries against the processed database. Tables: "
                "`case_summary`, `stage_durations`, `case_timeline`, `backlog_report`, `fact_*`, `dim_claims`, "
                "`dq_log` and the untouched `raw_*` copies.")
    samples = {
        "Median days per stage (completed, clean)":
            "SELECT stage, COUNT(*) AS n, ROUND(AVG(duration_days),1) AS mean_days\n"
            "FROM stage_durations WHERE is_open = 0 AND absorbs_missing_stages = 0\nGROUP BY stage ORDER BY MIN(stage_order);",
        "Ten oldest open claims":
            "SELECT claim_id, claim_type, reconstructed_stage, ROUND(age_days) AS age_days, waiting_on, missing_docs\n"
            "FROM case_summary WHERE is_open = 1 ORDER BY age_days DESC LIMIT 10;",
        "Open claims missing documents, by type":
            "SELECT claim_type, COUNT(*) AS claims, ROUND(AVG(days_in_stage),1) AS avg_days_in_stage\n"
            "FROM case_summary WHERE is_open = 1 AND n_docs_missing > 0 GROUP BY claim_type;",
        "Where each rebuilt stage time came from":
            "SELECT source, confidence, COUNT(*) AS stage_entries FROM case_timeline GROUP BY source, confidence\nORDER BY stage_entries DESC;",
    }
    pick = st.selectbox("Example queries", list(samples))
    q = st.text_area("Query", value=samples[pick], height=150, key=f"q_{pick}")
    if st.button("Run query", type="primary"):
        stmt = q.strip().rstrip(";")
        if not re.match(r"(?is)^(select|with)\b", stmt) or ";" in stmt:
            st.error("Only a single SELECT (or WITH ... SELECT) statement is allowed.")
        else:
            try:
                con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
                con.execute("PRAGMA query_only = ON")
                res = pd.read_sql_query(stmt, con)
                con.close()
                st.caption(f"{len(res)} rows")
                st.dataframe(res, width="stretch", hide_index=True)
                st.download_button("Download result (CSV)", res.to_csv(index=False).encode(), "query_result.csv",
                                   "text/csv")
            except Exception as e:  # noqa: BLE001 - show SQL errors to the user
                st.error(f"Query failed: {e}")

# ------------------------------------------------------------------ method
with tabs[7]:
    st.subheader("How the numbers are built")
    st.markdown(f"""
**Sources combined:** claims (with its overwritten status field), status history, document checklist,
adjuster assignments, payment events.

**Cleaning:** claim ids, status labels, claim types, document names and adjuster ids are normalised; mixed ISO and
day-first dates are parsed; amounts with Rs./rupee signs are converted; duplicates, orphans, future-dated and
unknown-label events are removed. Each fix is counted in the Data-quality tab.

**Case-history reconstruction:** for every stage the best available evidence is used, in this order:
the status-history event, the claim's current status and last-updated time, then domain evidence
(first adjuster assignment, first document request, last required document received, first successful final
payment). Each entry stores its source, record id and confidence. If evidence contradicts the stage order, the
next-best evidence is used.

**Measures**
- *Stage duration* = time from entering a stage to entering the next evidenced stage (open stages run to {AS_OF:%d %b %Y}).
- *Total turnaround* = report date to closure, closed claims only.
- *Bottleneck score* = a stage's share of all elapsed claim-days (completed plus still-open time).
- *Incomplete case* = missing required document, history gap, status field that disagrees with history,
  no adjuster record, or no final payment record.
- *Backlog* = claims not yet closed in the rebuilt history; "past target" uses the editable stage targets.

**Limits:** if a closure event is missing and the status field is stale, a closed claim can look open. Stages with
no evidence at all cannot be timed; their time is excluded from per-stage statistics by default. Adjuster
comparisons use each claim's latest adjuster and ignore claim mix.

**Scope:** time and completeness only. Nothing here assesses whether a claim is eligible or payable.
""")
