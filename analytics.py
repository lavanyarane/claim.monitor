"""analytics.py - read the processed SQLite database and compute the summary measures.

Shared by the Streamlit app (app.py) and the static report (export_report.py).
All functions take plain DataFrames, so they are easy to test.
"""
import sqlite3

import numpy as np
import pandas as pd

from config import DB_PATH, DEFAULT_SLA_DAYS, STALLED_DAYS, WORK_STAGES

DT_COLS = {"reported_at", "stage_entered_at", "last_activity_at", "entered_at", "left_at", "requested_at",
           "received_at", "assigned_at", "released_at", "ts", "paid_at"}
BOOL_COLS = {"is_open", "docs_complete", "status_mismatch", "no_adjuster", "progressed_with_missing_docs",
             "no_final_payment_record", "is_incomplete", "stalled", "required", "adjusted"}
TABLES = ["case_summary", "stage_durations", "case_timeline", "fact_documents", "fact_assignments",
          "fact_payments", "fact_status_events", "dq_log", "meta"]
SLOW_DOC_DAYS = 14


def load_tables(db_path=DB_PATH):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = {}
    for name in TABLES:
        df = pd.read_sql_query(f"SELECT * FROM {name}", con)
        for c in df.columns:
            if c in DT_COLS:
                df[c] = pd.to_datetime(df[c])
            elif c in BOOL_COLS:
                df[c] = df[c].astype(bool)
        out[name] = df
    con.close()
    return out


def as_of(tables):
    m = tables["meta"].set_index("k")["v"]
    return pd.Timestamp(m["as_of"])


def filter_tables(t, types=None, regions=None, date_from=None, date_to=None):
    cs = t["case_summary"]
    mask = pd.Series(True, index=cs.index)
    if types:
        mask &= cs["claim_type"].isin(types)
    if regions:
        mask &= cs["region"].isin(regions)
    if date_from is not None:
        mask &= cs["reported_at"] >= pd.Timestamp(date_from)
    if date_to is not None:
        mask &= cs["reported_at"] < pd.Timestamp(date_to) + pd.Timedelta(days=1)
    cs = cs[mask].copy()
    ids = set(cs["claim_id"])
    return dict(
        cs=cs,
        sd=t["stage_durations"][t["stage_durations"]["claim_id"].isin(ids)].copy(),
        tl=t["case_timeline"][t["case_timeline"]["claim_id"].isin(ids)].copy(),
        docs=t["fact_documents"][t["fact_documents"]["claim_id"].isin(ids)].copy(),
    )


# ---------------------------------------------------------------- backlog
def build_backlog(cs, sla=None):
    """Open claims with the service-target check applied (targets are editable in the app)."""
    sla = sla or DEFAULT_SLA_DAYS
    bl = cs[cs["is_open"]].copy()
    bl["sla_days"] = bl["reconstructed_stage"].map(sla)
    bl["days_over_sla"] = (bl["days_in_stage"] - bl["sla_days"]).clip(lower=0)
    bl["sla_breached"] = bl["days_in_stage"] > bl["sla_days"]
    bl["stalled"] = bl["days_since_last_activity"] >= STALLED_DAYS
    return bl.sort_values(["days_over_sla", "days_in_stage"], ascending=False).reset_index(drop=True)


def aging_buckets(bl):
    edges = [-1, 7, 14, 30, 60, np.inf]
    labels = ["0-7 days", "8-14 days", "15-30 days", "31-60 days", "61+ days"]
    out = bl.copy()
    out["age_bucket"] = pd.cut(out["age_days"], edges, labels=labels)
    return out.groupby(["age_bucket", "reconstructed_stage"], observed=False).size().reset_index(name="claims")


# ---------------------------------------------------------------- stage measures
def stage_stats(sd, clean_only=True):
    """Duration statistics per stage plus each stage's share of all elapsed time (the bottleneck score)."""
    done = sd[~sd["is_open"]]
    if clean_only:
        done = done[done["absorbs_missing_stages"] == 0]
    open_ = sd[sd["is_open"]]
    rows = []
    for s in WORK_STAGES:
        d = done.loc[done["stage"] == s, "duration_days"]
        o = open_.loc[open_["stage"] == s, "duration_days"]
        rows.append(dict(
            stage=s, completed=len(d),
            median_days=d.median() if len(d) else np.nan, mean_days=d.mean() if len(d) else np.nan,
            p90_days=d.quantile(0.9) if len(d) else np.nan,
            open_now=len(o), open_median_days=o.median() if len(o) else np.nan,
            elapsed_days=d.sum() + o.sum(),
        ))
    st = pd.DataFrame(rows)
    total = st["elapsed_days"].sum()
    st["share_pct"] = 100 * st["elapsed_days"] / total if total else np.nan
    st["bottleneck_rank"] = st["share_pct"].rank(ascending=False, method="min").astype("Int64")
    return st


def type_stage_matrix(sd, clean_only=True):
    done = sd[~sd["is_open"]]
    if clean_only:
        done = done[done["absorbs_missing_stages"] == 0]
    m = done.pivot_table(index="claim_type", columns="stage", values="duration_days", aggfunc="median")
    return m.reindex(columns=[s for s in WORK_STAGES if s in m.columns])


def adjuster_stats(sd, cs, min_n=10, clean_only=True):
    a = sd[(sd["stage"] == "Assessment") & (~sd["is_open"])]
    if clean_only:
        a = a[a["absorbs_missing_stages"] == 0]
    a = a.merge(cs[["claim_id", "adjuster_id", "adjuster_name"]], on="claim_id")
    g = a.groupby(["adjuster_id", "adjuster_name"]).agg(
        claims=("duration_days", "size"), median_days=("duration_days", "median"),
        p90_days=("duration_days", lambda x: x.quantile(0.9))).reset_index()
    cur = cs[cs["is_open"] & (cs["reconstructed_stage"] == "Assessment")].groupby("adjuster_id").size()
    g["open_in_assessment"] = g["adjuster_id"].map(cur).fillna(0).astype(int)
    return g[g["claims"] >= min_n].sort_values("median_days", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- documents
def missing_docs_summary(docs, cs):
    open_ids = set(cs.loc[cs["is_open"], "claim_id"])
    m = docs[docs["required"] & (docs["status"] == "Pending") & docs["claim_id"].isin(open_ids)]
    return (m.groupby("doc_type").agg(documents=("doc_id", "size"), claims=("claim_id", "nunique"),
                                      median_wait_days=("wait_days", "median"))
            .reset_index().sort_values("documents", ascending=False))


def doc_receipt_times(docs):
    r = docs[docs["required"] & (docs["status"] == "Received") & docs["wait_days"].notna()]
    return (r.groupby("doc_type").agg(documents=("doc_id", "size"), median_days=("wait_days", "median"),
                                      p90_days=("wait_days", lambda x: x.quantile(0.9)))
            .reset_index().sort_values("median_days", ascending=False))


def turnaround_by_doc_speed(cs):
    c = cs[(~cs["is_open"]) & cs["total_turnaround_days"].notna() & cs["max_doc_wait_days"].notna()].copy()
    c["document_speed"] = np.where(c["max_doc_wait_days"] > SLOW_DOC_DAYS,
                                   f"Slowest required document took >{SLOW_DOC_DAYS} days",
                                   f"All required documents within {SLOW_DOC_DAYS} days")
    return c


# ---------------------------------------------------------------- headline numbers
def kpis(cs, bl):
    closed = cs[~cs["is_open"]]
    tt = closed["total_turnaround_days"].dropna()
    oldest = bl.sort_values("age_days", ascending=False).head(1)
    return dict(
        claims=len(cs), open=int(cs["is_open"].sum()), closed=len(closed),
        median_turnaround=tt.median() if len(tt) else np.nan,
        mean_turnaround=tt.mean() if len(tt) else np.nan,
        p90_turnaround=tt.quantile(0.9) if len(tt) else np.nan,
        pct_docs_missing=100 * (~cs["docs_complete"]).mean() if len(cs) else np.nan,
        pct_incomplete=100 * cs["is_incomplete"].mean() if len(cs) else np.nan,
        open_breached=int(bl["sla_breached"].sum()), open_stalled=int(bl["stalled"].sum()),
        median_open_age=bl["age_days"].median() if len(bl) else np.nan,
        oldest_id=oldest["claim_id"].iloc[0] if len(oldest) else None,
        oldest_age=oldest["age_days"].iloc[0] if len(oldest) else np.nan,
    )


def insights(cs, sd, docs, bl, clean_only=True):
    """Plain-language findings, every number computed from the data on screen."""
    out = []
    k = kpis(cs, bl)
    if k["closed"]:
        out.append(f"**Turnaround.** {k['closed']} closed claims took a median of {k['median_turnaround']:.0f} days "
                   f"from report to closure (mean {k['mean_turnaround']:.0f}, 90th percentile "
                   f"{k['p90_turnaround']:.0f}). The mean sits well above the median, so a minority of very "
                   f"slow claims drives the average.")
    st = stage_stats(sd, clean_only)
    if st["share_pct"].notna().any():
        top = st.sort_values("share_pct", ascending=False).iloc[0]
        out.append(f"**Bottleneck.** {top['stage']} uses the largest share of elapsed time "
                   f"({top['share_pct']:.0f}% of all claim-days across stages). Completed cases spent a median "
                   f"of {top['median_days']:.1f} days there (90th percentile {top['p90_days']:.0f}), and "
                   f"{int(top['open_now'])} of the {k['open']} open claims are sitting in it right now.")
    miss = missing_docs_summary(docs, cs)
    if len(miss):
        n_cl = int(bl[bl["n_docs_missing"] > 0].shape[0])
        long_wait = int((bl["n_docs_missing"].gt(0) & bl["days_in_stage"].gt(30) &
                         bl["reconstructed_stage"].eq("Document Collection")).sum())
        out.append(f"**Documentation.** {n_cl} open claims are missing at least one required document "
                   f"({int(miss['documents'].sum())} documents in total); {long_wait} of them have been in "
                   f"Document Collection for more than 30 days. The most frequently missing item is "
                   f"{miss.iloc[0]['doc_type']} ({int(miss.iloc[0]['documents'])} claims).")
    ts = turnaround_by_doc_speed(cs)
    if ts["document_speed"].nunique() == 2:
        g = ts.groupby("document_speed")["total_turnaround_days"].agg(["median", "size"])
        slow = g.filter(like=">", axis=0).iloc[0]
        fast = g.filter(like="within", axis=0).iloc[0]
        out.append(f"**Documents vs. turnaround.** Closed claims where a required document took more than "
                   f"{SLOW_DOC_DAYS} days to arrive ({int(slow['size'])} claims) had a median turnaround of "
                   f"{slow['median']:.0f} days, against {fast['median']:.0f} days for the other "
                   f"{int(fast['size'])} claims. This is an association in observational data, not proof "
                   f"that documents alone caused the delay.")
    if len(bl):
        out.append(f"**Backlog.** {k['open']} claims are open (median age {k['median_open_age']:.0f} days); "
                   f"{k['open_breached']} are past the target for their current stage and {k['open_stalled']} "
                   f"have had no recorded activity for {STALLED_DAYS}+ days. Oldest: {k['oldest_id']} "
                   f"({k['oldest_age']:.0f} days).")
    closed = cs[~cs["is_open"]]
    if closed["claim_type"].nunique() > 1:
        bt = closed.groupby("claim_type")["total_turnaround_days"].median().sort_values(ascending=False)
        out.append(f"**Claim type.** {bt.index[0]} claims are slowest (median {bt.iloc[0]:.0f} days) and "
                   f"{bt.index[-1]} fastest ({bt.iloc[-1]:.0f} days).")
    adj = adjuster_stats(sd, cs, clean_only=clean_only)
    if len(adj) >= 3:
        med = adj["median_days"].median()
        top = adj.iloc[0]
        if top["median_days"] > 1.25 * med:
            out.append(f"**Adjuster workload.** {top['adjuster_name']} ({top['adjuster_id']}) has the longest "
                       f"median assessment time ({top['median_days']:.1f} days vs {med:.1f} across adjusters) and "
                       f"{top['open_in_assessment']} open claims in assessment. Worth a workload review before "
                       f"drawing conclusions about individual performance.")
    out.append(f"**Record quality.** {k['pct_incomplete']:.0f}% of cases carry at least one completeness flag "
               f"(missing documents, a history gap, a status field that disagrees with the history, or no "
               f"adjuster/payment record). {int(cs['n_inferred_stages'].gt(0).sum())} cases needed at least "
               f"one stage time rebuilt from another source; each is labelled with its source and confidence.")
    out.append("**Scope.** This tool measures time and completeness only. It does not assess whether any claim "
               "is eligible or payable.")
    return out
