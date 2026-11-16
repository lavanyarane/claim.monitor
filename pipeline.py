"""pipeline.py - clean the raw extracts, reconstruct every case history, and store the result in SQLite.

    python pipeline.py [--as-of 2026-11-12] [--db claims.db]

Steps
  1. Load the five raw CSVs (as text, nothing is trusted yet) and keep an untouched copy in raw_* tables.
  2. Clean: normalise ids / labels / dates / money, drop duplicates, orphans, future-dated and unknown events.
     Every fix is counted in the dq_log table.
  3. Reconstruct a stage-by-stage timeline per claim from ALL sources (status history, the overwritten
     claim status, document dates, adjuster assignments, payments). Each stage entry records which record
     it came from, so every number can be traced back.
  4. Derive stage durations, turnaround, document completeness, incomplete-case flags and the backlog.

The pipeline only measures time and completeness. It never decides whether a claim is eligible or payable.
"""
import argparse
import re
import sqlite3

import numpy as np
import pandas as pd

from config import AS_OF, DB_PATH, RAW_DIR, STAGE_ORDER, STAGES, STALLED_DAYS, TRUTH_DIR

RAW_FILES = ["claims", "status_history", "documents", "adjusters", "payments"]

STAGE_MAP = {
    "fnol": "Reported", "reported": "Reported", "first notice": "Reported",
    "registered": "Registered", "reg": "Registered",
    "docs pending": "Document Collection", "document collection": "Document Collection",
    "documents pending": "Document Collection", "doc collection": "Document Collection",
    "under assessment": "Assessment", "assessment": "Assessment", "assessing": "Assessment",
    "with adjuster": "Assessment",
    "settlement": "Settlement Processing", "settlement processing": "Settlement Processing",
    "paid": "Payment Issued", "payment issued": "Payment Issued", "payment released": "Payment Issued",
    "closed": "Closed", "claim closed": "Closed",
    "on hold": "On Hold",
}
TYPE_MAP = {"motor": "Motor", "motor vehicle": "Motor", "auto": "Motor",
            "health": "Health", "mediclaim": "Health", "medical": "Health",
            "property": "Property", "prop.": "Property", "prop": "Property", "home": "Property",
            "travel": "Travel"}
DOC_KEYS = {"claimform": "Claim Form", "policycopy": "Policy Copy", "idproof": "ID Proof",
            "repairestimate": "Repair Estimate", "incidentreport": "Incident Report",
            "hospitalbills": "Hospital Bills", "dischargesummary": "Discharge Summary",
            "damageestimate": "Damage Estimate", "photographs": "Photographs",
            "traveltickets": "Travel Tickets", "supportingletter": "Supporting Letter"}
DOC_STATUS = {"received": "Received", "rcvd": "Received", "pending": "Pending", "outstanding": "Pending",
              "waived": "Waived", "na": "Waived", "n/a": "Waived"}
PAY_TYPE = {"final": "Final", "settlement": "Final", "interim": "Interim", "part payment": "Interim"}
PAY_STATUS = {"processed": "Processed", "paid": "Processed", "success": "Processed",
              "failed": "Failed", "declined": "Failed", "reversed": "Reversed"}
NEAR_DUP_MINUTES = 10


# --------------------------------------------------------------------------- helpers
class DQ:
    """Collects one row per data-quality issue: what was wrong, how many records, what we did."""

    def __init__(self):
        self.rows = []

    def add(self, table, issue, count, action, examples=None):
        count = int(count)
        if count > 0:
            ex = ", ".join(str(e) for e in list(examples)[:4]) if examples is not None else ""
            self.rows.append(dict(table=table, issue=issue, records=count, action=action, examples=ex))

    def frame(self):
        return pd.DataFrame(self.rows, columns=["table", "issue", "records", "action", "examples"])


def blank_to_na(s):
    s = s.astype("string").str.strip()
    return s.mask(s.str.lower().isin(["", "na", "n/a", "null", "none", "-", "nan"]) & (s.str.lower() != "na")
                  | (s == ""))


def norm_id(s):
    return s.astype("string").str.replace(r"\s+", "", regex=True).str.upper()


def parse_dt(s):
    """Parse ISO ('2026-03-14 10:22:00') and day-first ('14/03/2026 10:22') timestamps."""
    s = s.astype("string").str.strip()
    s = s.mask(s == "")
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    dmy = s.str.contains("/", na=False)
    a = pd.to_datetime(s.where(iso), format="ISO8601", errors="coerce")
    b = pd.to_datetime(s.where(dmy), format="mixed", dayfirst=True, errors="coerce")
    out = a.fillna(b)
    return out.astype("datetime64[ns]"), iso.sum(), dmy.sum(), (s.notna() & out.isna()).sum()


def parse_money(s):
    t = s.astype("string").str.replace(r"(?i)(rs\.?|inr|₹)", "", regex=True)
    t = t.str.replace(r"[,\s]", "", regex=True)
    return pd.to_numeric(t.mask(t == ""), errors="coerce")


def key(s):
    return s.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def days(delta):
    return delta.total_seconds() / 86400.0


# --------------------------------------------------------------------------- cleaning
def clean_claims(raw, dq, as_of):
    df = raw.copy()
    raw_id = df["claim_id"]
    df["claim_id"] = norm_id(df["claim_id"])
    dq.add("claims", "Claim id formatting (case / stray spaces)", (raw_id != df["claim_id"]).sum(),
           "Upper-cased and trimmed so records join across files", raw_id[raw_id != df["claim_id"]])

    t = key(df["claim_type"])
    df["claim_type"] = t.map(TYPE_MAP)
    dq.add("claims", "Claim type spelt several ways", (df["claim_type"].astype("string") != raw["claim_type"]).sum(),
           "Mapped to Motor / Health / Property / Travel")
    dq.add("claims", "Unknown claim type", df["claim_type"].isna().sum(), "Left blank")

    amt_txt = df["claim_amount"].astype("string").str.strip()
    df["claim_amount"] = parse_money(df["claim_amount"])
    dq.add("claims", "Amount stored as text with symbols/commas (Rs., rupee sign, 1,00,000)",
           amt_txt.str.contains(r"[^\d.]", regex=True, na=False).sum(), "Converted to numbers")
    dq.add("claims", "Missing claim amount", df["claim_amount"].isna().sum(), "Left blank (not used in timings)")

    reg = df["region"].astype("string").str.strip().str.title()
    dq.add("claims", "Region missing", (reg.isna() | (reg == "")).sum(), "Set to 'Unknown'")
    dq.add("claims", "Region casing / spacing differs", (df["region"].astype("string") != reg).sum() -
           (reg.isna() | (reg == "")).sum(), "Title-cased and trimmed")
    df["region"] = reg.mask(reg == "").fillna("Unknown")

    for col in ["reported_date", "last_updated"]:
        df[col], n_iso, n_dmy, n_bad = parse_dt(df[col])
        dq.add("claims", f"{col}: day-first dd/mm/yyyy format mixed with ISO", n_dmy, "Parsed as day-first")
        dq.add("claims", f"{col}: missing or unreadable", df[col].isna().sum(),
               "Left blank; other sources are used where possible")

    status = key(df["current_status"]).map(STAGE_MAP)
    dq.add("claims", "Status label variants (e.g. 'Paid', 'Docs Pending', 'CLOSED')",
           (df["current_status"].astype("string").str.strip() != status).sum(),
           "Mapped to the 7 canonical stages")
    df["claims_status"] = status

    # duplicates: keep the most recently updated row for each claim
    before = len(df)
    df = df.sort_values("last_updated", ascending=False, na_position="last").drop_duplicates("claim_id")
    dq.add("claims", "Duplicate claim rows (stale copies of the record)", before - len(df),
           "Kept the row with the latest last_updated")
    df = df[df["claim_id"].notna()].drop(columns=["current_status"])
    return df.sort_values("claim_id").reset_index(drop=True)


def clean_events(raw, claim_ids, dq, as_of):
    df = raw.copy()
    df["claim_id"] = norm_id(df["claim_id"])
    df["raw_status"] = raw["status"]
    df["stage"] = key(df["status"]).map(STAGE_MAP)
    df["ts"], n_iso, n_dmy, n_bad = parse_dt(df["changed_at"])
    dq.add("status_history", "Timestamp in day-first dd/mm/yyyy format", n_dmy, "Parsed as day-first")
    dq.add("status_history", "Unreadable timestamp", n_bad, "Event excluded", df.loc[df["ts"].isna(), "event_id"])
    df = df[df["ts"].notna()]

    unk = df["stage"].isna()
    dq.add("status_history", "Status label not part of the workflow (e.g. 'Escalated to Supervisor')", unk.sum(),
           "Excluded from stage timing (kept in raw_status_history)", df.loc[unk, "event_id"])
    df = df[~unk]

    orph = ~df["claim_id"].isin(claim_ids)
    dq.add("status_history", "Orphan events (claim id not in claims)", orph.sum(), "Excluded",
           df.loc[orph, "claim_id"].unique())
    df = df[~orph]

    fut = df["ts"] > as_of
    dq.add("status_history", "Future-dated events (after the snapshot date, e.g. year typo)", fut.sum(),
           "Excluded; stage time is taken from other evidence", df.loc[fut, "event_id"])
    df = df[~fut]

    before = len(df)
    df = df.drop_duplicates("event_id")
    dq.add("status_history", "Exact duplicate events (same event id loaded twice)", before - len(df), "Dropped")

    df = df.sort_values(["claim_id", "stage", "ts"])
    gap = df.groupby(["claim_id", "stage"])["ts"].diff()
    near = gap.notna() & (gap < pd.Timedelta(minutes=NEAR_DUP_MINUTES))
    dq.add("status_history", f"Near-duplicate events (same claim and stage within {NEAR_DUP_MINUTES} min)",
           near.sum(), "Kept the first", df.loc[near, "event_id"])
    df = df[~near]
    return df[["event_id", "claim_id", "stage", "ts", "raw_status", "changed_by"]].reset_index(drop=True)


def clean_docs(raw, claim_ids, dq, as_of):
    df = raw.copy()
    df["claim_id"] = norm_id(df["claim_id"])
    k = df["doc_type"].astype("string").str.lower().str.replace(r"[^a-z]", "", regex=True)
    df["doc_type"] = k.map(DOC_KEYS)
    dq.add("documents", "Document type spelt several ways (case, hyphens, underscores)",
           (df["doc_type"] != raw["doc_type"]).sum(), "Mapped to one name per document")
    dq.add("documents", "Unknown document type", df["doc_type"].isna().sum(), "Excluded")
    df = df[df["doc_type"].notna()]

    df["required"] = key(df["required"]).isin(["y", "yes", "true", "1"])
    df["requested_at"], _, n_dmy1, _ = parse_dt(df["requested_date"])
    df["received_at"], _, n_dmy2, _ = parse_dt(df["received_date"])
    dq.add("documents", "Dates in day-first dd/mm/yyyy format", n_dmy1 + n_dmy2, "Parsed as day-first")
    df["status_raw"] = key(df["doc_status"]).map(DOC_STATUS)

    orph = ~df["claim_id"].isin(claim_ids)
    dq.add("documents", "Orphan documents (claim id not in claims)", orph.sum(), "Excluded")
    df = df[~orph]
    before = len(df)
    df = df.drop_duplicates("doc_id")
    dq.add("documents", "Duplicate document rows", before - len(df), "Dropped")

    status = df["status_raw"].copy()
    has_date = df["received_at"].notna()
    contra = (status != "Received") & has_date & (status != "Waived")
    dq.add("documents", "Status says Pending but a received date exists", contra.sum(),
           "Treated as Received (the date is evidence)", df.loc[contra, "doc_id"])
    status[has_date & (status == "Pending")] = "Received"
    nodate = (status == "Received") & ~has_date
    dq.add("documents", "Status Received but no received date", nodate.sum(),
           "Counted as received; excluded from timing", df.loc[nodate, "doc_id"])
    df["status"] = status
    early = has_date & (df["received_at"] < df["requested_at"])
    dq.add("documents", "Received before it was requested (submitted with claim form)", early.sum(),
           "Kept; waiting time floored at 0", df.loc[early, "doc_id"])
    df["wait_days"] = np.where(
        df["status"].eq("Received") & has_date,
        ((df["received_at"] - df["requested_at"]).dt.total_seconds() / 86400).clip(lower=0),
        np.where(df["status"].eq("Pending"), (as_of - df["requested_at"]).dt.total_seconds() / 86400, np.nan))
    df["wait_days"] = df["wait_days"].astype(float)
    return df[["doc_id", "claim_id", "doc_type", "required", "requested_at", "received_at", "status",
               "wait_days"]].reset_index(drop=True)


def clean_assign(raw, claim_ids, dq):
    df = raw.copy()
    df["claim_id"] = norm_id(df["claim_id"])
    num = df["adjuster_id"].astype("string").str.extract(r"(\d+)")[0].astype(float)
    new_id = "ADJ-" + num.astype("Int64").astype("string").str.zfill(2)
    dq.add("adjusters", "Adjuster id formatting (adj07, ADJ 07, adj-07)", (new_id != raw["adjuster_id"]).sum(),
           "Normalised to ADJ-NN")
    df["adjuster_id"] = new_id
    df["adjuster_name"] = df["adjuster_name"].astype("string").str.strip().str.title()
    df["assigned_at"], _, n_dmy1, _ = parse_dt(df["assigned_at"])
    df["released_at"], _, n_dmy2, _ = parse_dt(df["released_at"])
    dq.add("adjusters", "Dates in day-first dd/mm/yyyy format", n_dmy1 + n_dmy2, "Parsed as day-first")
    orph = ~df["claim_id"].isin(claim_ids)
    dq.add("adjusters", "Orphan assignments", orph.sum(), "Excluded")
    df = df[~orph]
    before = len(df)
    df = df.drop_duplicates("assignment_id")
    dq.add("adjusters", "Duplicate assignment rows", before - len(df), "Dropped")
    return df[["assignment_id", "claim_id", "adjuster_id", "adjuster_name", "assigned_at",
               "released_at"]].reset_index(drop=True)


def clean_pay(raw, claim_ids, dq):
    df = raw.copy()
    df["claim_id"] = norm_id(df["claim_id"])
    df["payment_type"] = key(df["payment_type"]).map(PAY_TYPE)
    df["status"] = key(df["payment_status"]).map(PAY_STATUS)
    dq.add("payments", "Payment type / status spelt several ways",
           ((df["payment_type"].astype("string") != raw["payment_type"].str.strip()) |
            (df["status"].astype("string") != raw["payment_status"].str.strip())).sum(),
           "Mapped to Final/Interim and Processed/Failed/Reversed")
    df["amount"] = parse_money(df["amount"])
    df["paid_at"], _, n_dmy, _ = parse_dt(df["paid_at"])
    dq.add("payments", "Dates in day-first dd/mm/yyyy format", n_dmy, "Parsed as day-first")
    orph = ~df["claim_id"].isin(claim_ids)
    dq.add("payments", "Orphan payments", orph.sum(), "Excluded")
    df = df[~orph]
    before = len(df)
    df = df.drop_duplicates("payment_id")
    dq.add("payments", "Duplicate payment rows", before - len(df), "Dropped")
    dq.add("payments", "Failed payment attempts (kept as delay evidence)", (df["status"] == "Failed").sum(),
           "Counted per claim")
    return df[["payment_id", "claim_id", "payment_type", "amount", "paid_at", "status"]].reset_index(drop=True)


# --------------------------------------------------------------------------- reconstruction
def reconstruct(claims, events, docs, assigns, pays, dq, as_of):
    """Build the stage timeline of every claim from all available evidence."""
    ev_by = {k: g for k, g in events.groupby("claim_id")}
    doc_by = {k: g for k, g in docs.groupby("claim_id")}
    asg_by = {k: g for k, g in assigns.groupby("claim_id")}
    pay_by = {k: g for k, g in pays.groupby("claim_id")}
    empty = lambda df: df.iloc[0:0]

    timeline, summary = [], []
    counters = dict(out_of_order=0, status_hist=0, inferred=0)

    for c in claims.itertuples(index=False):
        cid = c.claim_id
        ev = ev_by.get(cid, empty(events))
        dc = doc_by.get(cid, empty(docs))
        ag = asg_by.get(cid, empty(assigns))
        py = pay_by.get(cid, empty(pays))
        req_docs = dc[dc["required"]]
        cur_stage = c.claims_status

        def cands(stage):
            """Evidence for entering `stage`, best first: (timestamp, source, record id, confidence)."""
            out = []
            e = ev[ev["stage"] == stage]
            if len(e):
                r = e.sort_values("ts").iloc[0]
                out.append((r["ts"], "status_history", r["event_id"], "high"))
            if stage == "Reported" and pd.notna(c.reported_date):
                out.append((c.reported_date, "claims.reported_date", cid, "high"))
            if stage == cur_stage and pd.notna(c.last_updated):
                out.append((c.last_updated, "claims.last_updated", cid, "medium"))
            if stage == "Registered" and len(ag):
                r = ag.sort_values("assigned_at").iloc[0]
                out.append((r["assigned_at"], "adjusters.assigned_at", r["assignment_id"], "medium"))
            if stage == "Document Collection" and len(dc) and dc["requested_at"].notna().any():
                r = dc.sort_values("requested_at").iloc[0]
                out.append((r["requested_at"], "documents.requested_at", r["doc_id"], "medium"))
            if stage == "Assessment" and len(req_docs):
                need = req_docs[req_docs["status"] != "Waived"]
                if len(need) and (need["status"] == "Received").all() and need["received_at"].notna().all():
                    r = need.sort_values("received_at").iloc[-1]
                    out.append((r["received_at"], "documents.last_required_received", r["doc_id"], "medium"))
            if stage == "Payment Issued":
                ok = py[(py["status"] == "Processed") & (py["payment_type"] == "Final")]
                if len(ok):
                    r = ok.sort_values("paid_at").iloc[0]
                    out.append((r["paid_at"], "payments.paid_at", r["payment_id"], "high"))
            return out

        entries, prev_ts = {}, None
        for stage in STAGES:
            options = cands(stage)
            chosen = None
            for ts, src, rid, conf in options:
                if prev_ts is None or ts >= prev_ts:
                    chosen = (ts, src, rid, conf, False, "")
                    break
            if chosen is None and options:                       # every candidate breaks the order
                ts, src, rid, conf = options[0]
                chosen = (prev_ts, src, rid, "low", True, f"time moved forward from {ts:%Y-%m-%d %H:%M}")
            if chosen is None:
                continue
            first = options[0]
            if options and chosen[1] != first[1] and not chosen[4]:
                counters["out_of_order"] += 1                   # best evidence conflicted, used the next one
                chosen = chosen[:5] + (f"ignored out-of-order {first[1]} ({first[2]})",)
            entries[stage] = chosen
            prev_ts = chosen[0]

        for stage, (ts, src, rid, conf, adj, note) in entries.items():
            timeline.append(dict(claim_id=cid, stage=stage, stage_order=STAGE_ORDER[stage], entered_at=ts,
                                 source=src, source_record_id=rid, confidence=conf,
                                 adjusted=adj, note=note))

        recon_stage = max(entries, key=lambda s: STAGE_ORDER[s]) if entries else None
        stage_at = entries[recon_stage][0] if recon_stage else pd.NaT
        reported_at = entries["Reported"][0] if "Reported" in entries else (
            entries[min(entries, key=lambda s: STAGE_ORDER[s])][0] if entries else pd.NaT)
        closed_at = entries["Closed"][0] if "Closed" in entries else pd.NaT
        pay_at = entries["Payment Issued"][0] if "Payment Issued" in entries else pd.NaT

        # document completeness
        need = req_docs[req_docs["status"] != "Waived"]
        missing = need[need["status"] == "Pending"]
        n_req = len(need)
        docs_complete = len(missing) == 0
        wait = req_docs["wait_days"].dropna()

        # adjuster
        ag_sorted = ag.sort_values("assigned_at")
        cur_adj = ag_sorted.iloc[-1] if len(ag_sorted) else None

        # payments
        n_failed = int((py["status"] == "Failed").sum())
        has_final = bool(((py["status"] == "Processed") & (py["payment_type"] == "Final")).any())

        # unevidenced stages between first and last evidenced stage
        orders = sorted(STAGE_ORDER[s] for s in entries)
        gaps = [STAGES[o] for o in range(orders[0], orders[-1] + 1) if STAGES[o] not in entries] if orders else []
        inferred = [s for s, v in entries.items() if v[1] not in ("status_history", "claims.reported_date")]
        counters["inferred"] += len(inferred)

        is_open = recon_stage != "Closed"
        age = days(as_of - reported_at) if pd.notna(reported_at) else np.nan
        last_activity = max([t for t in [*[v[0] for v in entries.values()],
                                         *dc["received_at"].dropna().tolist(),
                                         *py["paid_at"].dropna().tolist()] if t <= as_of], default=pd.NaT)

        reasons = []
        if not docs_complete:
            reasons.append(f"{len(missing)} required document(s) missing")
        if gaps:
            reasons.append("history gap: " + ", ".join(gaps))
        if len(ag) == 0 and recon_stage and STAGE_ORDER[recon_stage] >= STAGE_ORDER["Document Collection"]:
            reasons.append("no adjuster assignment on record")
        if recon_stage in ("Payment Issued", "Closed") and not has_final:
            reasons.append("no final payment record")
        if not docs_complete and recon_stage and STAGE_ORDER[recon_stage] >= STAGE_ORDER["Assessment"]:
            reasons.append("progressed with documents missing")
        mismatch = bool(cur_stage and recon_stage and cur_stage != recon_stage)
        if mismatch:
            reasons.append(f"status field says '{cur_stage}', history shows '{recon_stage}'")

        # what is the open claim waiting on? (workflow wording only - no eligibility judgement)
        waiting = None
        if is_open:
            if recon_stage in ("Reported", "Registered"):
                waiting = "Awaiting document request"
            elif recon_stage == "Document Collection":
                waiting = "Awaiting documents" if not docs_complete else "Documents complete - awaiting assessment"
            elif recon_stage == "Assessment":
                waiting = "With adjuster" if cur_adj is not None else "No adjuster on record"
            elif recon_stage == "Settlement Processing":
                waiting = "Payment attempt failed - retry" if n_failed and not has_final else "In settlement"
            elif recon_stage == "Payment Issued":
                waiting = "Awaiting closure"

        src_rec = entries[recon_stage] if recon_stage else None
        summary.append(dict(
            claim_id=cid, claim_type=c.claim_type, region=c.region, claim_amount=c.claim_amount,
            reported_at=reported_at, claims_status=cur_stage, reconstructed_stage=recon_stage,
            is_open=is_open, stage_entered_at=stage_at,
            days_in_stage=days(as_of - stage_at) if (is_open and pd.notna(stage_at)) else np.nan,
            age_days=age if is_open else np.nan,
            total_turnaround_days=days(closed_at - reported_at) if pd.notna(closed_at) and pd.notna(reported_at)
            else np.nan,
            turnaround_to_payment_days=days(pay_at - reported_at) if pd.notna(pay_at) and pd.notna(reported_at)
            else np.nan,
            n_required_docs=n_req, n_docs_missing=len(missing),
            missing_docs=", ".join(sorted(missing["doc_type"])), docs_complete=docs_complete,
            max_doc_wait_days=float(wait.max()) if len(wait) else np.nan,
            adjuster_id=cur_adj["adjuster_id"] if cur_adj is not None else None,
            adjuster_name=cur_adj["adjuster_name"] if cur_adj is not None else None,
            n_adjuster_changes=max(len(ag) - 1, 0), n_failed_payments=n_failed,
            hold_events=int((ev["stage"] == "On Hold").sum()),
            n_inferred_stages=len(inferred), n_gap_stages=len(gaps), gap_stages=", ".join(gaps),
            status_mismatch=mismatch, no_adjuster=len(ag) == 0,
            progressed_with_missing_docs=(not docs_complete) and bool(recon_stage) and
            STAGE_ORDER[recon_stage] >= STAGE_ORDER["Assessment"],
            no_final_payment_record=recon_stage in ("Payment Issued", "Closed") and not has_final,
            is_incomplete=len(reasons) > 0, incomplete_reasons="; ".join(reasons),
            last_activity_at=last_activity,
            days_since_last_activity=days(as_of - last_activity) if pd.notna(last_activity) and is_open
            else np.nan,
            waiting_on=waiting,
            stage_entry_source=src_rec[1] if src_rec else None,
            stage_entry_record=src_rec[2] if src_rec else None,
            stage_entry_confidence=src_rec[3] if src_rec else None,
        ))

    timeline = pd.DataFrame(timeline)
    summary = pd.DataFrame(summary)
    dq.add("reconstruction", "Stage entries rebuilt from other sources (event missing from status history)",
           (~timeline["source"].isin(["status_history", "claims.reported_date"])).sum(),
           "Inferred from claim status, documents, adjuster or payment records; confidence is stored")
    dq.add("reconstruction", "Best evidence conflicted with the stage order - next-best evidence used",
           counters["out_of_order"], "Recorded in timeline.note")
    dq.add("reconstruction", "Stage entry moved forward to keep order (no consistent evidence)",
           timeline["adjusted"].sum(), "Flagged adjusted=1 in timeline")
    dq.add("reconstruction", "Stages with no evidence at all between known stages", summary["n_gap_stages"].sum(),
           "Their time is excluded from per-stage statistics by default")
    dq.add("reconstruction", "Claim status field disagrees with the rebuilt history", summary["status_mismatch"].sum(),
           "Rebuilt history is used; claim flagged incomplete")
    return timeline, summary


def stage_durations(timeline, summary, as_of):
    rows = []
    ctype = summary.set_index("claim_id")["claim_type"]
    for cid, g in timeline.sort_values(["claim_id", "stage_order"]).groupby("claim_id"):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            if r["stage"] == "Closed":
                continue
            if i + 1 < len(g):
                left, is_open = g.at[i + 1, "entered_at"], False
                absorbed = int(g.at[i + 1, "stage_order"] - r["stage_order"] - 1)
            else:
                left, is_open, absorbed = as_of, True, 0
            rows.append(dict(claim_id=cid, claim_type=ctype.get(cid), stage=r["stage"],
                             stage_order=r["stage_order"], entered_at=r["entered_at"], left_at=left,
                             duration_days=days(left - r["entered_at"]), is_open=is_open,
                             absorbs_missing_stages=absorbed, entry_source=r["source"],
                             entry_confidence=r["confidence"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- orchestration
def run(db_path=DB_PATH, as_of=AS_OF, raw_dir=RAW_DIR):
    dq = DQ()
    raw = {n: pd.read_csv(raw_dir / f"{n}.csv", dtype=str, keep_default_na=False) for n in RAW_FILES}

    claims = clean_claims(raw["claims"], dq, as_of)
    ids = set(claims["claim_id"])
    events = clean_events(raw["status_history"], ids, dq, as_of)
    docs = clean_docs(raw["documents"], ids, dq, as_of)
    assigns = clean_assign(raw["adjusters"], ids, dq)
    pays = clean_pay(raw["payments"], ids, dq)

    timeline, summary = reconstruct(claims, events, docs, assigns, pays, dq, as_of)
    durations = stage_durations(timeline, summary, as_of)
    backlog = summary[summary["is_open"]].copy()
    backlog["stalled"] = backlog["days_since_last_activity"] >= STALLED_DAYS
    backlog = backlog.sort_values(["days_in_stage"], ascending=False)
    doc_detail = docs.merge(summary[["claim_id", "claim_type", "is_open"]], on="claim_id", how="left")

    con = sqlite3.connect(db_path)
    for name, df in raw.items():                                  # untouched originals + join key
        r = df.copy()
        r.insert(0, "_row", np.arange(2, len(r) + 2))             # line number in the CSV file
        r.insert(1, "claim_key", norm_id(r["claim_id"]))
        r.to_sql(f"raw_{name}", con, if_exists="replace", index=False)
    tables = dict(dim_claims=claims, fact_status_events=events, fact_documents=doc_detail,
                  fact_assignments=assigns, fact_payments=pays, case_timeline=timeline,
                  stage_durations=durations, case_summary=summary, backlog_report=backlog,
                  dq_log=dq.frame())
    for name, df in tables.items():
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
            elif pd.api.types.is_bool_dtype(out[col]):
                out[col] = out[col].astype(int)
        out.to_sql(name, con, if_exists="replace", index=False)
    meta = pd.DataFrame([dict(k="as_of", v=as_of.strftime("%Y-%m-%d %H:%M:%S")),
                         dict(k="built_at", v=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")),
                         dict(k="claims", v=str(len(claims)))])
    meta.to_sql("meta", con, if_exists="replace", index=False)
    for t, c in [("case_summary", "claim_id"), ("case_timeline", "claim_id"), ("stage_durations", "claim_id"),
                 ("fact_documents", "claim_id"), ("fact_status_events", "claim_id"),
                 ("raw_status_history", "claim_key"), ("raw_documents", "claim_key"),
                 ("raw_adjusters", "claim_key"), ("raw_payments", "claim_key"), ("raw_claims", "claim_key")]:
        con.execute(f"CREATE INDEX IF NOT EXISTS ix_{t}_{c} ON {t}({c})")
    con.commit()
    con.close()
    return dict(summary=summary, timeline=timeline, durations=durations, dq=dq.frame())


def self_test(timeline, summary):
    """Compare the rebuilt timeline with the generator's ground truth (only possible on synthetic data)."""
    p = TRUTH_DIR / "stage_entries_truth.csv"
    if not p.exists():
        return
    truth = pd.read_csv(p, parse_dates=["entered_at"])
    m = truth.merge(timeline, on=["claim_id", "stage"], how="left", suffixes=("_true", "_rebuilt"))
    m["err_days"] = (m["entered_at_rebuilt"] - m["entered_at_true"]).dt.total_seconds().abs() / 86400
    found = m["entered_at_rebuilt"].notna()
    print("\nSelf-test against ground truth (synthetic data only)")
    print(f"  true stage entries recovered: {found.mean():.1%}")
    print(f"  recovered within 1 day:       {(m.loc[found, 'err_days'] <= 1).mean():.1%}")
    by = m[found].groupby("source")["err_days"].agg(n="size", within_1d=lambda s: (s <= 1).mean())
    print(by.round(3).to_string())
    t = truth[truth["stage"].isin(["Reported", "Closed"])].pivot(index="claim_id", columns="stage",
                                                                values="entered_at").dropna()
    t["true_days"] = (t["Closed"] - t["Reported"]).dt.total_seconds() / 86400
    j = summary.set_index("claim_id").join(t["true_days"], how="inner")
    j = j[j["total_turnaround_days"].notna()]
    print(f"  total turnaround within 1 day of truth: {((j['total_turnaround_days'] - j['true_days']).abs() <= 1).mean():.1%}"
          f"  (n={len(j)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", default=str(AS_OF), help="snapshot date used to age open claims")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()
    res = run(db_path=args.db, as_of=pd.Timestamp(args.as_of))
    s = res["summary"]
    print(f"Built {args.db}: {len(s)} claims, {int(s['is_open'].sum())} open, {int((~s['is_open']).sum())} closed")
    print("\nData-quality log")
    print(res["dq"].to_string(index=False, max_colwidth=70))
    self_test(res["timeline"], s)
