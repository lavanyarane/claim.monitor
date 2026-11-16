"""generate_data.py - build a synthetic, deliberately messy insurance-claims dataset.

Writes five CSV files to data/raw/ (the "source system extracts"):
    claims.csv, status_history.csv, documents.csv, adjusters.csv, payments.csv
and a ground-truth file to data/truth/ that is ONLY used to self-test the pipeline.

Everything is synthetic. Run:   python generate_data.py [--n 600] [--seed 42]
"""
import argparse

import numpy as np
import pandas as pd

from config import AS_OF, RAW_DIR, STAGES, TRUTH_DIR

TYPE_CFG = {  # median days per stage; lognormal noise is applied on top
    "Motor":    dict(w=0.38, amt=(55_000, 0.8),  reg=0.4, docs=3.5, assess=5.0,  settle=3.0, close=1.5),
    "Health":   dict(w=0.30, amt=(80_000, 1.0),  reg=0.6, docs=6.0, assess=8.0,  settle=4.0, close=2.0),
    "Property": dict(w=0.18, amt=(250_000, 1.0), reg=0.8, docs=7.0, assess=14.0, settle=5.0, close=2.0),
    "Travel":   dict(w=0.14, amt=(30_000, 0.8),  reg=0.5, docs=4.0, assess=4.0,  settle=2.5, close=1.0),
}
REQUIRED_DOCS = {
    "Motor":    ["Claim Form", "Policy Copy", "ID Proof", "Repair Estimate", "Incident Report"],
    "Health":   ["Claim Form", "Policy Copy", "ID Proof", "Hospital Bills", "Discharge Summary"],
    "Property": ["Claim Form", "Policy Copy", "ID Proof", "Damage Estimate", "Photographs"],
    "Travel":   ["Claim Form", "Policy Copy", "ID Proof", "Travel Tickets", "Incident Report"],
}
OPTIONAL_DOC = "Supporting Letter"
REGIONS = ["Pune", "Mumbai", "Nagpur", "Nashik", "Aurangabad", "Kolhapur", "Thane"]
REGION_W = [0.30, 0.22, 0.12, 0.10, 0.08, 0.08, 0.10]
CHANNELS = ["App", "Agent", "Branch", "Call Centre"]
ADJUSTERS = {
    "ADJ-01": "Priya Nair", "ADJ-02": "Rahul Deshmukh", "ADJ-03": "Sneha Kulkarni",
    "ADJ-04": "Amit Joshi", "ADJ-05": "Neha Patil", "ADJ-06": "Vikram Rao",
    "ADJ-07": "Anjali Mehta", "ADJ-08": "Rohan Shah", "ADJ-09": "Kavita Iyer",
    "ADJ-10": "Suresh Pawar",
}
ADJ_WEIGHT = {a: (2.0 if a == "ADJ-07" else 1.0) for a in ADJUSTERS}      # ADJ-07 carries extra load
ADJ_SLOWDOWN = {a: 1.0 for a in ADJUSTERS} | {"ADJ-07": 1.7, "ADJ-03": 1.2}  # ...and is slower

STATUS_LABELS = {
    "Reported": ["FNOL", "Reported", "Reported", "First Notice"],
    "Registered": ["Registered", "registered", "REGISTERED", "Reg"],
    "Document Collection": ["Docs Pending", "Document Collection", "Documents Pending", "doc collection"],
    "Assessment": ["Under Assessment", "Assessment", "Assessing", "With Adjuster"],
    "Settlement Processing": ["Settlement", "Settlement Processing", "settlement processing"],
    "Payment Issued": ["Paid", "Payment Issued", "Payment Released"],
    "Closed": ["Closed", "CLOSED", "Claim Closed"],
}
EVENT_DROP_P = {  # chance an event was never logged (status field simply overwritten)
    "Reported": 0.03, "Registered": 0.08, "Document Collection": 0.12, "Assessment": 0.12,
    "Settlement Processing": 0.18, "Payment Issued": 0.10, "Closed": 0.10,
}
TYPE_LABELS = {
    "Motor": ["Motor", "motor", "MOTOR ", "Motor Vehicle"],
    "Health": ["Health", "health", "Health ", "Mediclaim"],
    "Property": ["Property", "property", "Prop.", "PROPERTY"],
    "Travel": ["Travel", "TRAVEL", "travel "],
}


def td(days):
    return pd.Timedelta(days=float(days))


def fmt_ts(ts, rng, p_dmy=0.30, p_t=0.10):
    if pd.isna(ts):
        return ""
    r = rng.random()
    if r < p_dmy:
        return ts.strftime("%d/%m/%Y %H:%M")
    if r < p_dmy + p_t:
        return ts.strftime("%Y-%m-%dT%H:%M:%S")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def fmt_date(ts, rng, p_dmy=0.30):
    if pd.isna(ts):
        return ""
    return ts.strftime("%d/%m/%Y") if rng.random() < p_dmy else ts.strftime("%Y-%m-%d")


def fmt_money(x, rng):
    if pd.isna(x):
        return ""
    r = rng.random()
    if r < 0.40:
        return f"{x:.2f}"
    if r < 0.70:
        return f"{x:,.0f}"
    if r < 0.85:
        return f"₹{x:,.0f}"
    return f"Rs. {x:,.0f}"


def messy_id(cid, rng):
    r = rng.random()
    if r < 0.03:
        return cid.lower()
    if r < 0.05:
        return cid + " "
    return cid


def messy_doc(name, rng):
    r = rng.random()
    if r < 0.60:
        return name
    if r < 0.75:
        return name.lower()
    if r < 0.85:
        return name.upper()
    if r < 0.93:
        return name.replace(" ", "-")
    return name.replace(" ", "_").lower()


def simulate(n, seed):
    rng = np.random.default_rng(seed)

    def ln(median, sigma):
        return float(rng.lognormal(np.log(median), sigma))

    start = pd.Timestamp("2026-01-05")
    span_days = (AS_OF - start).days - 1
    reported = sorted(start + td(rng.uniform(0, span_days)) + pd.Timedelta(hours=int(rng.integers(8, 19)))
                      for _ in range(n))
    types = rng.choice(list(TYPE_CFG), size=n, p=[c["w"] for c in TYPE_CFG.values()])
    adj_ids = list(ADJUSTERS)
    adj_p = np.array([ADJ_WEIGHT[a] for a in adj_ids], float)
    adj_p /= adj_p.sum()

    claims, events, docs, assigns, pays, truth = [], [], [], [], [], []
    ev_n = doc_n = as_n = pay_n = 0

    for i in range(n):
        cid = f"CLM-2026-{i + 1:05d}"
        ctype = str(types[i])
        cfg = TYPE_CFG[ctype]
        amount = float(np.round(rng.lognormal(np.log(cfg["amt"][0]), cfg["amt"][1]), -2))
        t_rep = reported[i]

        entries = {"Reported": t_rep}
        t_reg = t_rep + td(ln(0.4, 0.6))
        reg_len = ln(cfg["reg"], 0.6)
        t_doc = t_reg + td(reg_len)
        entries["Registered"] = t_reg
        entries["Document Collection"] = t_doc

        # ---- documents -------------------------------------------------------------
        outstanding = rng.random() < 0.20
        late = rng.random() < 0.22
        n_req = len(REQUIRED_DOCS[ctype])
        missing_idx = set(rng.choice(n_req, size=int(rng.integers(1, 3)), replace=False)) if outstanding else set()
        late_idx = set(rng.choice(n_req, size=int(rng.integers(1, 3)), replace=False)) if late else set()
        doc_rows = []
        for k, dname in enumerate(REQUIRED_DOCS[ctype]):
            req_at = t_doc + td(rng.uniform(0, 0.2))
            wait = ln(70, 0.8) if k in missing_idx else ln(cfg["docs"], 0.7)
            if k in late_idx and k not in missing_idx:
                wait += ln(18, 0.5)
            rec_at = req_at + td(wait)
            waived = k not in missing_idx and rng.random() < 0.03
            if not waived and k not in missing_idx and rng.random() < 0.04:
                rec_at = t_rep + td(rng.uniform(0, 0.3))      # submitted with the claim form
            doc_rows.append(dict(name=dname, required=True, req=req_at, rec=rec_at, waived=waived))
        if rng.random() < 0.25:
            req_at = t_doc + td(rng.uniform(0, 0.2))
            doc_rows.append(dict(name=OPTIONAL_DOC, required=False, req=req_at,
                                 rec=req_at + td(ln(4, 0.7)), waived=False))
        pending_real = [d for d in doc_rows if d["required"] and not d["waived"] and d["rec"] > AS_OF]
        proceed = bool(pending_real) and rng.random() < 0.12
        counted = [d["rec"] for d in doc_rows if d["required"] and not d["waived"]
                   and (d["rec"] <= AS_OF or not proceed)]
        docs_end = max(counted) if counted else t_doc
        if proceed:
            docs_end = docs_end + td(ln(4, 0.5))

        # ---- assignment & assessment ---------------------------------------------
        adj1 = str(rng.choice(adj_ids, p=adj_p))
        reassigned = rng.random() < 0.18
        adj2 = str(rng.choice([a for a in adj_ids if a != adj1], p=None)) if reassigned else None
        final_adj = adj2 or adj1
        t_assess = docs_end + td(ln(0.3, 0.5))
        season = 1.5 if (ctype == "Property" and reported[i].month in (8, 9)) else 1.0
        assess_len = ln(cfg["assess"] * ADJ_SLOWDOWN[final_adj] * season, 0.55)
        if reassigned:
            assess_len += ln(2, 0.4)
        entries["Assessment"] = t_assess
        t_settle = t_assess + td(assess_len)
        entries["Settlement Processing"] = t_settle
        settle_len = ln(cfg["settle"], 0.5)
        attempt = t_settle + td(settle_len)
        pay_rows = []
        if rng.random() < 0.06:
            failed_at = attempt + td(ln(0.3, 0.4))
            ok_at = failed_at + td(ln(3, 0.4))
            pay_rows.append(dict(type="Final", amt=0, at=failed_at, status="Failed"))
        else:
            ok_at = attempt + td(ln(0.4, 0.4))
        pay_rows.append(dict(type="Final", amt=amount * rng.uniform(0.85, 1.0), at=ok_at, status="Processed"))
        if ctype == "Health" and rng.random() < 0.12:
            pay_rows.append(dict(type="Interim", amt=amount * 0.25, at=t_assess + td(assess_len * 0.5),
                                 status="Processed"))
        entries["Payment Issued"] = ok_at
        entries["Closed"] = ok_at + td(ln(cfg["close"], 0.5))

        # ---- truncate at the snapshot date ---------------------------------------
        entries = {s: t for s, t in entries.items() if t <= AS_OF}
        latest_idx = max(STAGES.index(s) for s in entries)
        latest = STAGES[latest_idx]
        for s, t in entries.items():
            truth.append(dict(claim_id=cid, claim_type=ctype, stage=s, entered_at=t))

        # claim row (status overwritten: keeps only the latest status)
        cur_stage, cur_ts = latest, entries[latest]
        if latest_idx >= 3 and rng.random() < 0.05:     # stale status field
            cur_stage = STAGES[latest_idx - 1]
            cur_ts = entries[cur_stage]
        claims.append(dict(
            claim_id=cid, policy_no=f"POL-{int(rng.integers(100000, 999999))}", claim_type=ctype,
            claim_amount=amount, region=str(rng.choice(REGIONS, p=REGION_W)), reported_date=t_rep,
            current_status=cur_stage, last_updated=cur_ts, channel=str(rng.choice(CHANNELS)),
        ))

        # status events
        for s, t in entries.items():
            events.append(dict(event_id=f"EV-{ev_n + 1:06d}", claim_id=cid, stage=s, ts=t,
                               changed_by=str(rng.choice(["SYSTEM", "USR-12", "USR-27", final_adj]))))
            ev_n += 1
        if rng.random() < 0.08 and "Assessment" in entries:
            events.append(dict(event_id=f"EV-{ev_n + 1:06d}", claim_id=cid, stage="On Hold",
                               ts=entries["Document Collection"] + td(rng.uniform(0.5, 3)),
                               changed_by="USR-12"))
            ev_n += 1

        # documents (only once Document Collection has started)
        if "Document Collection" in entries:
            for d in doc_rows:
                received = d["rec"] <= AS_OF and not d["waived"]
                status = "Waived" if d["waived"] else ("Received" if received else "Pending")
                docs.append(dict(doc_id=f"DOC-{doc_n + 1:05d}", claim_id=cid, doc_type=d["name"],
                                 required=d["required"], requested=d["req"],
                                 received=d["rec"] if received else pd.NaT, status=status))
                doc_n += 1

        # adjuster assignments
        if rng.random() > 0.05 and "Registered" in entries:
            a_at = entries["Registered"] + (t_doc - t_reg) * rng.uniform(0.05, 0.9)
            if a_at <= AS_OF:
                re_at = t_assess + td(assess_len * rng.uniform(0.2, 0.7))
                if reassigned and re_at <= AS_OF:
                    assigns.append(dict(assignment_id=f"ASG-{as_n + 1:05d}", claim_id=cid, adjuster_id=adj1,
                                        assigned=a_at, released=re_at))
                    assigns.append(dict(assignment_id=f"ASG-{as_n + 2:05d}", claim_id=cid, adjuster_id=adj2,
                                        assigned=re_at, released=pd.NaT))
                    as_n += 2
                else:
                    assigns.append(dict(assignment_id=f"ASG-{as_n + 1:05d}", claim_id=cid,
                                        adjuster_id=adj2 if (reassigned and False) else adj1,
                                        assigned=a_at, released=pd.NaT))
                    as_n += 1

        # payments (final payment records are lost for ~3% of paid claims)
        lose_final = rng.random() < 0.03
        for p in pay_rows:
            if p["at"] > AS_OF:
                continue
            if lose_final and p["type"] == "Final" and p["status"] == "Processed":
                continue
            pays.append(dict(payment_id=f"PAY-{pay_n + 1:05d}", claim_id=cid, payment_type=p["type"],
                             amount=p["amt"], paid_at=p["at"], status=p["status"]))
            pay_n += 1

    return (pd.DataFrame(claims), pd.DataFrame(events), pd.DataFrame(docs),
            pd.DataFrame(assigns), pd.DataFrame(pays), pd.DataFrame(truth), rng)


def make_messy(claims, events, docs, assigns, pays, rng):
    """Turn the clean simulation into the kind of extracts a real insurer would hand over."""
    # ---------------- status_history ----------------
    ev = events.copy()
    keep = ~ev.apply(lambda r: r["stage"] in EVENT_DROP_P and rng.random() < EVENT_DROP_P[r["stage"]], axis=1)
    ev = ev[keep].reset_index(drop=True)

    # out-of-order timestamps: a later-stage event stamped before the stage that preceded it
    late_idx = ev.index[ev["stage"].isin(["Assessment", "Settlement Processing", "Payment Issued", "Closed"])]
    for ix in rng.choice(late_idx, size=max(1, int(0.015 * len(ev))), replace=False):
        prev = ev[(ev.claim_id == ev.at[ix, "claim_id"]) & (ev.stage == "Document Collection")]
        base = prev["ts"].min() if len(prev) else ev.at[ix, "ts"] - td(3)
        ev.at[ix, "ts"] = base - td(rng.uniform(0.2, 2))

    ev["status"] = ev["stage"].map(lambda s: str(rng.choice(STATUS_LABELS[s])) if s in STATUS_LABELS else s)
    ev["changed_at"] = [fmt_ts(t, rng) for t in ev["ts"]]

    # a few future-dated typos (year 2027)
    for ix in rng.choice(ev.index, size=4, replace=False):
        ev.at[ix, "changed_at"] = ev.at[ix, "ts"].replace(year=2027).strftime("%Y-%m-%d %H:%M:%S")
    # events with a label the workflow does not know
    extra = ev.sample(5, random_state=int(rng.integers(1_000_000))).copy()
    extra["status"] = "Escalated to Supervisor"
    extra["event_id"] = [f"EV-X{i:04d}" for i in range(len(extra))]
    # orphans (claim id not in claims)
    orph = ev.sample(6, random_state=int(rng.integers(1_000_000))).copy()
    orph["claim_id"] = [f"CLM-2026-{9000 + i}" for i in range(len(orph))]
    orph["event_id"] = [f"EV-O{i:04d}" for i in range(len(orph))]
    # exact duplicates and near duplicates
    dup = ev.sample(18, random_state=int(rng.integers(1_000_000))).copy()
    near = ev.sample(14, random_state=int(rng.integers(1_000_000))).copy()
    near["event_id"] = [f"EV-N{i:04d}" for i in range(len(near))]
    near["changed_at"] = [fmt_ts(t + pd.Timedelta(minutes=int(rng.integers(1, 9))), rng) for t in near["ts"]]
    ev = pd.concat([ev, extra, orph, dup, near], ignore_index=True)
    ev["claim_id"] = [messy_id(c, rng) for c in ev["claim_id"]]
    ev = ev.sample(frac=1, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    status_history = ev[["event_id", "claim_id", "status", "changed_at", "changed_by"]]

    # ---------------- claims ----------------
    cl = claims.copy()
    stale = cl.sample(12, random_state=int(rng.integers(1_000_000))).copy()   # stale duplicate rows
    stale["current_status"] = "Registered"
    stale["last_updated"] = stale["reported_date"] + td(0.5)
    cl["claim_type"] = [str(rng.choice(TYPE_LABELS[t])) for t in cl["claim_type"]]
    stale["claim_type"] = [str(rng.choice(TYPE_LABELS[[k for k, v in TYPE_LABELS.items() if t in v][0]]))
                           if t in sum(TYPE_LABELS.values(), []) else t for t in stale["claim_type"]]
    cl = pd.concat([cl, stale], ignore_index=True)
    cl["claim_amount"] = [fmt_money(a, rng) if rng.random() > 0.03 else "" for a in cl["claim_amount"]]
    cl["region"] = [("" if rng.random() < 0.04 else (r.lower() if rng.random() < 0.08 else
                     (r.upper() + " " if rng.random() < 0.05 else r))) for r in cl["region"]]
    cl["reported_date"] = [fmt_date(t, rng) if rng.random() > 0.01 else "" for t in cl["reported_date"]]
    cl["current_status"] = [str(rng.choice(STATUS_LABELS[s])) for s in cl["current_status"]]
    cl["last_updated"] = [fmt_ts(t, rng) for t in cl["last_updated"]]
    cl["claim_id"] = [messy_id(c, rng) for c in cl["claim_id"]]

    # ---------------- documents ----------------
    d = docs.copy()
    d["doc_type"] = [messy_doc(x, rng) for x in d["doc_type"]]
    d["required"] = [str(rng.choice(["Y", "Yes", "TRUE", "1"])) if r else str(rng.choice(["N", "No", "FALSE", "0"]))
                     for r in d["required"]]
    status_map = {"Received": ["Received", "received", "RCVD"], "Pending": ["Pending", "pending", "Outstanding"],
                  "Waived": ["Waived", "NA"]}
    # contradictions: status says Pending but a date exists / Received with no date
    dstat = []
    rec_txt = []
    for s, rcv in zip(d["status"], d["received"]):
        txt = fmt_ts(rcv, rng) if not pd.isna(rcv) else ""
        if s == "Received" and rng.random() < 0.02:
            txt = ""                       # received, date lost
        lab = str(rng.choice(status_map[s]))
        if s == "Received" and rng.random() < 0.02:
            lab = "Pending"                # status never updated
        dstat.append(lab)
        rec_txt.append(txt)
    d["doc_status"] = dstat
    d["received_date"] = rec_txt
    d["requested_date"] = [fmt_date(t, rng) for t in d["requested"]]
    d_dup = d.sample(15, random_state=int(rng.integers(1_000_000))).copy()
    d_orph = d.sample(4, random_state=int(rng.integers(1_000_000))).copy()
    d_orph["claim_id"] = [f"CLM-2026-{9500 + i}" for i in range(len(d_orph))]
    d_orph["doc_id"] = [f"DOC-O{i:03d}" for i in range(len(d_orph))]
    d = pd.concat([d, d_dup, d_orph], ignore_index=True)
    d["claim_id"] = [messy_id(c, rng) for c in d["claim_id"]]
    documents = d[["doc_id", "claim_id", "doc_type", "required", "requested_date", "received_date", "doc_status"]]

    # ---------------- adjusters ----------------
    a = assigns.copy()
    a["adjuster_name"] = [ADJUSTERS[x] if rng.random() > 0.15 else ADJUSTERS[x].lower() for x in a["adjuster_id"]]
    a["adjuster_id"] = [str(rng.choice([x, x, x, x.lower(), x.replace("-", ""), x.replace("-", " ")]))
                        for x in a["adjuster_id"]]
    a["assigned_at"] = [fmt_ts(t, rng) for t in a["assigned"]]
    a["released_at"] = [fmt_ts(t, rng) for t in a["released"]]
    a_dup = a.sample(6, random_state=int(rng.integers(1_000_000))).copy()
    a = pd.concat([a, a_dup], ignore_index=True)
    a["claim_id"] = [messy_id(c, rng) for c in a["claim_id"]]
    adjusters = a[["assignment_id", "claim_id", "adjuster_id", "adjuster_name", "assigned_at", "released_at"]]

    # ---------------- payments ----------------
    p = pays.copy()
    p["payment_type"] = [str(rng.choice({"Final": ["Final", "final", "FINAL", "Settlement"],
                                         "Interim": ["Interim", "interim", "Part Payment"]}[t]))
                         for t in p["payment_type"]]
    stat_lab = {"Processed": ["Processed", "PAID", "processed", "Success"], "Failed": ["Failed", "FAILED", "Declined"]}
    p["payment_status"] = [str(rng.choice(stat_lab[s])) for s in p["status"]]
    interim = p.index[p["payment_type"].str.lower().isin(["interim", "part payment"])]
    for ix in interim[: min(2, len(interim))]:
        p.at[ix, "payment_status"] = "Reversed"
    p["amount"] = [fmt_money(x, rng) if x else "0" for x in p["amount"]]
    p["paid_at"] = [fmt_ts(t, rng) for t in p["paid_at"]]
    p_dup = p.sample(min(8, len(p)), random_state=int(rng.integers(1_000_000))).copy()
    p = pd.concat([p, p_dup], ignore_index=True)
    p["claim_id"] = [messy_id(c, rng) for c in p["claim_id"]]
    payments = p[["payment_id", "claim_id", "payment_type", "amount", "paid_at", "payment_status"]]

    return cl, status_history, documents, adjusters, payments


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600, help="number of claims")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    claims, events, docs, assigns, pays, truth, rng = simulate(args.n, args.seed)
    cl, sh, dc, ad, pm = make_messy(claims, events, docs, assigns, pays, rng)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    cl.to_csv(RAW_DIR / "claims.csv", index=False)
    sh.to_csv(RAW_DIR / "status_history.csv", index=False)
    dc.to_csv(RAW_DIR / "documents.csv", index=False)
    ad.to_csv(RAW_DIR / "adjusters.csv", index=False)
    pm.to_csv(RAW_DIR / "payments.csv", index=False)
    truth.to_csv(TRUTH_DIR / "stage_entries_truth.csv", index=False)
    print(f"Wrote raw extracts to {RAW_DIR}")
    for name, df in [("claims", cl), ("status_history", sh), ("documents", dc), ("adjusters", ad), ("payments", pm)]:
        print(f"  {name:15s} {len(df):6d} rows")


if __name__ == "__main__":
    main()
