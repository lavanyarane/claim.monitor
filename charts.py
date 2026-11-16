"""charts.py - Plotly figures shared by the Streamlit app and the static HTML report.

Colour rules: one fixed colour per workflow stage (never re-assigned when a filter removes a stage),
one hue for single-series charts, thin marks, light gridlines, legends whenever there are 2+ series.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import GRID, INK, INK_2, MUTED, SEQ_BLUE, SERIES, WORK_STAGES

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
STAGE_COLOR = dict(zip(WORK_STAGES, SERIES[:6]))
BLUE, ORANGE = SERIES[0], SERIES[1]


def _style(fig, title=None, height=340, legend=True, xtitle=None, ytitle=None):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=15, color=INK)) if title else None,
        height=height, margin=dict(l=8, r=16, t=(78 if legend else 48) if title else (40 if legend else 16), b=8),
        font=dict(family=FONT, size=12, color=INK_2), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, bargap=0.35,
        legend=dict(orientation="h", y=1.0, x=0, xanchor="left", yanchor="bottom", font=dict(size=12)),
        hoverlabel=dict(font_family=FONT, bgcolor="#ffffff", font_color=INK, bordercolor=GRID),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
                     title=dict(text=xtitle, font=dict(color=INK_2)) if xtitle else None)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
                     title=dict(text=ytitle, font=dict(color=INK_2)) if ytitle else None)
    return fig


def _empty(msg="No data for the current filters"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=240, legend=False)


def backlog_by_stage(bl):
    if bl.empty:
        return _empty()
    g = bl.groupby(["reconstructed_stage", "sla_breached"]).size().unstack(fill_value=0)
    stages = [s for s in WORK_STAGES if s in g.index]
    fig = go.Figure()
    for flag, name, color in [(False, "Within target", BLUE), (True, "Past target", ORANGE)]:
        vals = [int(g.at[s, flag]) if flag in g.columns else 0 for s in stages]
        fig.add_bar(y=stages, x=vals, orientation="h", name=name, marker=dict(color=color, line=dict(color="#fcfcfb", width=2)),
                    hovertemplate="%{y}<br>" + name + ": %{x} claims<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Open claims by current stage", xtitle="Open claims", height=300)


def stage_duration_bars(st):
    st = st[st["completed"] > 0]
    if st.empty:
        return _empty()
    fig = go.Figure()
    fig.add_bar(y=st["stage"], x=st["median_days"], orientation="h", name="Median", marker_color=BLUE,
                text=st["median_days"].round(1), textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>Median: %{x:.1f} days<extra></extra>")
    fig.add_bar(y=st["stage"], x=st["p90_days"], orientation="h", name="90th percentile", marker_color=ORANGE,
                text=st["p90_days"].round(0), textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>90th percentile: %{x:.1f} days<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Days spent per stage (completed stages)", xtitle="Days", height=360)


def time_share(st):
    st = st[st["elapsed_days"] > 0]
    if st.empty:
        return _empty()
    top = st["share_pct"].idxmax()
    colors = [ORANGE if i == top else BLUE for i in st.index]
    fig = go.Figure(go.Bar(y=st["stage"], x=st["share_pct"], orientation="h", marker_color=colors,
                           text=[f"{v:.0f}%" for v in st["share_pct"]], textposition="outside", cliponaxis=False,
                           customdata=np.c_[st["elapsed_days"], st["open_now"]],
                           hovertemplate="%{y}<br>%{x:.1f}% of elapsed time<br>%{customdata[0]:,.0f} claim-days"
                                         "<br>%{customdata[1]} open claims here<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Share of all elapsed claim-days (bottleneck score)", xtitle="% of elapsed time",
                  legend=False, height=360)


def turnaround_hist(cs):
    t = cs.loc[~cs["is_open"], "total_turnaround_days"].dropna()
    if t.empty:
        return _empty()
    fig = go.Figure(go.Histogram(x=t, nbinsx=40, marker=dict(color=BLUE, line=dict(color="#fcfcfb", width=1)),
                                 hovertemplate="%{x} days<br>%{y} claims<extra></extra>"))
    for q, label, dash in [(0.5, "Median", "solid"), (0.9, "90th pct", "dot")]:
        v = t.quantile(q)
        fig.add_vline(x=v, line=dict(color=ORANGE, width=2, dash=dash), annotation_text=f"{label} {v:.0f}d",
                      annotation_position="top", annotation_font=dict(color=INK_2))
    return _style(fig, "Total turnaround of closed claims", xtitle="Days from report to closure",
                  ytitle="Claims", legend=False, height=340)


def turnaround_by_type(cs):
    c = cs[~cs["is_open"] & cs["total_turnaround_days"].notna()]
    if c.empty:
        return _empty()
    fig = go.Figure()
    for t in sorted(c["claim_type"].dropna().unique()):
        fig.add_box(y=c.loc[c["claim_type"] == t, "total_turnaround_days"], name=t, marker_color=BLUE,
                    line=dict(width=1.5, color=BLUE), fillcolor="rgba(42,120,214,0.18)", boxpoints="outliers",
                    hovertemplate=t + "<br>%{y:.0f} days<extra></extra>")
    return _style(fig, "Turnaround by claim type (closed claims)", ytitle="Days", legend=False, height=340)


def aging_chart(buckets):
    if buckets["claims"].sum() == 0:
        return _empty()
    fig = go.Figure()
    for s in WORK_STAGES:
        d = buckets[buckets["reconstructed_stage"] == s]
        if d["claims"].sum() == 0:
            continue
        fig.add_bar(x=d["age_bucket"].astype(str), y=d["claims"], name=s,
                    marker=dict(color=STAGE_COLOR[s], line=dict(color="#fcfcfb", width=2)),
                    hovertemplate="%{x}<br>" + s + ": %{y} claims<extra></extra>")
    fig.update_layout(barmode="stack")
    return _style(fig, "Backlog ageing: days since the claim was reported", ytitle="Open claims", height=340)


def missing_docs_bar(m):
    if m.empty:
        return _empty("No required documents are outstanding")
    m = m.sort_values("documents")
    fig = go.Figure(go.Bar(y=m["doc_type"], x=m["documents"], orientation="h", marker_color=BLUE,
                           text=m["documents"], textposition="outside", cliponaxis=False,
                           customdata=m["median_wait_days"],
                           hovertemplate="%{y}<br>%{x} open claims missing it<br>median wait so far "
                                         "%{customdata:.0f} days<extra></extra>"))
    return _style(fig, "Required documents still outstanding (open claims)", xtitle="Documents missing",
                  legend=False, height=340)


def doc_receipt_bar(r):
    if r.empty:
        return _empty()
    r = r.sort_values("median_days")
    fig = go.Figure(go.Bar(y=r["doc_type"], x=r["median_days"], orientation="h", marker_color=BLUE,
                           text=r["median_days"].round(1), textposition="outside", cliponaxis=False,
                           customdata=np.c_[r["p90_days"], r["documents"]],
                           hovertemplate="%{y}<br>median %{x:.1f} days<br>90th pct %{customdata[0]:.0f} days"
                                         "<br>%{customdata[1]} documents<extra></extra>"))
    return _style(fig, "Median days from request to receipt, by document", xtitle="Days", legend=False, height=360)


def turnaround_by_doc_speed_box(ts):
    if ts.empty or ts["document_speed"].nunique() < 2:
        return _empty("Not enough closed claims to compare")
    fig = go.Figure()
    for (name, color, fill) in [(sorted(ts["document_speed"].unique())[0], BLUE, "rgba(42,120,214,0.18)"),
                                (sorted(ts["document_speed"].unique())[1], ORANGE, "rgba(235,104,52,0.18)")]:
        d = ts.loc[ts["document_speed"] == name, "total_turnaround_days"]
        fig.add_box(y=d, name=name, marker_color=color, line=dict(width=1.5, color=color), fillcolor=fill,
                    boxpoints="outliers", hovertemplate=name + "<br>%{y:.0f} days<extra></extra>")
    return _style(fig, "Total turnaround vs. slowest required document (closed claims)", ytitle="Days",
                  height=360, legend=False)


def heatmap(m):
    if m.empty:
        return _empty()
    z = m.values
    fig = go.Figure(go.Heatmap(
        z=z, x=list(m.columns), y=list(m.index), colorscale=[[i / 6, c] for i, c in enumerate(SEQ_BLUE)],
        text=np.round(z, 1), texttemplate="%{text}", hovertemplate="%{y} / %{x}<br>median %{z:.1f} days<extra></extra>",
        colorbar=dict(title="Days", thickness=10), xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Median days per stage by claim type", height=300, legend=False)


def adjuster_bar(a):
    if a.empty:
        return _empty("No adjuster has enough completed assessments")
    a = a.sort_values("median_days")
    med = a["median_days"].median()
    fig = go.Figure(go.Bar(y=a["adjuster_name"], x=a["median_days"], orientation="h", marker_color=BLUE,
                           text=a["median_days"].round(1), textposition="outside", cliponaxis=False,
                           customdata=np.c_[a["claims"], a["p90_days"], a["open_in_assessment"]],
                           hovertemplate="%{y}<br>median %{x:.1f} days<br>90th pct %{customdata[1]:.0f}"
                                         "<br>%{customdata[0]} completed, %{customdata[2]} open now<extra></extra>"))
    fig.add_vline(x=med, line=dict(color=ORANGE, width=2, dash="dot"), annotation_text=f"Median of adjusters {med:.1f}d",
                  annotation_position="bottom right", annotation_font=dict(color=INK_2))
    return _style(fig, "Assessment time by adjuster (days)", xtitle="Median days in Assessment", legend=False,
                  height=380)


def case_timeline(sd_claim, closed_at=None):
    """Gantt-style view of one claim: one bar per stage, plus a marker for closure."""
    if sd_claim.empty:
        return _empty("No stage history for this claim")
    d = sd_claim.sort_values("stage_order")
    fig = go.Figure()
    for r in d.itertuples():
        dur = max((r.left_at - r.entered_at).total_seconds() / 86400, 0)
        fig.add_bar(y=[r.stage], x=[(r.left_at - r.entered_at).total_seconds() * 1000], base=[r.entered_at],
                    orientation="h", name=r.stage, marker=dict(color=STAGE_COLOR[r.stage],
                                                                line=dict(color="#fcfcfb", width=2)),
                    hovertemplate=f"{r.stage}<br>{dur:.1f} days"
                                  f"{' (still open)' if r.is_open else ''}<br>entered via {r.entry_source}"
                                  f" ({r.entry_confidence})<extra></extra>")
    if closed_at is not None and pd.notna(closed_at):
        fig.add_trace(go.Scatter(x=[closed_at], y=[d.iloc[-1]["stage"]], mode="markers", name="Closed",
                                 marker=dict(symbol="diamond", size=12, color=INK_2),
                                 hovertemplate=f"Closed {closed_at:%Y-%m-%d %H:%M}<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=list(d["stage"]))
    fig.update_xaxes(type="date")
    return _style(fig, None, height=max(200, 46 * len(d) + 70), legend=False)
