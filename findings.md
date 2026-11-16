# Findings

Snapshot date: 12 Nov 2026. 600 synthetic claims (96 open, 504 closed).

- **Turnaround.** 504 closed claims took a median of 33 days from report to closure (mean 43, 90th percentile 81). The mean sits well above the median, so a minority of very slow claims drives the average.
- **Bottleneck.** Document Collection uses the largest share of elapsed time (68% of all claim-days across stages). Completed cases spent a median of 14.6 days there (90th percentile 60), and 58 of the 96 open claims are sitting in it right now.
- **Documentation.** 59 open claims are missing at least one required document (112 documents in total); 30 of them have been in Document Collection for more than 30 days. The most frequently missing item is Claim Form (26 claims).
- **Documents vs. turnaround.** Closed claims where a required document took more than 14 days to arrive (256 claims) had a median turnaround of 47 days, against 23 days for the other 248 claims. This is an association in observational data, not proof that documents alone caused the delay.
- **Backlog.** 96 claims are open (median age 44 days); 56 are past the target for their current stage and 31 have had no recorded activity for 21+ days. Oldest: CLM-2026-00082 (267 days).
- **Claim type.** Property claims are slowest (median 52 days) and Travel fastest (24 days).
- **Adjuster workload.** Anjali Mehta (ADJ-07) has the longest median assessment time (10.9 days vs 7.3 across adjusters) and 5 open claims in assessment. Worth a workload review before drawing conclusions about individual performance.
- **Record quality.** 38% of cases carry at least one completeness flag (missing documents, a history gap, a status field that disagrees with the history, or no adjuster/payment record). 253 cases needed at least one stage time rebuilt from another source; each is labelled with its source and confidence.
- **Scope.** This tool measures time and completeness only. It does not assess whether any claim is eligible or payable.

## Summary measures

| Measure | Value |
|---|---|
| Median total turnaround (closed) | 32.7 days |
| 90th percentile turnaround | 80.9 days |
| Claims with a missing required document | 10.5% |
| Cases with any completeness flag | 37.8% |
| Open claims past their stage target | 56 of 96 |

## Stage statistics (completed stages)

| Stage | Completed | Median days | P90 days | Open now | Share of elapsed time |
|---|---|---|---|---|---|
| Reported | 596 | 0.4 | 1.0 | 0 | 1.2% |
| Registered | 596 | 0.5 | 1.0 | 0 | 1.3% |
| Document Collection | 535 | 14.6 | 59.6 | 58 | 68.1% |
| Assessment | 411 | 7.2 | 17.4 | 22 | 16.1% |
| Settlement Processing | 403 | 4.2 | 8.5 | 10 | 8.0% |
| Payment Issued | 502 | 1.6 | 3.3 | 6 | 5.3% |
