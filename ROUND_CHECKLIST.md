# Research round checklist — non-negotiable

Every exploration round MUST follow these steps in order. Skipping any step
is how we ended up with inconsistent numbers in round 3 vs round 3.5.
This file is the enforcement mechanism.

Committed to git, referenced from [CONVENTIONS.md](CONVENTIONS.md), and
every round's `findings.md` must include a filled-out **COMPLIANCE**
section proving these steps were followed.

---

## Before writing any code for a round

- [ ] **Clear objective written down.** What axis are we exploring?
      What factor do we expect to move the needle? Write it in the
      first paragraph of `docs/research/round<N>_findings.md`.
- [ ] **Prior rounds' output grepped.** Don't repeat sweeps we already
      ran. Check `backtest_results/` directory list.
- [ ] **Hypothesis stated.** What would SUCCESS look like? What would
      REFUTE the hypothesis? Write it in the findings doc.

## During the round

- [ ] **Factors being varied are enumerated explicitly** in the findings
      doc, not just in code. Reader must be able to reconstruct the
      sweep from the doc alone.
- [ ] **Every exploration writes raw CSV to `backtest_results/<id>/`.**
      No in-memory-only results.
- [ ] **Per-trade records captured for top configs** via
      `pf.trades.records_readable` — saved as `<run_id>/trades_top<N>.parquet`.
      Required so MAE/MFE analysis is possible later.
- [ ] **Walk-forward OOS is used.** No full-sample metrics for decisions.
- [ ] **Diary event emitted** via `src.utils.diary.log_event("exploration_complete", ...)`.

## Metric computation rules — non-negotiable

- [ ] **Annualization is computed from actual OOS coverage years**,
      not a hardcoded ratio. Formula:
      ```
      annual_profit = (sum of per-split profits) / (total OOS years covered)
      total OOS years = n_splits × (avg_split_bars × bar_minutes) / 525600
      ```
      Do NOT use `12/8` or `12/13` shortcuts.
- [ ] **Profit factor, expectancy, win rate, max DD** are the primary
      ranking metrics. Sharpe is reported but NOT a gate (annualization
      is frequency-dependent and has been buggy).
- [ ] **RRR (Return/Risk Ratio)** = annual profit / max drawdown $.
      Always compute + report.
- [ ] **Multi-testing caveat** noted in every findings doc: "N winners
      of M configs at threshold X = Y%; at random we'd expect Z% at p=0.05."

## Intelligence layer — mandatory, not optional

- [ ] **Results passed through `src.utils.ai_research.ask()`** via a
      prompt built with `src.backtest.iterate.build_prompt(...)`.
      Required. The artifact lands under
      `fx-scalper/docs/research/ai_queries/`.
- [ ] **AI response reviewed** before round is closed. Notable
      recommendations get actioned into the next round's plan.
- [ ] **Budget cap honored** ($10/UTC-day default). Check before
      running cost-heavy queries.

## Writing up

Every `docs/research/round<N>_findings.md` MUST include these sections,
in this order:

1. **Objective** — 1-2 sentences. What this round tests.
2. **Hypothesis** — what we expect, what would refute.
3. **Factors varied** — enumerated list with ranges.
4. **Method** — data source, walk-forward setup, metrics computation rules.
5. **Headline numbers** — winners table, dollar expectancy, DD.
6. **Per-timeframe / per-factor breakdown** — what moved the needle.
7. **Caveats** — multi-testing, degradation, correlation assumptions.
8. **AI analysis integration** — quote key recommendations from the
   vbt.chat artifact + link to the `ai_queries/*.md` file.
9. **Action items for next round** — what this round's results imply
   for round N+1.
10. **COMPLIANCE** — filled-out checklist confirming each item above
    was done.

## Commit message template

```
explore round <N>: <short headline>

Objective: <one line>
Factors varied: <list>
Headline: <1-2 key numbers>
Winners: <count at PF>1.1 OOS>, top PF <x.xx>, top expectancy $<y.yy>
AI analysis: docs/research/ai_queries/<file>
Full writeup: docs/research/round<N>_findings.md

Next round will: <one line>
```

## Lessons learned (update this section each round)

### Round 1 → 2
- Naive unfiltered families all lose on M1 after spread. Regime/session
  filtering was the key missing piece. Lesson: always include context
  filters in initial sweeps.

### Round 2 → 3
- Session filter was the "edge," not the indicator choice. M1 is
  spread-dominated; higher TFs dramatically change the profile.
  Lesson: always test multiple timeframes before concluding an edge
  is real.

### Round 3 → 3.5
- My annualization math was inconsistent across docs ($3,290 vs $1,237
  vs correct ~$1,496). **Root cause:** hardcoded "12/8" factor instead
  of computing from actual OOS coverage years. **Fix**: annualization
  formula documented above, no shortcuts. Also: round 3.5 meta-analysis
  was NOT passed through vbt.chat before publishing, so the error
  wasn't caught.
- **Process fix:** THIS FILE. Every round's doc must include a filled-
  out COMPLIANCE section. Every round must consult the intelligence
  layer before findings are committed.
