# Extraction prunes the causal step on realistic traces

The primary finding of this project is a negative one about its own tool.

On traces of realistic depth, the extraction engine does not merely fail to help
a reader locate a failure. It removes the evidence needed to locate it. The
top-five critical steps contain the faulted step in **0 of 24** failed runs, and
the engine ranks that step **13th of about 20** on average, below the planning
and verification steps it promotes instead.

This was invisible on the shallow corpus the engine was built against, and it
appears as soon as traces grow.

## 1. Mechanism

Two separate weightings are involved. Both are driven by the same underlying
assumption, and both break the same way.

**Ranking, which decides what the extracted condition is shown.**
`app/extraction/scoring.py` scores every step as

    0.4 × evidence_survival  +  0.3 × branch  +  0.3 × error

`evidence_survival` is the embedding cosine similarity between a step's output
and the run's final output.

**Attribution, which decides the predicted failure origin.**
`app/extraction/attribution.py` scores rules as

    error_signal 3.0 | first_unsupported_claim 2.0 | divergent_branch 1.5 | thin_retrieval 1.0

The failure is the same in both. A `plan`, `retry`, `decision` or `critique`
step contains procedural text — *"Round 2 of 3: look for the passage naming the
station"*, *"Best match scored 0.31, below the 0.45 threshold"*. Procedural text
never resembles a final answer, so its similarity to the final output is low by
construction. The engine reads low similarity as **divergence**: the run
departed from this step, therefore this step is a branch point, therefore it is
critical.

That inference is wrong for procedural steps. A plan is *supposed* not to look
like the answer. The signal cannot distinguish "the run diverged here" from
"this step was never going to resemble the output".

Meanwhile the step that actually lost a chunk is a `retrieval` whose output
still contains three or four plausible passages that closely resemble the final
answer. It scores **high** on evidence survival, so it does not look anomalous,
and the one rule that would catch it, `thin_retrieval` at weight 1.0, is
outweighed by `divergent_branch` at 1.5.

The result, on one representative run:

```
seq  1 score 1.50  plan       "plan step and the run diverged from it afterwards"
seq  4 score 1.50  plan       "plan step and the run diverged from it afterwards"
seq  7 score 1.50  retry      "retry step and the run diverged from it afterwards"
seq  9 score 1.50  plan       "plan step and the run diverged from it afterwards"
seq 12 score 1.50  retry      "retry step and the run diverged from it afterwards"
seq 19 score 1.50  critique   "critique step and the run diverged from it afterwards"
seq 20 score 1.50  critique   "critique step and the run diverged from it afterwards"
seq  3 score 1.00  retrieval  "retrieval returned 3 references, fewer than the 4 expected"   <- the fault
```

Seven procedural steps tie at 1.5. The faulted retrieval, correctly identified
by the one rule designed to catch it, comes eighth.

## 2. Numbers

| Measure | Shallow corpus (median 4 steps) | Deep corpus (median 20 steps) |
|---|---|---|
| Heuristic attribution | 11/31 (35.5%) | **0/24 (0.0%)** |
| Top-5 contains the causal step | not measurable, top-5 was the whole trace | **0/24** |
| Median rank of the causal step | n/a | **13 of ~20** |
| Extracted condition, primary study | 19/31 (61.3%) | **0/24 (0.0%)** |

What the extracted condition is shown instead, across all 24 runs' top-five
slots: `plan` 48, `revision` 22, `final` 22, `reasoning` 13, `critique` 9,
`retry` 6. **`retrieval` appears zero times.** The event type that carries every
fault in this population is entirely absent from the summary.

The extracted condition's answers are correspondingly scattered: 11 distinct
step numbers across 24 runs. It is not systematically wrong, it is guessing,
because it is not being shown the answer. The raw-log condition, reading the
same runs unpruned, answered the same value on 23 of 24.

## 3. Why this scales with trace length

This is the part worth generalising, and it is the reason the defect could not
have been found on the original corpus.

The two classes of signal scale differently with trace length:

- **Fault-presence signals** are roughly constant. A run has one injected fault,
  so one step is thin, or empty, or errors, however long the trace is.
- **Procedural-structure signals grow with the trace.** More retrieval rounds
  mean more plans and more retries; more claims mean more critiques. Every one
  of them scores as a divergent branch.

On the shallow corpus a run held one or two procedural steps against one faulted
step. On the deep corpus the median run holds **7** procedural steps (range 3 to
8) against the same single faulted step. The noise floor rose above the signal.

Since the top-k is fixed at 5, this is a ranking competition the causal step
loses more decisively the longer the trace gets. Extrapolating, a production
trace of 100+ steps would bury it further.

The general form: **an extraction heuristic tuned on short traces can encode a
signal-to-noise ratio that only holds at that length.** Nothing in the shallow
evaluation would reveal it. The engine scored 35.5% on attribution and looked
merely mediocre; it was in fact relying on there being almost nothing to rank
against.

## 4. What this means

Stated plainly: **on traces of realistic depth this extraction engine is not
neutral, it is harmful.** A reader given the extracted summary is worse off than
a reader given the raw log, because the summary has removed the step that
explains the failure and replaced it with the pipeline's own bookkeeping.

The honest version of the project's original claim is therefore inverted. The
hypothesis was that extracting critical steps would help a reader locate a
failure faster and at least as accurately. On the corpus where the question is
answerable at all, extraction took a task the raw log solves and made it
unsolvable.

No weight, threshold or signal definition was changed in response to this
result. The numbers above are what the engine as specified produces.

## 5. Post-hoc exploration, not a pre-registered result

Everything below was computed **after** seeing the result above. It is
exploration to characterise the defect, not a corrected result, and the figures
in sections 1 to 4 remain the finding. No committed weight was altered to
produce it.

The obvious hypothesis is that the branch weight is simply too high. Re-ranking
read-only with the branch component set to zero:

| Ranking | Top-5 contains causal step | Median rank |
|---|---|---|
| As shipped (branch 0.3) | 0/24 | 13 |
| Branch component zeroed | **3/24** | **8** |

Removing the noise entirely recovers the causal step in 3 runs of 24 and lifts
it from 13th to 8th — still outside a top-5.

**So the defect is not a mis-set weight.** Even with the confounding signal
deleted, the engine has no signal that positively identifies a retrieval which
returned plausible-but-incomplete evidence. `evidence_survival` actively works
against it, because a retrieval that dropped one chunk still returns three that
resemble the answer closely. Detecting this would need a signal the engine does
not have: comparing retrieved chunks against each other, or against what the
question asks for, rather than against the final output.

Re-weighting would therefore be fitting the existing signals to this corpus
rather than fixing the gap, which is why none was done.
