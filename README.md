# Multi-Agent Execution Trace Inspector

Ingests multi-agent LLM execution traces, extracts the steps that determined the
outcome, predicts where failed runs went wrong, and presents all of it in a web
interface. The evaluation is the deliverable; the interface is packaging.

## 1. Research question

Agent execution traces are too long for humans to inspect. Which parts of a
trace actually matter, and does extracting only those parts help a reader locate
the cause of a failure faster and more accurately than reading the raw log?

Three questions go unanswered by a raw log. **Which steps mattered?** Most steps
in a trace are inert; a handful determine the outcome. **Where did it break?**
Error origin is distinct from error manifestation: an agent may produce a wrong
final answer because a retrieval three steps earlier returned nothing. **What is
the answer grounded in?** A final claim may or may not trace back to retrieved
evidence.

## 2. Method

Two agentic workflows were instrumented to emit traces conforming to a single
framework-agnostic schema: a `reviewer_pipeline` (executor → reviewer → revision
over mathematics tasks) and a `rag_qa` agent over a controlled document set.
Both are LangGraph state machines. Faults of four kinds were injected
deliberately so that failure causes have ground truth: dropped retrieval,
truncated tool result, forced false rejection, and injected contradiction. Each
run records which fault was applied and at which step, at the **first**
application rather than the last, since attribution asks which step introduced
the error.

`rag_qa` is at version 2.0.0 and runs gather → tools → reason → verify →
revise → answer: two or three retrieval rounds, each with its own plan, query
reformulation and retrieval, plus a retry when a round comes back thin; an
extraction tool and an arithmetic tool, each with separate `tool_call` and
`tool_result` steps; then per-claim verification, with a revision when any claim
is rejected. Only the draft and the revision call a model. The earlier 1.0.0
version emitted 4 to 8 steps and is described in 3.4, because its depth is the
reason the first study could not test its own hypothesis.

The extraction engine scores every step on three signals — evidence survival
(embedding cosine similarity of the step's output against the final output),
branch points, and error markers — and ranks the most critical. It is
deterministic and contains no LLM calls; embeddings only.

The primary study takes every failed run with a known injected fault and asks an
LLM judge one question in two conditions: *which step introduced the error?*
Condition A presents the full raw trace as JSON. Condition B presents only the
top five critical steps plus the final output. **The prompt and the model are
identical in both conditions; the only difference is the trace content.** Both
conditions are blinded: `injected_fault` and `ground_truth` are stripped, and
fault labels are removed from the text.

## 3. Results

The headline is a negative result about this tool, and it leads.

All figures come from `gemini-3.5-flash-lite` for corpus generation and
`gemini-3.1-flash-lite` as the study judge, both on the free tier. Free-tier
prompts may be used to improve Google's products, so nothing confidential went
into this corpus, and every cost figure in this repository is **notional**:
computed from published list prices to show what the same token usage would have
cost, not money that was spent.

The `rag_qa` workflow was rewritten to version 2.0.0 partway through. The
original emitted 4 to 8 steps a run, which made the primary study untestable: a
top-5 extraction kept the whole trace, so the two conditions differed by
formatting rather than content. The deep corpus has a median of 20 steps in the
study population, where extraction removes 14.17 steps a run and 84.7% of input
tokens. Every number below is from the deep corpus unless it says otherwise.

### 3.1 Extraction prunes the causal step

**On traces of realistic depth the extraction engine removes the evidence needed
to locate the failure.** Full write-up in
[docs/finding-extraction-prunes-causal-steps.md](docs/finding-extraction-prunes-causal-steps.md).

| Measure | Shallow (median 4 steps) | Deep (median 20 steps) |
|---|---|---|
| Heuristic attribution | 11/31 (35.5%) | **0/24 (0.0%)** |
| Top-5 contains the causal step | top-5 was the whole trace | **0/24** |
| Median rank of the causal step | n/a | **13 of ~20** |
| Extracted condition, primary study | 19/31 (61.3%) | **0/24 (0.0%)** |

**Mechanism.** A `plan`, `retry`, `decision` or `critique` step contains
procedural text, and procedural text never resembles a final answer, so its
similarity to the final output is low by construction. Both weightings read that
low similarity as *divergence*, and so as criticality. Meanwhile the retrieval
that actually lost a chunk still returns three plausible passages, scores high
on evidence survival, and looks healthy. `divergent_branch` at weight 1.5
outranks `thin_retrieval` at 1.0, so on a typical run seven procedural steps tie
above the one rule designed to catch the fault.

Across all 24 runs' top-five slots the engine selected `plan` 48 times,
`revision` 22, `final` 22, `reasoning` 13, `critique` 9, `retry` 6, and
**`retrieval` zero times**. The event type carrying every fault in this
population never appears in the summary.

**This scales with trace length, which is why it was invisible before.**
Fault-presence signals are constant in trace length: one fault, one anomalous
step. Structural signals grow with it: more retrieval rounds mean more plans and
retries, more claims mean more critiques. The median deep run holds **7**
procedural steps against 1 faulted step, where a shallow run held 1 or 2. With a
fixed top-5 this is a ranking competition the causal step loses more decisively
the longer the trace. An extraction heuristic tuned on short traces can encode a
signal-to-noise ratio that only holds at that length, and no shallow evaluation
reveals it.

**Stated plainly: on realistic traces this engine is not neutral, it is
harmful.** A reader given the extracted summary is worse off than one given the
raw log, because the summary removed the step that explains the failure and
replaced it with the pipeline's own bookkeeping. No weight, threshold or signal
definition was changed in response to this result.

### 3.2 Fault potency

How often each injected fault actually changed the outcome. Direct measurement,
no inference and no judge: the fault either flipped the run to a failure or it
did not. It shapes everything downstream, because a fault that changes nothing
produces no failed run to attribute.

| Workflow | Fault type | Caused failure | Potency |
|---|---|---|---|
| rag_qa | dropped_retrieval | 12/12 | 100% |
| rag_qa | injected_contradiction | 11/11 | 100% |
| rag_qa | truncated_tool_result | **0/11** | **0%** |
| rag_qa | forced_false_rejection | **0/11** | **0%** |
| reviewer_pipeline | injected_contradiction | 1/10 | 10% |
| reviewer_pipeline | forced_false_rejection | 0/10 | 0% |
| reviewer_pipeline | truncated_tool_result | 0/9 | 0% |

**Two of the four fault types are now inert.** Both were weakened by the deeper
workflow, and neither was tuned back into potency.

`truncated_tool_result` fell from 13% to zero. The deep workflow has a real
tool_result step, so the fault now truncates the extraction tool's digest rather
than the retrieval context it used to hit. That is the semantically correct
target for a fault of this name, and it is also a redundant channel: the
answerer is given the digest *and* the raw passages, so it reads the value
straight out of the passages and the truncation changes nothing. Making it
potent again would mean removing the raw context from the answerer's prompt,
which would be redesigning the workflow to make a fault work.

`forced_false_rejection` is zero in both workflows for the same reason: the
revision step re-derives the same answer. The reviewer pipeline overall is close
to immune, at 1 failure from 29 injections, and the mechanism is visible in the
traces rather than inferred.

The consequence is that fault *type* matters far more than fault *presence*.
Retrieval and contradiction faults are certain to cause failure here; truncation
and forced rejection never do.

### 3.3 The label-leak effect

A methodological finding, and the one most likely to be useful to anyone else
building a fault-injected corpus.

The injected contradiction chunk was originally given the id
`injected-contradiction`. Renaming it to a neutral `doc-47` and regenerating the
corpus **dropped its potency from 15/15 to 13/15**, with nothing else changed.

| Injected chunk id | Potency |
|---|---|
| `injected-contradiction` | 15/15 (100%) |
| `doc-47` | 13/15 (87%) |

An agent that can see a chunk announcing itself as an injection treats it as
authoritative more readily than one that cannot. The label was not merely a
leak in the *measurement*; it changed the **behaviour being measured**. A corpus
built with self-describing fault markers therefore overstates how potent its
faults are, and no amount of downstream blinding recovers the difference,
because the agent already read the label at generation time.

The leak was also unrecoverable by string substitution. Blinding removes the
literal id, but an agent that read it can paraphrase it: one run's final answer
said *"corrected by the injection notice"*, which no literal substitution
catches. An earlier version of the primary study handled this by excluding three
of 33 runs after the fact and reported 63.3% against 63.3% at n=30. Excluding
contaminated runs was the wrong fix; not generating the label was the right one.
The exclusion guard remains in the code and `--include-compromised` disables it,
but it now excludes nothing.

**Practical rule: name injected artifacts exactly as ordinary ones are named,
and verify at generation time rather than at analysis time.**

### 3.4 Primary study

Population: all 24 failed runs carrying a known injected fault, with **no
exclusions**. Judge: `gemini-3.1-flash-lite`, temperature 0, identical prompt in
both conditions.

| Condition | n | correct | accuracy | mean input tokens | median input tokens |
|---|---|---|---|---|---|
| **raw_log** | 24 | 24 | **100.0%** | 9,204 | 9,459 |
| **extracted** | 24 | 0 | **0.0%** | 1,413 | 1,366 |

Input token reduction: **84.7%**. No judge response was unparseable in either
condition. All 24 discordant pairs favour the raw log:
**McNemar exact two-sided p = 1.19 × 10⁻⁷**.

Extraction now genuinely prunes, which is what makes the comparison meaningful
rather than a floor effect. On the study population it removes **14.17 steps** a
run, and **no run** is short enough for a top-5 to keep it whole:

| Measure | Shallow corpus | Deep corpus |
|---|---|---|
| Median steps per run (study population) | 4 | **20** |
| Runs where top-5 keeps everything | 18 of 31 (58%) | **0 of 24 (0%)** |
| Mean steps removed by extraction | 0.48 | **14.17** |
| Input token reduction | 49.8% | **84.7%** |

#### What the 100% does and does not mean

**It is not evidence that the judge comprehends a 20-step trace.** Correcting the
answer key (see 3.5) puts the causal step at the run's first retrieval, which
sits at a fixed position in the deep workflow: the correct answer is seq 3 on
**23 of the 24 runs**. The raw-log judge answered 3 on exactly those 23 and 0 on
the remaining one. A judge that always answered "3" would score 23/24, so this
result cannot be distinguished from positional luck. The label has almost no
variance, and the raw-log figure should be read as a ceiling artifact.

**The extracted 0% is not a label-variance artifact, and that asymmetry is the
finding.** The answer is trivially "3" and available to anything that can see
the first retrieval. The extracted condition still never finds it, because the
step is not in what it is shown: its top-5 contains the causal step in 0 of 24
runs. Its answers are scattered across 11 distinct step numbers, which is what
guessing looks like. See 3.1.

So the study does not establish that raw logs are readable. It establishes that
extraction removes the answer even when the answer is easy.

### 3.5 The answer key was mislabelled, and is corrected

Worth recording because it inverted the study's result and nothing failed while
it was wrong.

A fault acting on retrieved context is re-applied on every retrieval round, or a
later round re-retrieves from the full corpus and silently undoes it. Each
application produced its own record and **the last one won**, so the answer key
named the final retrieval while the error was introduced at the first.

| Condition | Against the old key | Against the corrected key |
|---|---|---|
| raw_log | 1/24 (4.2%) | 24/24 (100%) |
| extracted | 0/24 (0.0%) | 0/24 (0.0%) |
| heuristic attribution | 0/24 (0.0%) | 0/24 (0.0%) |
| McNemar | p = 1.000 | p = 1.19 × 10⁻⁷ |

The judges were reading the traces correctly and being marked wrong. On a
single-round workflow first and last are the same step, so no test caught it and
no number looked odd. The fix records the first application; the corpus was
corrected in place by
[harness/restamp_fault_labels.py](harness/restamp_fault_labels.py), which changed
one integer on 23 of 120 runs and left every step untouched, so the judge's
cached answers stayed valid and rescoring cost no quota. An invariant test now
asserts the recorded seq is the earliest application, checked both on freshly
generated runs and directly against the committed corpus.

### 3.6 Rejection outcomes

The repair / damage / no-change taxonomy applied to every critique step, 78
critiques across 60 runs:

| Population | n | repair | damage | no_change |
|---|---|---|---|---|
| All critiques | 226 | 0.9% | 0.4% | 98.7% |
| **Critiques that actually rejected** | **96** | **2.1%** | **1.0%** | **96.9%** |

The second row is the one that means anything. The specification defines
`no_change` to cover a critique with no following revision, which is correct as
written but lumps an *approving* reviewer together with one that objected and
was ignored.

Over actual rejections, 96.9% of criticism changed nothing: 93 of 96.

**This is directionally consistent with prior work, and n is too small to call
it more than that.** The author's earlier study found a reviewer whose apparent
safety was an artifact of the executor ignoring its critiques rather than of
good reviewing, and the direction here matches. But 96 rejections in a single
task family under a single model family is not enough to establish that, and the
deep `rag_qa` verifier is a lexical check rather than a model, so most of these
critiques are not the same object the prior study coded. Treated as a suggestive
observation.

### 3.7 Provenance

Across all 120 runs, **66 of 318 claims (20.8%) have no supporting step**,
affecting 50 of 120 runs. Support means an upstream step produced text with
cosine similarity ≥ 0.6 to the claim.

**Read this as a diagnostic of the tool, not as a property of the agents.** The
number says how often this implementation failed to link a sentence to a step,
which is not the same as how often an agent asserted something ungrounded. Three
reasons it cannot carry the stronger reading:

- The 0.6 similarity threshold was never validated against human judgement. It
  was chosen as a reasonable default and the figure moves with it.
- Sentence-level splitting is not proposition-level splitting, so a sentence
  carrying two assertions counts once, and a subordinate clause can be split off
  and stranded.
- Short text embeds badly. **26 of the 66 unsupported claims (39%) are under 40
  characters**, and the median unsupported claim is exactly 40 characters. A
  bare `ANSWER: 312` has little to embed and is easily scored unsupported even
  when the step above it produced the number.

A prior version of this figure was 120 of 388 (30.9%) and was inflated by
citation markup being counted as claims: fragments such as `Support: [doc-04#c2]`
are sentence-shaped but assert nothing. Eighteen of them cited a source while
being labelled unsupported, which is self-contradictory on its face. Those are
now discarded structurally rather than by threshold, which is why the current
figure is lower. That fix removed an obvious artifact; it did not validate what
remains.

### 3.8 Corpus

120 runs, 60 per workflow, 1,552 steps, 120,643 tokens. 74 runs (62%) carry an
injected fault, spread across all four types. 24 runs both failed and carry a
known fault, which is the evaluation population.

The two halves differ in depth: `rag_qa` is at version 2.0.0 and emits 16 to 24
steps a run, `reviewer_pipeline` is unchanged at 1.0.0 and emits 6 to 8. Overall
median is 16 and mean 12.9, but the evaluation population is 23 deep `rag_qa`
runs and 1 shallow `reviewer_pipeline` run, so its median is 20.

## 4. Reviewer leniency toward agent-authored code

**Collected, not yet labelled.** 207 review comments across 12 public
repositories and 75 pull requests are committed at
[pr_corpus/data/comments.jsonl](pr_corpus/data/comments.jsonl).

| Measure | Value |
|---|---|
| Comments collected | 207 |
| Repositories | 12 |
| Repositories containing both agent- and human-authored PRs | 7 |
| Comments in those repositories (the comparable set) | 159 |
| Agent-authored PRs / human-authored PRs | 142 / 65 comments |
| Agent authorship by bot account | 84 |
| Agent authorship by commit trailer | 58 |
| Comments labelled | **0** |

Agent authorship is established **only** by an identifiable bot account or a
commit trailer, never inferred from writing style, and every row records which
of the two was used. The comparison is within repository, so reviewer identity
and project norms are held constant; cross-repository comparison is explicitly
not performed.

The outcome table is not filled in because labelling is manual by design. The
repair / damage / no-change judgement *is* the taxonomy, and automating it would
measure a proxy for the thing under study rather than the thing itself. Run
`python pr_corpus/label.py` to label (the session is resumable) and
`python pr_corpus/label.py --report` to produce the comparison.

## 5. The attribution view

![Attribution view showing a correct prediction](docs/screenshot-attribution.png)

A failed `rag_qa` run with a `dropped_retrieval` fault. The banner shows
predicted against actual, the reasoning names the rule that fired, and the rule
weights are shown rather than hidden behind a single number.

Provenance, with one of three claims ungrounded:

![Provenance view](docs/screenshot-provenance.png)

## 6. Trace schema

The published contract is [schema/trace.schema.json](schema/trace.schema.json),
generated from the Pydantic models and validated against by every corpus file.

A **Run** has `run_id`, `workflow_type`, `workflow_version`, `task_input`,
`final_output`, `success`, `ground_truth`, `injected_fault`, timestamps, cost and
token totals, and a list of **Step**.

A **Step** has `step_id`, `run_id`, `parent_step_id`, `seq`, `agent_id`,
`agent_role`, `model`, `event_type`, `input`, `output`, `timestamp`,
`latency_ms`, token counts, `cost_usd`, `evidence_refs`, `error`, `retry_of` and
`rejection_outcome`.

`event_type` is one of `plan`, `tool_call`, `tool_result`, `retrieval`,
`reasoning`, `critique`, `revision`, `decision`, `error`, `retry`, `final`. The
vocabulary is deliberately small: larger would be more expressive but less
portable across frameworks, smaller would collapse distinctions the extraction
engine depends on. `critique` and `revision` are separate because a critique and
the revision it triggers can have different outcomes, which is the whole point of
the rejection taxonomy.

Six invariants are enforced in the models, so a malformed trace cannot enter
through corpus generation, an import adapter, or `POST /runs`:

1. `seq` values are contiguous from 0
2. `parent_step_id` references a step in the same run, or is null
3. `retry_of` references a step with a lower `seq`
4. `rejection_outcome` is non-null only on `critique` steps
5. `total_cost_usd` equals the sum of step costs within 1e-6
6. An injected fault targets a `seq` that exists in the run

## 7. Limitations

Stated plainly, worst first.

**Two of the four fault types are inert, so the effective population is 24 runs
across two fault types.** `truncated_tool_result` (0/11) and
`forced_false_rejection` (0/11) produced no failed runs in the deep workflow, so
the evaluation population is entirely `dropped_retrieval` (12) and
`injected_contradiction` (12), and 23 of the 24 come from one workflow.
**Per-fault-type claims are not supportable from this corpus**, and neither are
claims about "agent traces" in general: these are conclusions about retrieval
faults in one RAG pipeline.

**The answer key has almost no variance.** The corrected label puts the causal
step at the run's first retrieval, which sits at seq 3 in 23 of 24 runs. A judge
that always answered "3" scores 23/24, so the raw-log condition's 100% cannot be
distinguished from positional luck and is a ceiling artifact. The asymmetry with
the extracted condition (0/24 on the same easy answer) is what carries meaning;
the raw-log number on its own does not. A corpus that varied where the fault
lands would fix this and has not been built.

**Median 20 steps is still short relative to production agent traces.** It is
enough for extraction to prune (14.17 steps removed a run, 84.7% of tokens) and
therefore enough to ask the question, but real agent traces run to hundreds of
steps. Section 3.1 argues the extraction defect gets *worse* with length, so the
direction is known even though the magnitude at production scale is not.

**Single model family.** Corpus generation used `gemini-3.5-flash-lite` and the
judge was `gemini-3.1-flash-lite`. Both are Gemini tiers, so a judge evaluating
traces produced by a closely related model cannot be separated from a judge
evaluating traces produced by a *more or less capable* model. Any family-level
effect — shared failure modes, shared phrasing, shared blind spots — is
confounded with capability. A cross-family control (one Gemini, one non-Gemini)
is the obvious next step and has not been run.

**An LLM judge is not the reader the question is about.** The research question
concerns a human inspecting a trace. A judge that consumes 9,204 tokens without
effort has none of the length limits that motivate extraction, so this study
answers a different question than the one asked. The SRS anticipated
hand-checking a subsample against human readers; that has not been done.

**n = 24, single judge, single run, no variance estimate.** Temperature 0 gives
one sample per condition per run, so judge variance is unmeasured.

**Provenance (20.8%) is a tool diagnostic, not a property of the agents.** It
says how often this implementation failed to link a sentence to a step. The 0.6
similarity threshold was never validated against human judgement, and 39% of
unsupported claims are under 40 characters, where embedding comparison is
weakest. See 3.7.

**Claims are sentences, not propositions.** Semantic claim extraction would need
an LLM in a path that must stay deterministic, so sentence splitting is a
deliberate fallback. A sentence carrying two assertions counts once, and a
subordinate clause can be split off and stranded.

**Rejection outcomes (96.9% no-change, n = 96) are directionally consistent with
prior work and no more than that.** One task family, one model family, and the
deep `rag_qa` verifier is a lexical check rather than a model, so most of these
critiques are not the same object the earlier study coded.

**Injected faults are probably easier to find than natural ones.** They are
introduced at a known point by construction, and 3.3 shows the injection itself
can change agent behaviour. The corpus contains no naturally failing runs to
compare against.

**Gemini replaces the pinned `anthropic` SDK.** This is a documented deviation
from the specified stack, made on the repository owner's instruction. Free-tier
daily quotas are small and vary by model between 20 and 500 requests per day,
which constrains how often the corpus and study can be regenerated.

**All costs in this repository are notional.** Generation ran entirely on a free
tier and actual spend was zero. `cost_usd` figures are computed from published
list prices so that the cost view is not vacuous; total notional cost of the
committed corpus is $0.0135. No figure here represents money spent.

**Docker is verified.** `docker compose up --build` builds both images and
serves the committed corpus: backend 2.15GB, frontend 454MB, 120 runs loaded
from a read-only mount, and the frontend container reaching the backend over the
compose network. Running it for the first time exposed three defects that no
test had caught, which are described in the commit history: `/runs/{id}/export`
returned 500 on every run with a critique step because the grading rule lived in
a package that is not importable with `backend/` as the working directory; the
database path pointed at the read-only mount; and the image was pulling the CUDA
build of torch, several gigabytes of unreachable NVIDIA libraries, against the
specification's requirement that the embedding model run on CPU. Torch is now
installed from the CPU index and `torch.cuda.is_available()` is False in the
image.

**The PR corpus is unlabelled**, so the secondary study has no results yet. Agent
authorship detection is also imperfect: a PR written with agent assistance but
carrying neither a bot account nor a trailer is recorded as human.

**No participants, no private data.** All PR data is archival and public. QA
documents are synthetic. Nothing here required ethics approval.

## 8. Setup

Requires Python 3.11 and Node 20. `uv` is used if present and will provision
Python 3.11 itself.

```bash
make install
```

Serving the committed corpus needs no API key. To run the app:

```bash
make dev-backend     # http://localhost:8000
make dev-frontend    # http://localhost:5173
```

Or, on a machine with Docker:

```bash
make up
```

Run the tests (230 tests; 94.7% coverage on the extraction module):

```bash
make test
```

Recompute every published number that does not need an API call:

```bash
make reproduce
```

To regenerate the corpus or rerun the study, copy `.env.example` to `.env` and
add a `GEMINI_API_KEY` from [AI Studio](https://aistudio.google.com/apikey):

```bash
make corpus
make eval
```

Both are throttled for the free tier and checkpoint after every run, so a run
interrupted by a daily quota limit resumes where it stopped. The judge's answers
are committed alongside the summary in
[evaluation/results](evaluation/results), so `make reproduce` rebuilds the
published table from that record without spending quota; delete the trials file
to rejudge from scratch.

`make eval` pins the judge to the model named in the results table rather than
reading it from `.env`, and a study cut short by a quota wall is written to a
`_partial` file rather than over a complete one. Both guards exist because the
opposite happened: a rerun picked up a 20-per-day model from a local `.env`,
stopped at 11 of 31, and overwrote a finished study with the partial figures
while reporting success.

## Licence

MIT. See [LICENSE](LICENSE).
