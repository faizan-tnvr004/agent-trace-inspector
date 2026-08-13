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
over mathematics tasks) and a `rag_qa` agent (retrieve → reason → answer over a
controlled document set). Both are LangGraph state machines. Faults of four
kinds were injected deliberately so that failure causes have ground truth:
dropped retrieval, truncated tool result, forced false rejection, and injected
contradiction. Each run records which fault was applied and at which step.

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

Ordered by how much the design can support, strongest first. The primary study
is fourth because, as run, it does not test the question it was built to answer;
that is explained in 3.4 rather than buried in the limitations.

All figures come from `gemini-3.5-flash-lite` for corpus generation and
`gemini-3.1-flash-lite` as the study judge, both on the free tier. Free-tier
prompts may be used to improve Google's products, so nothing confidential went
into this corpus, and every cost figure in this repository is **notional**:
computed from published list prices to show what the same token usage would have
cost, not money that was spent.

### 3.1 Fault potency

How often each injected fault actually changed the outcome. Direct measurement,
no inference and no judge: the fault either flipped the run to a failure or it
did not. This is the headline because it shapes everything downstream, and
because it is the one number here that nothing else is contingent on.

| Workflow | Fault type | Caused failure | Potency |
|---|---|---|---|
| rag_qa | dropped_retrieval | 15/15 | 100% |
| rag_qa | injected_contradiction | 13/15 | 87% |
| rag_qa | truncated_tool_result | 2/15 | 13% |
| reviewer_pipeline | injected_contradiction | 1/10 | 10% |
| reviewer_pipeline | forced_false_rejection | 0/10 | 0% |
| reviewer_pipeline | truncated_tool_result | 0/9 | 0% |

**The reviewer pipeline is close to immune to injected faults: 1 failure from 29
injections.** The mechanism is visible in the traces rather than inferred. The
reviewer catches the corrupted answer, the revision repairs it, and the run ends
correct. `forced_false_rejection` is the sharpest case: forcing the reviewer to
reject a correct answer produced zero failures in ten attempts, because the
revision step re-derived the same answer.

Two consequences worth stating. A reviewer-equipped workflow contributes almost
nothing to an injected-fault study, so the evaluation population is dominated by
one workflow and one fault family. And fault *type* matters far more than fault
*presence*: retrieval faults are near-certain to cause failure, truncation
faults rarely do, and the difference is an order of magnitude.

### 3.2 The label-leak effect

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

### 3.3 Heuristic attribution

The tool's own failure attribution, which uses no LLM at all. This is a clean
negative about the tool.

| Fault type | correct | accuracy |
|---|---|---|
| dropped_retrieval | 11/15 | 73.3% |
| injected_contradiction | 0/14 | 0.0% |
| truncated_tool_result | 0/2 | 0.0% |
| **overall** | **11/31** | **35.5%** |

Seven of the 31 produced no prediction at all, which the tool reports as "the
trace does not localise the cause" rather than guessing at step 0.

**The 0/14 on injected contradictions is a genuine gap in the design, not a
tuning problem.** The rule set scores steps that are missing, empty, thin or
erroring. A retrieval that returns a full set of plausible chunks, one of which
happens to contradict the others, is none of those things: it looks healthy by
every signal the engine has. Detecting it would need a contradiction signal
comparing retrieved chunks against each other, which does not exist in the
engine and was not added, because adding it after seeing the result would be
fitting the rules to the test set. The weights were fixed before the evaluation
ran and have not been touched since.

### 3.4 Primary study

Population: all 31 failed runs carrying a known injected fault, with **no
exclusions**. Judge: `gemini-3.1-flash-lite`, temperature 0, identical prompt in
both conditions.

| Condition | n | correct | accuracy | mean input tokens | median input tokens | mean latency |
|---|---|---|---|---|---|---|
| **raw_log** | 31 | 18 | **58.1%** | 2,879 | 2,774 | 1,177 ms |
| **extracted** | 31 | 19 | **61.3%** | 1,445 | 1,366 | 1,151 ms |

Input token reduction: **49.8%**. No judge response was unparseable in either
condition.

Agreement pattern: both conditions correct on 15 runs, both wrong on 9, 4 that
only extraction got and 3 that only the raw log got. Seven discordant pairs,
four to three: **McNemar exact two-sided p = 1.000**.

Per fault type:

| Fault type | raw_log | extracted |
|---|---|---|
| dropped_retrieval | 11/15 | 11/15 |
| injected_contradiction | 5/14 | 7/14 |
| truncated_tool_result | 2/2 | 1/2 |

#### The study as built does not test the stated hypothesis

The hypothesis is that pruning a long trace to its critical steps helps a reader
locate a failure. This corpus cannot test that, because **extraction barely
prunes anything**:

| Measure | Value |
|---|---|
| Median steps per run (study population) | 4 |
| Runs where the trace has ≤5 steps, so top-5 keeps everything | **18 of 31 (58%)** |
| Mean steps removed by extraction | **0.48** |

On more than half the population the two conditions contain **identical
content**, and across the whole population extraction removes under half a step
per run. Condition B is therefore not a subset of condition A; it is very largely
a reformatting of it, prose where condition A is JSON.

This is why the result is **not** reported as a null result. A null result means
the design could have detected an effect and did not. This design could not:
the conditions are near-identical by construction on a corpus whose median run is
four steps, so p = 1.000 reflects the absence of a manipulation rather than the
absence of an effect.

The 49.8% token reduction is real as a measurement but must not be read as
pruning. It is **roughly half the tokens but nearly all the content**, and the
saving comes mostly from prose-versus-JSON serialisation rather than from
dropping steps. Format and length vary together here and cannot be separated.

Fixing this requires a deeper corpus, not a different analysis. Tuning the
extraction weights until a difference appeared would be fitting to the outcome,
and was not done.

### 3.5 Rejection outcomes

The repair / damage / no-change taxonomy applied to every critique step, 78
critiques across 60 runs:

| Population | n | repair | damage | no_change |
|---|---|---|---|---|
| All critiques | 78 | 2.6% | 1.3% | 96.2% |
| **Critiques that actually rejected** | **19** | **10.5%** | **5.3%** | **84.2%** |

The second row is the one that means anything. The specification defines
`no_change` to cover a critique with no following revision, which is correct as
written but lumps an *approving* reviewer together with one that objected and
was ignored. Blended, the no-change rate is 96.2% simply because most critiques
are approvals.

Over actual rejections, 84.2% of criticism changed nothing: 16 of 19.

**This is directionally consistent with prior work, and n is too small to call a
reproduction.** The author's earlier study found a reviewer whose apparent safety
was an artifact of the executor ignoring its critiques rather than of good
reviewing, and the direction here matches. But 19 rejections in a single task
family under a single model is not enough to establish that, and one
reclassified critique moves the rate by five points. Treated as a suggestive
observation, not as evidence that the earlier finding generalises.

### 3.6 Provenance

Across all 120 runs, **89 of 335 claims (26.6%) have no supporting step**,
affecting 65 of 120 runs. Support means an upstream step produced text with
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
- Short text embeds badly. **25 of the 89 unsupported claims (28%) are under 40
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

### 3.7 Corpus

120 runs, 60 per workflow, 684 steps, 82,601 tokens. 74 runs (62%) carry an
injected fault, spread across all four types. 31 runs both failed and carry a
known fault, which is the evaluation population.

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

Stated plainly, worst first. The first one invalidates the primary study as a
test of the hypothesis; the rest bound what the other results mean.

**Traces are too short for the hypothesis to be testable.** The median run in the
study population is **4 steps** and the maximum is 8. Top-5 extraction therefore
keeps the entire trace on **18 of 31 runs (58%)** and removes **0.48 steps** per
run on average. The two conditions are near-identical by construction, so the
study measures serialisation format, not pruning. Everything else in 3.4 follows
from this. A corpus of 20 to 40 step runs is required before the question can be
asked at all, and no reanalysis of this corpus substitutes for it.

**The effective corpus is one workflow and one fault family.** Fault potency is
so skewed (3.1) that although the corpus holds 120 runs across two workflows and
four fault types, the evaluation population is 31 runs of which 30 are `rag_qa`
and 15 are `dropped_retrieval`. `forced_false_rejection` contributed zero failed
runs and is absent from every per-type breakdown despite being injected ten
times. Conclusions about "agent traces" are really conclusions about retrieval
faults in one RAG pipeline.

**Single model family.** Corpus generation used `gemini-3.5-flash-lite` and the
judge was `gemini-3.1-flash-lite`. Both are Gemini tiers, so a judge evaluating
traces produced by a closely related model cannot be separated from a judge
evaluating traces produced by a *more or less capable* model. Any family-level
effect — shared failure modes, shared phrasing, shared blind spots — is
confounded with capability. A cross-family control (one Gemini, one non-Gemini)
is the obvious next step and has not been run.

**An LLM judge is not the reader the question is about.** The research question
concerns a human inspecting a trace. A judge that consumes 2,879 tokens without
effort has none of the length limits that motivate extraction, so even a
well-powered version of this study would answer a different question than the one
asked. The SRS anticipated hand-checking a subsample against human readers; that
has not been done.

**n = 31, single judge, single run, no variance estimate.** Temperature 0 gives
one sample per condition per run, so judge variance is unmeasured. Seven
discordant pairs give McNemar exact p = 1.000. No significance is claimed.

**Claims are sentences, not propositions.** Semantic claim extraction would need
an LLM in a path that must stay deterministic, so sentence splitting is a
deliberate fallback. A sentence carrying two assertions counts once; a short
claim can be marked unsupported because short text embeds badly, and 28% of
unsupported claims are under 40 characters. The 0.6 similarity threshold was
never validated against human judgement. See 3.6.

**Injected faults are probably easier to find than natural ones.** They are
introduced at a known point by construction, and 3.2 shows the injection itself
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
