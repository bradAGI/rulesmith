# Rules

Search does not write graph JSON. It writes rules, which compile to the graph
language and render back from it, so the runtime, the validators and the saved
`plan.json` are unchanged: only the surface a proposer writes is different, and
you can edit what search wrote.

```
read target.x in [-1, 1]
ask threat = "Is this worth fighting?" [fight: "A live opponent is in view."] [clear: "Nothing is."]
when target.x and threat == "fight": ATTACK
always: MOVE_FORWARD
```

Rules are tried in order and the first whose condition holds decides. A judgment
can sit behind a cheap read, so the judge is asked only where a number cannot
settle it. Conditions join with `and` and `or`, group with parentheses, invert
with `not`, name a reading to test that it has a value, or compare one with `>=`,
`>`, `<=`, `<` or `==` against a number, a label, or another reading.

`let` names a condition so the rules can test it by name, or computes a number
from readings with `+`, `-` and a constant weight:

```
read ammo in [0, 50]
read health in [0, 200]
read enemy.health in [0, 200]
let armed = ammo >= 1 and health >= 25
let lead = health - enemy.health
when armed and lead >= 20: ATTACK
when not armed or health < enemy.health: MOVE_BACKWARD
always: MOVE_FORWARD
```

A reading can be named the same way, `read health = variables.HEALTH in [0, 200]`,
and the rules then say `health`. `read length of company_profile in [0, 10000]`
reads how long a text or list is, so `when length of company_profile < 20` can
settle an empty field without asking anyone. A name is a node in the graph, so it survives the
round trip: search reads and rewrites `armed`, not the condition spelled out in
every rule, and `draw` shows it as a box of its own.

Rules decide the label, and `output <name>:` starts another list that decides
something else from the same readings and judgments, asked once for both:

```
read severity in [0, 3]
read account.vip one of [yes, no]
when severity >= 2: review
always: accept
output priority:
when severity >= 3 or account.vip == "yes": high
always: low
```

Programs that rules cannot express, such as two judgments merged into one input,
are proposed as whole-graph JSON instead.

## What a judgment can be

A judge answers three kinds of question, and only the first is classification:

| Kind | Answer | Written as | Tested with |
| --- | --- | --- | --- |
| `choice` | one of the labels you name | `ask route = "Which team?" [billing: "..."] [technical: "..."]` | `route == "billing"` |
| `score` | a position on the rubric, ordered | `ask severity = "How bad is this?" [minor] [major] [critical]` | `severity >= 2` |
| `noul` | the probability its statement is true | `ask grounded = "The reply follows from the context."` | `grounded >= 0.85` |

A judgment reads the whole input unless you say what it is shown. `given` hands
it named values instead, which is less to read and keeps it on the question:

```
read health = variables.HEALTH in [0, 200]
read enemy.distance in [0, 4096]
ask close = "The enemy is close enough to fight." given [enemy.distance, health]
```

`given [observation, first]` shows it the whole input and another judgment's
answer, which is how one judgment checks another's work.

A judgment may carry a briefing, which the judge reads before the input:

```
ask kind = "What kind of answer does this question ask for?" [task labels] briefed "Classify the answer, not the subject. 'How deep' asks for a number; 'what color' asks for an entity, a thing with a name, not a description."
```

The briefing goes into what the judge is shown, ahead of the input, not into the
question. That is where a small judge takes guidance from: on TREC, the same
eleven thousand characters of label definitions moved nothing as question
instructions and lifted the direct question from 0.66 to 0.74 on validation as a
briefing. Search writes briefings where a judge confuses labels.

A choice over every label the task has need not list them.
`ask intent = "Which intent is this?" [task labels]` asks over the task's
labels, each described by the task's `descriptions` (`"descriptions":
{"card_arrival": "The customer is waiting for a card to arrive."}`) or, where
it has none, by its own words with underscores read as spaces. Search writes
such a question this way, so a proposer revising a 77-label question rewrites one
line, not 77. The form needs a task, so `rulesmith draw` and `run` refuse it.

Two things a rule can say about a judgment besides what it answered.

**How sure it was.** A choice or a score comes back with a confidence, and
`<name>.confidence` puts it where a condition can test it, so an answer the judge
hedged on falls through to the cheap rule instead of being acted on:

```
ask threat = "Fight or fall back?" [ATTACK: "Ammo and health allow it."] [MOVE_BACKWARD: "They do not."]
when threat.confidence >= 0.7 and threat == "ATTACK": ATTACK
always: MOVE_FORWARD
```

A threshold like that only means something if the judge's probabilities do: a
small model that says 0.97 and is right three times in five is overconfident.
`rulesmith calibrate task.json plan.rules --output calibrated.rules` fits one
temperature per choice question over the task's labels, from its answers on the
training examples, and writes it into the question (`... [ATTACK: "..."]
calibrated 1.8`). The answers stay the same; how sure they read changes,
`threat.confidence` included.

The line a rule draws on a judgment, the 0.7 in `threat.confidence >= 0.7`, is
a number the proposer wrote, and search only moves it by rewriting the rule and
paying examples to score the rewrite. `calibrate` fits it instead: for every
`>=` or `<` a rule draws on a probability, a score's expected level or a
confidence, and every threshold node on one, it tries each line between two
answers the judge actually gave on the training examples and keeps the one that
gets the most of them right. It fits on the task's `calibration` examples when
it has them, and on its training examples otherwise. A move is kept only when it also wins held out: the
fitting examples are dealt into five folds, the line is chosen without each
fold and scored on it, and those scores together must beat the line as written,
so one example on the far side of a line does not move it. The questions and
the words of the rules stay as they were; lines drawn on readings, `rating >=
4`, are what the rule says about the input and are not touched. Each line tried
is reported under `cutoffs`, with its old and new place and the training
accuracy before and after, and `validation` gives the accuracy on the
validation examples before and after the fit, which the fit never sees.

On the [triage graph](../examples/ticket-triage/graph.rules) against a
[hand-written task](../examples/ticket-triage/task.json) of forty tickets, with a local
Ruling judge:

```sh
uv run rulesmith calibrate examples/ticket-triage/task.json examples/ticket-triage/graph.rules \
  --backend ruling --output runs/triage.rules
```

```
when severity >= 2.5465: page_oncall calibrated 4/4 of 32
```

The judge never scores an outage a full 3 on the four-level scale; the six in
training land between 2.58 and 2.98, with two it misjudges lower, so the line at
3 paged nobody. Moved to 2.55, between the highest ticket labeled merely blocked
and the lowest outage, it pages four of the six, and training accuracy goes
from 0.67 to 0.83. Validation accuracy stays at 0.625: its two outages score
2.53 and 2.54, a hair under the line, because one blocked ticket in training
scored 2.51. That is the honest finding about this judge, not a failure of the
fit: on the score scale a 4B model does not separate blocked from outage by
more than a few hundredths, and no single line on it will.

The benchmark data gives the fit a larger test. A one-question graph over the
jobs postings, `ask scam = "Is this job posting a scam?"` and `when scam >=
0.5: fraudulent`, fitted on 200 training postings: the judge almost never says
scam, half the scams in the test rows get 0.04 or less, so the line moves to
0.057, and on 300 test postings accuracy goes from 0.59 to 0.64 (McNemar
p = 0.016), on 100 validation postings from 0.66 to 0.67. The same graph over
the clothing reviews keeps its line: the judge already puts recommended
reviews at 0.98 and the rest at 0.06, and 0.92 on test is what it was. The fit
corrects a judge's scale where it is off and leaves it where it is right; it
does not turn a direct question into the searched graph, which scores 0.79 to
0.82 on the seeds where it found its scam rule.

**Placing an escalation line.** Where a rule escalates on how sure the judge
was, as in [asking a larger judge](#asking-a-larger-judge-when-the-small-one-is-unsure),
accuracy alone always favors escalating more: the larger judge is never worse.
Give `calibrate` the prices search takes, `--call-cost`, `--judge-cost
NAME=COST` and `--extract-cost`, and each training example also pays for the
calls it made, so the line lands where escalating stops paying for itself:

```sh
uv run rulesmith calibrate task.json graph.rules --output calibrated.rules \
  --backend ruling \
  --judge large=ruling:mlx-community/Qwen3.6-35B-A3B-4bit@http://127.0.0.1:8012 \
  --judge-cost large=0.2
```

The report then records the prices under `prices`, and `train_before` and
`train_after` are training accuracy less what the calls cost. A line placed for
one price holds for that price; calibrate again when it changes.

## How sure a rule is

Every decision reports `decided_by`, the rule or judgment that settled it, and a
`confidence`: the least sure of the judgments asked and, once the rule is
measured, how often that rule is right. Until then a rule claims nothing, and a
decision it settles without asking reads as 1.0. `calibrate` also measures
every rule, on the task's `calibration` examples or without them on the labeled
training and validation examples, and writes the count after it:

```
when rating >= 4: recommended calibrated 37/40 of 100
```

That rule decided 40 of 100 examples and was right on 37, so its decisions read
as (37 + 1) / (40 + 2) = 0.90 sure. The smoothing keeps a rule seen twice from
claiming certainty. Training and validation examples are ones search saw, so
counts on them flatter a searched graph; a `calibration` split gives an honest
figure, and the report's `fitted_on` says which the counts came from.

`evaluate` checks the counts against the inputs it scores. A rule that decides
far more or less often than it did when measured (at `--drift`, default 0.01,
by an exact binomial test) is reported under `drift` in the output and warned
about. The inputs have moved, and so may how often the rule is right: a
scam-posting rule measured on balanced data decides 40% of postings, and at the
real scam rate 14%.

Where a rule over a field anyone can set is a way around the judge, as in fraud,
`optimize --keep-judge 0.99` keeps only graphs whose rules, wherever they answer
without the judge, are right at least that often; see
[optimization settings](optimization.md#settings).

## Second opinions and probabilities

**What several of them agree on.** A vote settles two or more judgments into
one value: `agree` reports whether they all said the same, which is how a second
opinion verifies the first, and `majority` takes the answer most of them gave.

```
ask first = "Is this order fraudulent?" [yes: "It is."] [no: "It is not."]
ask second = "Reading it again, is this order fraudulent?" [yes: "It is."] [no: "It is not."]
vote checked = agree of [first, second]
when checked == "yes" and first == "yes": review
always: accept
```

Each judgment in a vote is a separate model call, so a second opinion costs a
second question. Everything measured in this repo says calls are the expensive
part and rarely the deciding one, so reach for this where being wrong is costly,
not by default.

A probability is a number, so it composes like any read:

```
ask grounded = "The reply follows from the retrieved context."
when grounded >= 0.85: send
when grounded < 0.4: escalate
always: ask_for_review
```

That covers uses a classifier has no form for: gating an action on confidence,
verifying a generated answer before it goes out, ranking candidates by asking
which is better, or serving as the metric a search optimizes against rather than
the policy it searches. What a judge will not do is extract or compute: if the
answer is already a value in the input, read it.

## Testing a list's items

Inputs carry lists: an order's line items, a thread's messages, a customer's
past orders. A list is read as its items' fields, each declared with its domain,
and rules test whether any or all items meet a condition, or count those that do:

```
read order.items each {price in [0, 100000], status one of [paid, open]}
let big = count of order.items where price > 500
when any order.items has (price > 500 and status == "open"): review
when big >= 2: review
when all order.items have status == "paid": ship
always: hold
```

A compound test on an item goes in parentheses. An item that is not an object,
or lacks a field or holds one outside its domain, fails a test on that field,
the way a comparison with a missing reading is false. `all` over an empty list
holds, and a missing list leaves the rule to the next one. A count is a number,
so it can be compared, summed, or thresholded like any reading. Search is told
this syntax only when the task's inputs hold lists of records. Whether rules
over lists lift accuracy on a real task is not yet measured here.

## Asking a larger judge when the small one is unsure

A judgment may end `using <name>` to be asked of a named judge instead of the
graph's own. The graph names the judge; where it runs, `--judge` says which
model that name is. This is how a graph escalates: the small judge answers
first, rules settle its sure answers, and only the rest go to the larger one.

```
ask team = "Which team handles this?" [billing: "Charges."] [technical: "Bugs."]
when team.confidence >= 0.8 and team == "billing": billing
when team.confidence >= 0.8 and team == "technical": technical
always: ask "Which team handles this?" [billing: "Charges."] [technical: "Bugs."] using large
```

```sh
uv run rulesmith run graph.rules input.json --backend ruling \
  --judge large=ruling:mlx-community/Qwen3.6-35B-A3B-4bit@http://127.0.0.1:8012
```

One Ruling process can serve both judges. `RULING_MODELS` names a JSON file of
further models the server answers for by name, and a name served there is
the model in a `--judge` spec. With no `@URL`, a named Ruling judge goes to
`RULING_BASE_URL`, the same default server as the graph's own:

```json
{"large": {"model": "mlx-community/Qwen3.6-35B-A3B-4bit"}}
```

```sh
RULING_MODELS=models.json uv run --directory ../ruling ruling serve
uv run rulesmith run graph.rules input.json --backend ruling --judge large=ruling:large
```

An entry may also name an `adapter` and its `calibration`, and adapters on one
base share a single copy of its weights. `GET /v1/models` lists each name with
any temperature the server already applies to its answers.

A judgment kept as a fallback is asked only when every rule ahead of it has
declined, so the larger judge costs nothing on the inputs the small one settles.
Each call in a result and a scorecard names the judge it went to, so how often
a graph escalated is counted, not guessed. In search, `--judge-cost large=0.2`
prices a question to that judge, and the proposer is told which judges it may
use and what each costs. `calibrate` takes the same prices to place the
escalation line where escalating pays; see
[what a judgment can be](#what-a-judgment-can-be).

The small judge is worth most when it agrees with the large one. `rulesmith
distill` trains it on the large judge's answers to the task's inputs, labeled
or not: the teacher is asked the task's direct question, and its probabilities,
doubt included, become the student's targets through Ruling's trainer, with
the test split held out as `tune` holds it:

```sh
uv run rulesmith distill task.json --output runs/student \
  --teacher ruling:mlx-community/Qwen3.6-35B-A3B-4bit@http://127.0.0.1:8012
```

Serve the adapter it reports as the graph's own judge, and the large one
answers only what the student is unsure of.

Measured on TREC, with the tuned 4B escalating to Clef-flash, 9% of questions
reached the larger judge and accuracy went from 0.884 to 0.920, half the gap to
Clef-flash alone; the full result, and where it does not pay, is under
[escalating to a larger judge](benchmarks.md#escalating-to-a-larger-judge).

[Ruling](https://github.com/bradAGI/ruling) can also escalate on its own, with
`RULING_CASCADE_TO`: every question below one confidence goes to a second
model, inside the server. Rules do the same per judgment, with a cutoff each
rule can draw, and the scorecard shows which judge answered.

## Reading fields out of free text

A judge answers a question; it does not pull a value out of text. When the
input is a message rather than a record, an extraction has a generative model
read it into a few typed fields, and the rules test those fields as they would
an input's own:

```
extract claim = "Read the refund request." {amount in [0, 10000], refund_asked one of [yes, no]}
when claim.refund_asked == "yes" and claim.amount < 50: auto_refund
always: review
```

Each field declares its domain the way a reading does. A field the model cannot
fill, fills with something that is not a number, or fills outside its domain is
absent, like a missing path, so the rules fall through to the next one. An
extraction may end `given [...]` and `when <condition>` as a judgment may.

```sh
uv run rulesmith run graph.rules input.json \
  --extractor-model openai/qwen3.8-27b --extractor-base-url http://127.0.0.1:8080/v1
```

The extractor is any model DSPy reaches, asked through a small typed DSPy
program; each record is read once per run, and each call in a result names the
fields it filled and what it cost. In search, `--extract-cost` prices an
extraction, and the proposer is told the syntax only when an extractor is given.
Whether extraction lifts accuracy on free text is not yet measured here.

## Worked examples

One worked example of each, runnable from [examples](../examples), where each
decision has a folder of its own:

**Route a ticket** ([support-routing](../examples/support-routing/graph.rules)) — a choice is
a label, and the rules act on it:

```
# choice: pick one of the labels you named, the classic classification.
ask team = "Which team should answer this ticket?" [billing: "Charges, invoices, refunds."] [technical: "Errors, outages, anything broken."] [sales: "Pricing, plans, buying more."]
when team == "billing": billing
when team == "technical": technical
always: sales
```

**Triage by how blocked the customer is**
([ticket-triage](../examples/ticket-triage/graph.rules)) — a score is ordered, so one
judgment feeds several thresholds, and a read decides who jumps the queue:

```
# score: a position on a rubric, so it can be compared and thresholded.
read variables.customer_tier in [0, 3]
ask severity = "How badly is this customer blocked?" [annoyance] [degraded] [blocked] [outage]
when severity >= 3: page_oncall
when severity >= 2 and variables.customer_tier >= 2: escalate
when severity >= 2: queue_urgent
always: queue_normal
```

<details>
<summary>Drawn with <code>rulesmith draw</code></summary>

```mermaid
flowchart TD
    n0[/"read variables.customer_tier"/]
    n1{{"judge: severity<br/>score: How badly is this customer blocked?"}}
    n2(("page_oncall"))
    n3(("escalate"))
    n4(("queue_urgent"))
    n5(("queue_normal"))
    n6(["decision<br/>first that answers"])
    c0["severity >= 3"]
    n1 --> c0
    c0 -.-> n2
    c1["severity >= 2 and variables.customer_tier >= 2"]
    n0 --> c1
    n1 --> c1
    c1 -.-> n3
    c2["severity >= 2"]
    n1 --> c2
    c2 -.-> n4
    n2 -->|"1"| n6
    n3 -->|"2"| n6
    n4 -->|"3"| n6
    n5 -->|"4"| n6
    o0(("label"))
    n6 ==> o0
    classDef judge fill:#fcd34d,stroke:#b45309,stroke-width:2px,color:#1c1917
    class n1 judge
    classDef read fill:#bfdbfe,stroke:#1d4ed8,color:#1c1917
    class n0 read
    classDef rule fill:#bbf7d0,stroke:#15803d,color:#1c1917
    class n2,n3,n4,n5 rule
    classDef condition fill:#e5e7eb,stroke:#6b7280,color:#1c1917
    class c0,c1,c2 condition
```

</details>

**Spend the judge only where it decides something**
([order-review](../examples/order-review/graph.rules)) — readings settle the clear cases,
and the question is asked only about orders big enough to matter:

```
# Cheap readings decide the clear cases; the judge is asked only about the rest.
read order.total in [0, 1000000]
read order.country one of [domestic, eu, rest]
read customer.orders in [0, 10000]
ask suspicious = "This order reads like card testing or resale fraud." when order.total >= 500
when customer.orders >= 5 and order.total < 500: ship
when order.total < 50: ship
when suspicious >= 0.7: hold_for_review
when suspicious >= 0.3 and order.country == "rest": hold_for_review
always: ship
```

<details>
<summary>Drawn with <code>rulesmith draw</code></summary>

```mermaid
flowchart TD
    n0[/"read order.total"/]
    n1[/"read order.country"/]
    n2[/"read customer.orders"/]
    n3{{"judge: suspicious<br/>noul: This order reads like card testing or resale..."}}
    n4(("ship"))
    n5(("ship"))
    n6(("hold_for_review"))
    n7(("hold_for_review"))
    n8(("ship"))
    n9(["decision<br/>first that answers"])
    c0["order.total >= 500"]
    n0 --> c0
    c0 -.-> n3
    c1["customer.orders >= 5 and order.total < 500"]
    n2 --> c1
    n0 --> c1
    c1 -.-> n4
    c2["order.total < 50"]
    n0 --> c2
    c2 -.-> n5
    c3["suspicious >= 0.7"]
    n3 --> c3
    c3 -.-> n6
    c4["suspicious >= 0.3 and order.country == rest"]
    n1 --> c4
    n3 --> c4
    c4 -.-> n7
    n4 -->|"1"| n9
    n5 -->|"2"| n9
    n6 -->|"3"| n9
    n7 -->|"4"| n9
    n8 -->|"5"| n9
    o0(("label"))
    n9 ==> o0
    classDef judge fill:#fcd34d,stroke:#b45309,stroke-width:2px,color:#1c1917
    class n3 judge
    classDef read fill:#bfdbfe,stroke:#1d4ed8,color:#1c1917
    class n0,n1,n2 read
    classDef rule fill:#bbf7d0,stroke:#15803d,color:#1c1917
    class n4,n5,n6,n7,n8 rule
    classDef condition fill:#e5e7eb,stroke:#6b7280,color:#1c1917
    class c0,c1,c2,c3,c4 condition
```

</details>

**Check a draft reply before sending it**
([reply-check](../examples/reply-check/graph.rules)) — probabilities compose, so
two judgments set a floor and a ceiling with a human in between:

```
# noul: a probability, so the decision is a threshold on confidence.
ask grounded = "Every claim in the draft reply follows from the retrieved context."
ask harmful = "The draft reply would embarrass us if a customer read it."
when harmful >= 0.2: rewrite
when grounded >= 0.85: send
when grounded < 0.4: escalate
always: ask_for_review
```

<details>
<summary>Drawn with <code>rulesmith draw</code></summary>

```mermaid
flowchart TD
    n0{{"judge: grounded<br/>noul: Every claim in the draft reply follows from..."}}
    n1{{"judge: harmful<br/>noul: The draft reply would embarrass us if a..."}}
    n2(("rewrite"))
    n3(("send"))
    n4(("escalate"))
    n5(("ask_for_review"))
    n6(["decision<br/>first that answers"])
    c0["harmful >= 0.2"]
    n1 --> c0
    c0 -.-> n2
    c1["grounded >= 0.85"]
    n0 --> c1
    c1 -.-> n3
    c2["grounded < 0.4"]
    n0 --> c2
    c2 -.-> n4
    n2 -->|"1"| n6
    n3 -->|"2"| n6
    n4 -->|"3"| n6
    n5 -->|"4"| n6
    o0(("label"))
    n6 ==> o0
    classDef judge fill:#fcd34d,stroke:#b45309,stroke-width:2px,color:#1c1917
    class n0,n1 judge
    classDef rule fill:#bbf7d0,stroke:#15803d,color:#1c1917
    class n2,n3,n4,n5 rule
    classDef condition fill:#e5e7eb,stroke:#6b7280,color:#1c1917
    class c0,c1,c2 condition
```

</details>

## Beyond rules

The full graph language is available. This hand-written example
([examples/support-routing/dag.json](../examples/support-routing/dag.json)) asks two questions in parallel, then
runs only the specialist that matches. Drawn with `rulesmith draw`:

```mermaid
flowchart TD
    n0{{"judge: department<br/>choice: Which department should handle the primary..."}}
    n1{{"judge: urgency<br/>score: Assess how urgently the ticket needs a response."}}
    n2["context<br/>merge"]
    n3{{"judge: billing_review<br/>noul: Does this ticket describe a duplicate or..."}}
    n4{{"judge: technical_review<br/>noul: Does this ticket describe a reproducible..."}}
    n5{{"judge: sales_review<br/>noul: Does this prospective customer describe a..."}}
    n6(["specialist_result<br/>first that answers"])
    n7["assessment<br/>merge"]
    n0 --> n2
    n1 --> n2
    n2 --> n3
    c0["department == billing"]
    n0 --> c0
    c0 -.-> n3
    n2 --> n4
    c1["department == technical"]
    n0 --> c1
    c1 -.-> n4
    n2 --> n5
    c2["department == sales"]
    n0 --> c2
    c2 -.-> n5
    n3 -->|"1"| n6
    n4 -->|"2"| n6
    n5 -->|"3"| n6
    n0 --> n7
    n6 --> n7
    n1 --> n7
    o0(("label"))
    n0 ==> o0
    o1(("assessment"))
    n7 ==> o1
    classDef judge fill:#fcd34d,stroke:#b45309,stroke-width:2px,color:#1c1917
    class n0,n1,n3,n4,n5 judge
    classDef condition fill:#e5e7eb,stroke:#6b7280,color:#1c1917
    class c0,c1,c2 condition
```

The amber hexagons are judge calls, each a question to Jev or Ruling; the rest
runs without a model. Solid arrows carry values, a gray box is a condition, and
a dashed arrow leads from a condition to what it gates: each review is shown the
merged `context`, and asked only when `department` answered its team. The two
judgments nothing feeds are shown the whole ticket. `specialist_result` takes the first of its
three inputs that answered, in the numbered order, though here at most one ever
does. Thick arrows lead to the graph's two outputs.

## How search revises them

```mermaid
flowchart LR
  T["task.json<br/>labels + examples"] --> S["Starting graph<br/>yours, or the majority label"]
  S --> E["Run on training examples<br/>via Ruling or Jev"]
  E -->|"errors + expected labels"| P["DSPy proposer<br/>revises one part of the graph"]
  P -->|"assemble + schema check"| E
  E -->|"better on the batch"| V["Score on validation examples"]
  V --> B["Best graph<br/>plan.json"]
```

The proposer sees where the graph answered wrong and revises one part: the
readings it may look at, or the rules that decide. The revision is compiled with
the part it did not touch and validated as a complete graph. A proposal is kept
only if it beats its parent on the training batch; the best validation score
wins. The test split never reaches the optimizer.

## Reference

Every construct in one place, and a program that uses most of them:

```
read variables.HEALTH in [0, 200]
read compass.2.bearing in [-180, 180]
ask threat = "Fight or fall back?" [fight: "Ammo and health allow it."] [flee: "They do not."] when variables.HEALTH < 40
when variables.HEALTH >= 25 and threat == "fight": ATTACK
when compass.2.bearing >= 45: TURN_LEFT
always: MOVE_FORWARD
```

- **Readings** declare what the program may look at: `read <path> in [low, high]`,
  `read <path> one of [a, b]`, `read length of <path> in [low, high]` for how long a text or list is (tested as
  `length of <path> < 20`), `read <name> = <path> in [low, high]` to give a
  reading a name the rules use instead of its path, or `ask <name> = "question" [label: "when it applies"] ...`
  for a judgment, which may end `when <condition>` to be asked only then. A judgment
  may carry `briefed "..."`, guidance the judge reads before the input; it
  reads the whole input unless it names what it is shown with `given [a, b]`;
  `observation` in that list is the whole input. A judgment ending `using <name>`
  is asked of that named judge, which `--judge` gives a model where the graph
  runs. `[task labels]` in place of the
  labels, as in `ask intent = "Which intent is this?" [task labels]`, asks over
  every task label as the task's `descriptions` describe it. A choice over exactly
  those, in the task's order, is shown to the proposer in this short form.
- **Lists** are read as their items' fields, `read <path> each {field in [low, high], field one of [a, b]}`,
  and tested with `any <path> has <test>` or `all <path> have <test>`, or counted with
  `let n = count of <path> where <test>`.
- **Extractions** read free text into fields:
  `extract <name> = "what to read" {field in [low, high], field one of [a, b]}`,
  optionally then `given [...]` and `when <condition>`. Rules test a field as
  `<name>.<field>`, and a field the model could not fill within its domain is absent.
- **Rules** are `when <condition>: <action>` or `always: <action>`, tried in order.
  An action is a label, a conditional such as
  `route.bearing >= 0 ? TURN_LEFT : TURN_RIGHT`, or an ask without a name, whose
  answer is the label: `always: ask "Fight or fall back?" [ATTACK: "Ammo and health allow it."] [MOVE_BACKWARD: "They do not."]`.
  When no rule fires, the output has no value and the run fails rather than
  guess. `rulesmith validate` lists any output an input could leave that way
  under `unanswered`; a list ending on `always:` never does. A comparison with a
  reading the input lacks is false, so `when not (...)` over every rule above it
  catches what they missed, while a rule list of comparisons alone misses
  inputs without the reading, and values that fall between its cutoffs.
- **Conditions** join with `and` and `or`, group with parentheses, invert with `not`,
  name a reading to test that it has a value, or compare one with `>=`, `>`, `<=`, `<`
  or `==`, against a number, a quoted label, or another reading
  (`capture.best_gain > hanging.worst`). The graph compares with three of those;
  `<=` and `>` compile to the same thing said longhand. `threat.confidence` is how
  sure the judgment `threat` was.
- **Names** stand for a condition the rules test more than once:
  `let armed = ammo >= 1 and health >= 25`, then `when armed: ATTACK`. A name is
  declared before the rules that use it and cannot reuse a reading's or judgment's
  name. It compiles to a node, so it comes back as `armed` when the graph is
  rendered, not as the condition written out again. `let` also computes a number:
  `let margin = 2 * gain - worst + 1` adds, subtracts and scales readings.
- **Votes** settle several choice judgments into one value:
  `vote verdict = agree of [first, second]` or `vote verdict = majority of [a, b, c]`.
- **Outputs** beyond the label each get a rule list of their own, after the
  label's, under an `output <name>:` heading. Search scores the label only.
- **Measurements** are what `rulesmith calibrate` writes and search never invents.
  A judgment may end `calibrated 1.8`, the temperature fitted to how sure it
  should read; a rule's line on a judgment may have been moved to where the
  labeled examples put it; a rule may end `calibrated 37/40 of 100`, how many of the labeled
  examples it decided and how many rightly. Search drops a starting graph's rule
  measurements, since they describe that graph and not its rewrites, and the
  proposer is told to keep a temperature as written.
- The two parts, readings and rules, are revised separately and may be merged
  between candidates. A rules part may declare any reading its rules need.
- A program beyond rules is proposed as whole-graph JSON, which is accepted as the
  complete program.

Why: in past runs, two fifths of proposals against a 36-node graph scored zero, not because the
graphs played badly but because the JSON came back malformed. Every searched graph
had the same shape anyway, so that shape became the surface.

See [Graph format](graphs.md) for the JSON behind each node, and
[Optimization](optimization.md) for how search runs and what it reports.
