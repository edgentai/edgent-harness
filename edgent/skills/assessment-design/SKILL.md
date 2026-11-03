---
name: assessment-design
description: Writing items a machine can key: kinds, stems, distractors from real misconceptions, difficulty, rubrics, and the items.json schema for assess_create and assess_items_check.
---
# Assessment design

Use this to write a quiz, an exercise set, an assignment or an exam, for a chapter's objectives or for a
person. An item is good when it tests the objective and nothing else, when its key can be checked by code
wherever the kind allows, and when a wrong answer tells you which misconception produced it.

## The tools and the files

- `assess_create` asks the examiner (the model, in a separate context) for a set matching the schema below,
  validates every item and drops the invalid ones with reasons. Arguments: `objectives` (required, one short
  sentence each), `n_items`, `level`, `kinds`, `concepts`, and the set id. You can also write the set yourself
  and reply with the JSON object when a brief asks for that.
- `assess_items_check` recomputes numeric keys, runs code assertions in the sandbox, checks regex keys against
  their own answers, and checks MCQ options for distinctness and exactly one correct. A clean check records
  evidence; a set with failing items is not finished.
- `assess_render` writes `assessments/<set-id>/rendered.html` (no scripts); `with_key: true` renders the key
  page, which the learner must not see before answering.
- `assess_grade` scores responses and records `answered` events in the learner model itself.
- On disk: `assessments/<set-id>/key.json` (everything), `items.json` (the learner view: no answers,
  solutions, tolerances or verification recipes), `rubric.md`, `rendered.html`.

## The items.json schema

The set:

```
{"set_id": "ch03-bayes", "title": "Bayes on a grid", "objectives": ["can compute a posterior with a frequency grid"],
 "level": "undergraduate", "items": [ ... ]}
```

One item (all of `id`, `kind`, `stem`, `answer`, `solution`, `concept`, `difficulty`, `verify` are required):

```
{"id": "q1", "kind": "numeric",
 "stem": "Of 1000 people, 10 have the condition. The test catches 9 of them and flags 50 of the others. What fraction of positives has the condition? Give a decimal to two places.",
 "answer": 0.15, "tolerance": 0.01,
 "solution": "Positives are 9 + 50 = 59; 9/59 = 0.1525... [textbook:p.12]",
 "concept": "posterior from counts", "difficulty": 2,
 "verify": {"type": "numeric", "expression": "9 / (9 + 50)"}}
```

Field rules:
- `kind`: `mcq`, `numeric`, `short`, `code` or `proof`.
- `options`: mcq only, 3 to 5 strings, distinct, plausible; `answer` is the integer index of the correct one.
- `answer`: numeric, a number; short and proof, text; code, the reference code; mcq, the index.
- `tolerance`: numeric only, absolute, default 0. Set it to what the rounding in the stem allows.
- `solution`: the worked solution a learner reads after answering; with a locator when the sources have it.
- `concept`: a short lower-case phrase, the same name the learner model uses.
- `difficulty`: integer 1 to 5 (below).
- `verify.type`: `numeric` with `expression` (a Python expression that evaluates to the answer; `math` is
  available); `code` with `code` (Python assertions run with `answer` bound to the submitted source as a
  string, the key's reference code at check time and the learner's code at grading: `exec` it into a
  namespace, take the function out, then assert); `regex` with `pattern` (full match against the trimmed
  response, case-insensitive); or `none` when nothing mechanical recognises a correct answer (then a rubric
  is required).

Keys must be computable wherever the kind allows: every numeric item has an expression; every code item has
assertions; a short item has a regex when a regex can recognise the answer (a term, a number with units, a
yes/no with a reason pattern), otherwise `none` and a rubric row.

## Item kinds and what each is for

| Kind | Tests | Keyed by | Use when |
|---|---|---|---|
| mcq | recognition, discrimination between models | index; options checked distinct | the misconceptions are known and can be written as options |
| numeric | a computation, an estimate | expression recomputed; tolerance | the objective is "can compute"; the answer is a number |
| short | a term, a one-line explanation, a classification | regex, or rubric | the answer is a word or a short phrase |
| code | can write or fix a program | assertions run in the sandbox over `answer` | the objective is "can implement"; the sandbox decides |
| proof | can derive, justify, argue | rubric (verify none) | the objective is analyse, evaluate or create |

Match the kind to the objective's verb (`course-design`): a Remember objective gets mcq or short; Apply gets
numeric or code; Analyse and above get proof or short-with-rubric. A proof objective tested by mcq is not
tested.

## Writing stems

- The stem asks one thing, states every given, and says the form of the answer ("to two decimal places", "as a
  fraction", "one word", "a function named f").
- The stem tests the objective, not reading: short sentences, no trick wording, no double negatives, no "which
  of the following is not false".
- Context from the sources or the persona's world; numbers chosen so the idea is visible and the arithmetic is
  not the test (unless arithmetic is the objective).
- Independent items: no item's stem gives away another's answer; no item depends on getting an earlier one
  right (unless the set is explicitly a chain and says so).
- A code stem names the function signature, the inputs' types and one example input and output; the assertions
  test the behaviour, not the style.
- Below secondary level, stems are read aloud by a parent or teacher: write them to be spoken (`narration`).

## Distractors from real misconceptions

An MCQ option is wrong for a reason a learner actually has. Build each distractor from a named misconception,
so a wrong choice is diagnostic:

- The common procedural error (forgot to split the population before applying the test: "9 of 1000").
- The confusion of the two conditionals ("90 percent", the test's catch rate).
- The misreading of the question ("50 of 59", the fraction of positives who are healthy).
- The plausible magnitude error (an answer off by a power of ten, when learners drop a factor).

Not distractors: options that are obviously absurd, options longer or more specific than the key, "all of the
above", two options that are the same claim worded twice. Open misconceptions in `learner/misconceptions.md`
are the first source of distractors for a personalised set.

Write the key with the same length and style as the distractors; vary the key's position across the set.

## Difficulty calibration

| Level | Meaning | Looks like |
|---|---|---|
| 1 | recall or recognition | state the definition; pick the term |
| 2 | apply the standard procedure to a standard case | the worked example with new numbers |
| 3 | apply with a twist | a missing given to derive first; an unusual form of the input |
| 4 | analyse: find the error, compare, trace | here is a wrong solution; where does it fail and why |
| 5 | transfer: a new context, combine two ideas, construct | a situation the chapter never showed; design the test |

A set for a chapter runs 2-3-3-1-1 across the levels (ten items). A set for a person: items at the level just
above the learner's last success on the concept, one below to warm up, one above to probe. The
`learner_summary` says where they stand; a lapsed concept gets difficulty 2 before anything harder.

## Rubrics for what code cannot key

For `proof`, and for `short` with `verify.type: none`, write `rubric.md` with rows the grader (a person, or
the model as a separate critic) scores:

```
## q7 (proof, difficulty 4)
- [2] States the claim with its conditions.
- [3] The key step: shows why the denominator counts all positives.
- [2] Handles the edge case d = 1.
- [1] Notation consistent with the chapter.
```

Points per row, what earns them in one line, total per item. A model answer per rubric item, with locators.
Rubric rows are about the content of the answer, never about length or effort.

## Answer keys that compute

- Numeric: write the expression from the givens in the stem, not the number; the check recomputes it. Set
  `tolerance` to the rounding the stem asks for. Try it with `run_python` first.
- Code: `answer` is a string holding the submitted source, so the recipe starts with
  `ns = {}; exec(answer, ns); f = ns["posterior"]` and asserts over `f`: the example in the stem, an edge case
  (empty, zero, negative, large) and a property. The sandbox runs it against the reference code at check time
  and against the learner's code at grading. Avoid assertions on printing or on timing.
- Regex: anchored by the check (full match), case-insensitive; allow the forms a correct answer takes
  (`^(0\.15|15 ?%|9/59)$`); test it against the key answer and against two wrong answers before you commit it.
- MCQ: exactly one correct; the check rejects two.

An item whose key failed the check is fixed or dropped; it never ships.

## Personalisation

When the set is for a person: `learner_summary` for the weakest concepts and open misconceptions; two or three
items per weak concept at difficulty 2-3, one transfer item on a strong concept, distractors from their own
misconceptions, stems in the domain `STUDY.md` names. Tell the learner what the set targets in one line.

## Anti-patterns

- Items that test the stem's vocabulary rather than the objective.
- "None of the above", "all of the above", absurd distractors, the longest option is the key.
- A numeric key typed as a number with no expression; a code key with no edge-case assertion.
- A proof item with no rubric; a rubric that rewards length.
- All items at difficulty 2; a set with no item above the learner's level.
- Reusing the chapter's worked example with the same numbers.
- Shipping a set whose check reported failures.

## Checklist

- [ ] Objectives listed; each item mapped to one; kinds match the verbs.
- [ ] Every item has id, kind, stem with the answer form stated, answer, solution (with locator when
      possible), concept, difficulty, verify.
- [ ] Numeric keys as expressions with tolerance; code keys that `exec` the submission, then assert the
      example, an edge and a property; regexes tested; MCQ with one correct.
- [ ] Distractors built from named misconceptions; keys and distractors matched in style; key position varied.
- [ ] Difficulty spread by the rule; personalised from `learner_summary` when for a person.
- [ ] `rubric.md` for every proof and unkeyed short item, with model answers.
- [ ] `assess_items_check` clean; rendered without the key for the learner.
