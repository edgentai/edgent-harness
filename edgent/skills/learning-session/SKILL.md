---
name: learning-session
description: Running a timed learning session: recall from the spaced-repetition schedule, new material, practice, reflection, and what to record in the learner model.
---
# Learning session

Use this when the learner has a block of time and wants to study, or when the heartbeat says reviews are due.
A session has four parts in a fixed order, a time plan, and a record at the end. The learner model
(`learner/`) is both the input (what is due) and the output (what was shown).

## Before the session (one turn)

1. `learner_schedule`: the queue. Items with `reason: "due"` come first, most overdue at the top; items with
   `reason: "new"` are concepts the outline introduces that have never been studied, with the unit that
   introduces them.
2. `learner_summary`: streak, open misconceptions, the weakest concepts. Open misconceptions are reviewed in
   the recall part even when not due.
3. Agree the length and the goal in one line: "Thirty minutes: review three due concepts, then the next
   section of chapter 4." The learner can override the queue; say what you are skipping and that it will come
   back.
4. Write the plan with `todo_write`: one item per part, with its minutes.

## The four parts and the time plan

| Part | Share of the session | What happens |
|---|---|---|
| Recall | 20-30 percent | due concepts retrieved from memory before anything is re-read |
| New material | 30-40 percent | one unit's worth of new concepts, read or watched, with the picture |
| Practice | 25-35 percent | items at the level the schedule suggests; the learner does them |
| Reflection | 5-10 percent | the learner says what changed; the record is written |

Defaults by level: kg 10-15 minutes (recall as a game, new material read aloud, practice as one item,
reflection as "what did we learn"); primary 15-25; middle 25-40; secondary 40-60; undergraduate 60-90;
graduate 90-120; professional 30-60 and task-shaped. A session shorter than fifteen minutes is recall and
practice only.

Keep time visibly: announce each part, say the minutes at the start, and move on when they are up. A session
that runs over on new material and skips practice has taught nothing that will stay.

### Recall

- For each due concept, ask for retrieval first: "Without looking, how do you compute the fraction of
  positives who have it?" A question, not a re-explanation.
- Grade the answer honestly (quality 0-5, below): a blank is a 0; a correct answer after a struggle is a 3;
  immediate and correct is a 5. Say the grade and why in one line.
- A lapse (below 3) gets the picture again, briefly, from the chapter (cite the section), then one more
  retrieval attempt later in the session.
- Open misconceptions: ask the question that would trigger the misconception; if the learner's words show the
  correction, it resolves.

### New material

- One unit, or one section of one, from the queue's `new` items or the learner's choice; read it in the
  companion or watch the film; stop at the section's check question.
- Teach it the way `socratic-tutoring` says when they stall; do not lecture through the section they could
  read.
- Two or three retrieval questions on the new concept at the end of the part, so it enters the schedule with
  evidence.

### Practice

- Items from the `examiner` at the right difficulty: 2-3 for a lapsed concept, 3-4 for a due concept that
  recalled well, 2 for a new one (`assessment-design`). An existing set in `assessments/` is reused when it
  covers the concept.
- The learner answers; `assess_grade` scores and records `answered` events itself. Do not record them again.
- Review each wrong answer by the step that went wrong, not by restating the key.

### Reflection

- Ask: what is clearer now, what is still fuzzy, what would you teach someone from today? Listen for
  misconceptions in the answer.
- State what will be due next and when, in plain terms ("the grid method comes back in three days; eigenvalues
  in a week").
- Write the record (below) and close with `session_done`.

## What to record in the learner model

Through the `observer` subagent or `learner_update` directly, from evidence only:

- `reviewed` with quality for each due concept retrieved in Recall.
- `answered` with quality for the retrieval questions on new material (practice items are recorded by
  `assess_grade`).
- `explained` with quality when the learner taught something back.
- `misconception` in the learner's words when one surfaced; `misconception_resolved` when their words showed
  the correction.
- `unit_done` when a unit's reading and its check question were completed.
- `session_done` once, with the parts completed and the minutes in `detail`.

Quality scale (used by the schedule): 0 blank or wrong with no idea; 1 wrong but on the right track; 2 wrong,
recognised the answer when shown; 3 correct with serious effort or a hint; 4 correct with a slip or
hesitation; 5 correct, fluent. Below 3 is a lapse.

Record nothing for a part that did not happen. A session that ends early is recorded as what it was.

## Spaced-repetition rules

The schedule is computed by the learner store from the ledger (an SM-2 style rule); you do not compute
intervals, but you should know the shape so you can explain it and plan around it:

- A concept's interval grows with each successful review and resets on a lapse (quality below 3).
- The first successful review schedules a short interval (about a day); the next a few days; then roughly
  doubling, scaled by how easy the recalls have been.
- Due items are ordered most overdue first; new concepts are offered only after due ones, in outline order.
- A concept never studied has no interval; its first retrieval question in a session is what creates one.

Planning rules that follow:
- Never clear the whole queue in one session; six to ten due concepts is a full Recall part at undergraduate
  level, three to five at school level.
- Do not introduce new material when more than a dozen concepts are overdue; recall and practice only, and say
  why.
- A lapse today means the concept returns tomorrow: tell the learner so the return is expected, not a
  surprise.
- Streaks, points and badges are computed from the ledger; mention them when the learner's settings allow and
  never award anything yourself.

## A thirty-minute session, worked

Undergraduate learner; `learner_schedule` returns three due concepts (posterior from counts, effect of the
prior, conditional probability) and one new (independence, from chapter 4); one open misconception ("thinks
the catch rate is the chance of having it").

- Minutes 0 to 7, recall: "Without looking: ten in a thousand have it, the test catches nine in ten and flags
  five in a hundred. What fraction of positives have it?" Graded 4 (right, one hesitation). Then the
  misconception trigger: "Is ninety percent the chance you have it after a positive?" The learner says no and
  explains why: resolved. Conditional probability: a blank; graded 1; the grid shown again from the chapter
  section, cited; one more attempt later.
- Minutes 7 to 18, new material: section 4.1 (independence) read in the companion; stop at the check question;
  two retrieval questions: "Does the test's mistake on a healthy person depend on whether someone else is
  sick?" and "Say in one sentence what independent means here." Graded 4 and 3.
- Minutes 18 to 27, practice: four items from the `examiner` at difficulty 2 and 3 (two on conditional
  probability, one on the prior, one on independence); `assess_grade` records them. One wrong: reviewed by the
  step that failed (the split before the test).
- Minutes 27 to 30, reflection: the learner says the split is what they keep forgetting; the tutor says
  conditional probability returns tomorrow and independence in three days.

Recorded by the observer: `reviewed` x3 with qualities 4, 1 and (second attempt) 3; `misconception_resolved`;
`answered` x2 for the retrieval questions; `unit_done` for section 4.1; `session_done` with "recall, new,
practice, reflection; 30 min". The practice items were recorded by `assess_grade` and are not repeated.

## The heartbeat version

When the heartbeat wakes the agent with reviews due: report the count and the three most overdue concepts in
two lines, offer the session, and stop. Do not run a session nobody asked for.

## Anti-patterns

- Re-reading before retrieval: recall is only recall if the answer comes from memory.
- Skipping practice because new material ran long.
- Grading generously to keep the streak; the schedule then stops working.
- Recording events for parts that were skipped, or recording what the tutor explained as what the learner
  learned.
- Introducing a new unit while a dozen reviews are overdue.
- A reflection the tutor does for the learner.

## Checklist

- [ ] `learner_schedule` and `learner_summary` read; plan agreed with minutes; todo written.
- [ ] Recall: retrieval before re-reading; each due concept graded 0-5 with the reason.
- [ ] New material: one unit or section; check question reached; two or three retrieval questions asked.
- [ ] Practice: items at the right difficulty; graded by `assess_grade`; wrong answers reviewed by step.
- [ ] Reflection: the learner spoke; what is due next said in plain terms.
- [ ] Events recorded from evidence only; `session_done` once; nothing awarded by hand.
