---
name: mock-interview
description: Running a mock interview: technical, viva, behavioural or teach-back; setup, rounds with escalating follow-ups, hard stop, scoring by a separate critic, debrief, learner/ updates.
---
# Mock interview

Use this when the learner wants to be interviewed, examined orally, or to practise answering under pressure.
The interview is grounded in the workspace (a chapter, a thesis, a resume, a job description in `sources/`),
runs in rounds with a fixed structure, is scored by someone other than the interviewer, and ends in a debrief
the learner can act on.

## The four formats

| Format | What is being tested | Shape of a round | Evidence of a strong answer |
|---|---|---|---|
| Technical | can solve and explain a problem in the domain (code, maths, systems) | a problem, then follow-ups on correctness, complexity, edge cases, trade-offs; code runs in the sandbox | a working solution the candidate can defend and extend under a changed constraint |
| Oral exam or viva | can defend the content of a chapter, a thesis, a paper | a question on the material, then why, then what if, then a challenge to a claim | precise claims with their support, limits stated, a challenge met or honestly conceded |
| Behavioural | can describe past work as situation, task, action, result | a prompt ("tell me about a time"), then STAR follow-ups that pin down each part | specific actions the candidate took, a measurable result, what they would do differently |
| Teach-back | can explain a concept to someone who does not know it | "explain X to a first-year", then the interviewer plays the confused student | the one sentence, a picture, steps in order, a correct answer to a naive question |

The format is chosen in setup and does not change mid-interview. A viva on a thesis and a technical interview
for a role can be combined only as two separate interviews.

## Setup (one turn, before any question)

Agree, and write to the session's todo list:

1. Format and subject: the role, the chapter, the thesis, the paper. Name the sources the questions will rest
   on (`sources_list`; grep for the topics).
2. Level: what a strong answer looks like for this person (`learner_summary` for where they stand; `STUDY.md`
   for the target).
3. Length: number of rounds and the hard stop in minutes. Defaults: technical 3 rounds of 12 minutes; viva 4
   rounds of 8; behavioural 4 prompts of 7; teach-back 2 concepts of 10.
4. The rules, stated once: one question at a time; the interviewer follows up and does not hint; a hard stop
   at the time; scoring afterwards by a separate critic; the debrief names moments by quoting them.
5. The rubric: the rows from `references/rubrics.md` for the format, shown to the learner before the start. No
   hidden criteria.

## Rounds

The `interviewer` subagent (or you, in the main loop) prepares each round from the sources: the opening
question, a ladder of three follow-ups, what a strong answer contains, and the locators behind the question.

The ladder, used in order and only as far as the answer requires:

1. Clarify: "What do you mean by X?" or "Walk me through the step where Y happens." Pins the answer down;
   exposes vagueness.
2. Challenge: "Suppose the input is empty / the prior is zero / the sample is biased. What changes?" Tests
   whether the model behind the answer is real.
3. Extend: "Now do it under constraint Z" or "What is the weakest part of what you just said?" Separates
   strong from adequate.

Rules during a round:
- One question per turn. Wait for the answer before the next.
- Follow up on the answer actually given, quoting their words when pushing ("you said the chain is always
  ergodic; is it, when d equals one?").
- Never answer your own question, never hint, never correct during the round. A wrong answer gets a follow-up
  that gives it a chance to surface, then the round moves on.
- Technical: ask them to run the code; check the output with `run_python` when they paste it. Silence while
  they work is fine; a timer reminder at two minutes to go.
- Behavioural: STAR follow-ups: "What was the situation exactly?", "What was your task, not the team's?",
  "What did you do, first?", "What was the result, in numbers?", then "What would you do differently?"
- Teach-back: play the student honestly; ask the naive question a real student would ask at the step they
  skipped.
- Keep the private log: turn number, the learner's words, the rubric row it touches, why it will cost or earn
  points. This log is scoring input, not a verdict.

The hard stop is hard. At the agreed time, stop even mid-answer, thank them, and move to scoring. Running over
is a defect of the interviewer.

## Scoring: a separate critic

The interviewer does not score. Hand to a `critic` subagent, with no interviewer commentary beyond the log:

- the transcript of the rounds (verbatim, with turn numbers);
- the rubric rows for the format;
- the private log of moments;
- the model answers prepared in setup, with locators.

The critic returns, per rubric row, a score on the row's scale with the quoted moment that decided it, plus an
overall score and the three things to practise. The critic judges against the rubric and the level, not
against the interviewer's expectation; a moment the log flagged that the critic does not find decisive is
dropped.

## The debrief

Deliver the debrief as one artifact (a Markdown block in the reply, or a file the learner names when they want
to keep it),
in this order:

1. Headline: overall score and one sentence on the strongest and weakest rows.
2. The rubric table: row, score, the quoted moment, what a stronger answer would have contained.
3. Model answers: for each round, the answer a strong candidate gives, with locators into the sources so the
   learner can read the passage.
4. Three things to practise: specific, each with a way to practise it (an exercise set, a chapter section to
   re-read, a teach-back to do tomorrow). Not "improve clarity".
5. What goes into the learner model (below), stated so the learner knows.

Quote the learner's own words in every judgement that costs points. A debrief that says "your explanation of
ergodicity was weak" without the quote is not actionable; "you said 'the chain always converges', which is
false when d = 1" is.

## What to write to learner/

Through `learner_update`, after the debrief, with the critic's scores:

- `interview_done` once, with the format, the subject and the overall score in `detail`.
- `answered` per scored moment: the concept (short lower-case phrase), the quality 0-5 mapped from the row's
  score, the quoted moment as `detail`. Quality below 3 is a lapse and schedules a review.
- `misconception` for any answer that revealed a wrong model, in the learner's words; `misconception_resolved`
  when a later round showed the correction.
- `explained` for a teach-back, with the quality the critic gave it.

Weak answers therefore become due reviews and misconception bullets, and the next learning session starts
there. Nothing is written during the rounds; nothing is written without a critic score behind it.

## A technical round, worked

Role: backend engineer; subject: chapter 7 of the workspace's book (rate limiting); twelve minutes.

- Opening: "Design a function that decides whether a request is allowed under a limit of one hundred requests
  per minute per client. Say what you would store and why, then write it." (Rests on [notes-07:L40], the token
  bucket passage.)
- Candidate answers with a counter reset every minute. Clarify (rung 1): "What happens to a client that sends
  a hundred requests at the last second of one minute and a hundred at the first second of the next?"
  Candidate: "They get two hundred in two seconds." Logged: turn 3, quoted, row "Correctness", costs points.
- Challenge (rung 2): "Change the design so that cannot happen." Candidate proposes a sliding window with
  timestamps. Extend (rung 3): "A million clients. What does that cost in memory, and what would you do about
  it?" Candidate estimates and proposes a token bucket with two numbers per client. Logged: turn 6, earns
  points on "Complexity and trade-offs".
- Time call at two minutes to go; the candidate runs the code; the output matches the example. Hard stop at
  twelve minutes, mid-sentence, with thanks.

The critic later scores the round from the transcript and the log: Correctness 2 (the fixed-window flaw,
quoted), Reasoning 3, Complexity 4, Under change 3. The debrief quotes turn 3, gives the token-bucket model
answer with its locator, and sets one practice item: "explain the burst problem of fixed windows in two
sentences, tomorrow".

## Per domain

- Software engineering: one problem per round, growing in constraint; complexity and tests are rubric rows;
  the sandbox is the judge of "it works".
- Machine learning: a modelling question, then data, then evaluation, then failure modes; a viva on their own
  project is stronger than textbook questions.
- Mathematics or physics viva: state, prove or justify, then the limit case, then the counterexample.
- Consulting or case: structure first, then numbers, then the recommendation under a changed fact.
- Thesis viva: contribution, method, the weakest assumption, what the examiner would attack; questions rest on
  the thesis itself in `sources/text/`.

## Anti-patterns

- Teaching during the interview. The learner asked to be tested.
- Questions from memory of the subject instead of from the sources in the workspace.
- A round that drifts into conversation; a follow-up that is really a new question.
- Scoring by the interviewer, or scoring that cannot quote the moment.
- A debrief longer than the interview; practice advice that is not specific.
- Writing to the learner model on impression, or during the rounds.

## Checklist

- [ ] Setup agreed: format, subject with sources, level, rounds and hard stop, rules, rubric shown.
- [ ] Each round prepared from the sources with a ladder of three follow-ups and locators.
- [ ] One question per turn; follow-ups quote the answer; no hints, no answers, no teaching.
- [ ] Private log kept with turn numbers and quotes.
- [ ] Hard stop honoured.
- [ ] Critic scored from transcript, rubric, log and model answers; interviewer did not score.
- [ ] Debrief: headline, rubric table with quotes, model answers with locators, three specific things to
      practise.
- [ ] `learner_update` with `interview_done`, `answered` per moment, misconceptions in their words.
