---
name: explaining
description: The craft of explaining one idea: the one sentence, the picture, the load-bearing steps, the curse-of-knowledge check, earned abstraction, the napkin test. Load before explaining.
---
# Explaining

You already know the subject. The risk is not ignorance; it is the curse of knowledge: what is obvious to you
is invisible to the reader, and you skip it without noticing. This skill is the procedure that catches the
skip.

## The procedure

Do these in order, in writing (a scratch list is fine), before the explanation is drafted.

### 1. The one sentence

Write the idea as one plain sentence the reader could repeat tomorrow. No symbols, no jargon the reader does
not already own, one verb.

- Bad: "Bayes' theorem relates conditional probabilities through the prior and the likelihood."
- Good: "A test result moves your belief by an amount that depends on how rare the thing is to begin with."

If you cannot write the sentence, you have not found the idea yet; you have found the topic. Keep going until
the sentence is about one thing.

### 2. The picture

Decide what the reader should see in their head. A picture is a concrete object with parts: a grid of a
thousand people, a lever, a queue, two columns of numbers, a map with one road. The picture is chosen so the
idea becomes a fact about the picture.

Checklist for a picture:
- It has at most three kinds of thing in it.
- The idea is a visible relationship between the things (bigger than, inside, moves to, splits into).
- A reader could draw it in a minute from your description.
- It does not require the formalism to understand; the formalism will later describe it.

If the idea is a process (something changes), the picture moves and may deserve a film; if it is a
relationship, a diagram; if it is a quantity the reader should push on, a widget. See `course-design` for that
decision.

### 3. The load-bearing steps

List the steps the reader must take from what they know to the idea. Three to six. Each step is one new thing.
For each, write what the reader would say if they did not get it ("why does that follow?") and answer it in
one line.

- A step that needs two new things is two steps.
- A step the reader cannot check (a fact asserted) must carry its support: an example, a citation with a
  locator, or a computation they can repeat.
- Order the steps so each uses only what came before; a step that needs a later step is a forward reference
  and a defect.

### 4. The curse-of-knowledge check

Read the steps as the persona in `STUDY.md`: their vocabulary, their last course, what they have never seen.
For every noun and every symbol ask: has this reader met it, in this text, before this line? Mark every miss.

Then fix each miss in one of three ways: introduce it (one line, with an example), replace it with a word they
own, or cut it. Do not fix a miss by adding a parenthesis of jargon.

Common skips to look for:
- A symbol used before it is read aloud ("the posterior" before "what you believe after the test").
- A convention assumed (indices start at zero; probabilities are between zero and one; the x-axis is time).
- A size assumption (the reader does not know a million is big relative to a thousand here).
- A step done in your head: "simplifying" means three algebra moves the reader has to see once.
- A word with a technical meaning that differs from its plain meaning (significant, normal, bias, work, force,
  random).

### 5. Earned abstraction

The general statement, the symbol, the theorem, the definition: each arrives only after the reader has seen
two concrete instances of it and can feel the need for a name. The formula is the last thing on the page, and
when it arrives, every symbol in it points back to a thing in the picture.

- Example before definition. Two examples before a general rule. Rule before notation.
- When introducing notation, say what each symbol is in the picture, in the same sentence.
- When an abstraction is not needed for what the reader will do, do not introduce it. Name the omission
  ("there is a general version; you do not need it here").

### 6. The napkin test

Could you explain this on a napkin to the persona in five minutes, drawing the picture and walking the steps,
and would they be able to do the next problem? If the explanation needs a page of setup before the picture
appears, the setup is in the wrong place or the idea is two ideas.

### 7. End with what the reader can now do

The last lines of an explanation say what the reader can now do that they could not before, and give them one
thing to try that uses it. Not a summary of what was said; a capability.

## A worked example: Bayes' theorem for a reader who knows fractions

Persona: a strong secondary-school reader or first-year undergraduate; comfortable with fractions and
percentages; has never seen conditional probability notation.

**The one sentence.** A positive test tells you less than you think when the condition is rare, because most
positives come from the many healthy people who test positive by mistake.

**The picture.** A grid of 1000 people. One in a hundred has the condition: 10 people, coloured. The test
catches 9 of the 10 (it misses one). Of the 990 healthy people, the test wrongly flags 5 per hundred: about
50. So the positives are 9 coloured and 50 plain, 59 in all.

**The load-bearing steps.**
1. Start with a population you can count, not a probability: 1000 people. (Why 1000? So every number is a
   whole person.)
2. Split by the condition first: 10 with, 990 without. (The reader must see that the split is lopsided before
   anything about the test.)
3. Apply the test to each group separately: 9 of the 10 flagged; 50 of the 990 flagged. (Two separate counts;
   the mistake every reader makes is to think about the test's accuracy as one number.)
4. Collect the positives: 59. Ask what fraction of them has the condition: 9 of 59, about 15 percent. (The
   idea arrives here, as a fact about the picture.)
5. Change one thing and watch: if 1 in 10 had the condition instead of 1 in 100, the positives would be 90
   coloured and 45 plain, and the fraction jumps to two thirds. (The sentence from step 1 becomes visible:
   rarity is what drags the fraction down.)

**The curse-of-knowledge check.** Misses found on a first pass: "false positive rate" (replace with "flags 5
in 100 healthy people"); "prior" (not needed; cut); "sensitivity" (introduce only after step 3, as the name
for "catches 9 of 10"); the assumption that the two mistakes of the test are independent of each other (state
it once as "the test does not know who is sick"). The symbol P(A|B) does not appear until the abstraction is
earned.

**Earned abstraction.** After steps 4 and 5 the reader has computed the same fraction twice. Now name it: the
fraction of positives who have the condition is (people with it who test positive) divided by (everyone who
tests positive). Then write it with symbols, pointing each at the grid: P(condition | positive) = P(positive |
condition) P(condition) / P(positive). The denominator is 59 of 1000, and the reader can see where 59 came
from. Only now mention that this is Bayes' theorem.

**The napkin test.** Grid, two splits, count the positives, take the fraction: four drawings. Passes.

**What the reader can now do.** Given any rate, any catch rate and any false-flag rate, build the grid of 1000
and read off the answer. Try: 1 in 1000 has the condition, the test catches 99 of 100 and flags 2 in 100
healthy. (About 5 percent of positives have it.)

## Register and length

- Second person, present tense, active voice. "You split the thousand" not "the population is partitioned".
- Numbers as whole things where possible (10 people, not 0.01). Percentages only after the count.
- One idea per paragraph; the first sentence of a paragraph says what the paragraph does.
- Short. A strong explanation of one idea is often under four hundred words plus the picture. Length is where
  the curse of knowledge hides: it is easier to write everything than to find the step.

## Anti-patterns

- Opening with history or a definition instead of the picture.
- "It is easy to see", "clearly", "obviously": each marks a step you skipped. Delete the phrase and write the
  step.
- Three analogies for one idea. One picture, used all the way through.
- The general case first "for completeness".
- Explaining the notation before the thing the notation names.
- A worked example whose numbers hide the idea (everything equals 1; the sizes are all the same).
- Ending with a summary instead of a capability and a thing to try.

## Checklist

- [ ] One sentence written, about one thing, no jargon.
- [ ] One picture, at most three kinds of thing, the idea visible in it.
- [ ] Three to six steps, each one new thing, each with its "why does that follow" answered.
- [ ] Every noun and symbol checked against the persona; misses introduced, replaced or cut.
- [ ] Two instances before the general statement; the formula last, each symbol pointed at the picture.
- [ ] Napkin test passes.
- [ ] Ends with what the reader can now do and one thing to try.
