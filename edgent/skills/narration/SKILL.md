---
name: narration
description: Writing spoken narration for films, read-aloud chapters and voice tutoring: the spoken register, what to say at a formula, a table or a figure, pacing, TTS-safe text.
---
# Narration

Use this for any text that will be heard rather than read: a film's narration, a chapter read aloud, a spoken
tutor turn, the stems of items read to a young learner. Spoken text is a different medium: no backtracking, no
scanning ahead, no typography. It must carry its structure in the words.

## The spoken register

- Second person, present tense, short sentences: twelve to eighteen words at undergraduate level, eight to
  twelve at school level, six to ten for kindergarten and primary.
- One idea per sentence; one sentence per thing that happens on screen.
- Say the structure out loud: "First", "Now the second group", "Here is the point", "So". Written text uses
  headings and whitespace; speech uses these words.
- Concrete nouns and numbers before abstractions, as in writing, but more so: the ear cannot hold "the
  conditional probability of the evidence given the hypothesis" and keep listening.
- Repetition is a tool, not a fault: repeat the key noun rather than using a pronoun ("the ten who have it",
  not "they") whenever a listener could lose the referent.
- Rhythm: vary sentence length; end the key sentence on the key word. "Most of the positives are healthy." not
  "Healthy people account for most of the positives."
- No stage directions, no "as you can see", no "in this video". The narrator talks about the idea, not about
  the medium.

## What a narrator says at a formula

Never read symbols as symbols. A formula is read as the sentence it stands for, with the picture it points to:

- Written: P(sick | positive) = P(positive | sick) P(sick) / P(positive).
- Spoken: "The chance you are sick, given a positive test, is the chance of a positive test when you are sick,
  times how common the sickness is, divided by how common positives are. On the grid: nine, out of
  fifty-nine."

Rules:
- Say what each symbol means in words the first time; after that a short name is enough ("the catch rate").
- Fractions: "nine out of fifty-nine", then "about fifteen percent"; never "nine over fifty-nine" for a
  listener below undergraduate level.
- Operations in words: "times", "divided by", "plus", "the square of", "the sum of all the", "x to the power
  of two" only when the listener is used to it.
- Subscripts and indices: "x one, x two, and so on up to x n"; "the i-th term". Greek letters by name, and say
  what they stand for ("theta, the parameter we are estimating").
- Equals: "is", "gives", "equals" for an identity, "comes to" for a computed value.
- Point at the screen with words: "the top line", "the left part", "the thing in the box", in the order the
  eye should move.
- A derivation is narrated as moves, one per sentence: "Multiply both sides by the number of positives. Now
  the left side counts the sick positives."

## What a narrator says at a table

Do not read a table. Say what it shows:

1. What the rows and columns are: "Each row is a rate of the condition; the columns are the fraction of
   positives who really have it."
2. The one comparison the table exists for: "Go down the first column: as the rate falls from one in ten to
   one in a thousand, the fraction drops from two thirds to about five percent."
3. The cell to remember, with its coordinates in words: "At one in a hundred, the middle row, it is fifteen
   percent."

Three sentences for a table; the listener who needs the numbers has the page.

## What a narrator says at a figure

- Orient: what the figure is, in the listener's words ("a thousand people in a grid").
- Direct the eye in reading order: "The thin bar on the left is the ten who are sick. The wide area is
  everyone else."
- Say the relationship the figure shows, which is the caption's sentence, after the eye has arrived: "The
  outlined parts are the positives: nine of the sick, fifty of the healthy."
- On a film, each sentence lands when its object appears or lights up (`manim-film`); the words and the motion
  are one thing.

## Pacing for the ear

- About 2.6 words per second for an adult listener; slower for young learners and for anything the listener
  must do while listening (a code walkthrough, a procedure).
- A pause (a full stop, a line break in the storyboard) after each key sentence; a longer pause before "Here
  is the point".
- A film beat of 20 to 35 words runs 8 to 14 seconds; write the words first and set the seconds from them.
- Silence is allowed: a beat where the picture moves and the narrator has said what to watch may have two
  seconds of nothing.
- Never narrate faster to fit; cut words instead.

## TTS-safe text

The synthesiser reads exactly what is written. Write for it:

- Spell out numbers that a reader would say differently from how they are written: "one in a hundred" not
  "1/100"; "fifteen percent" not "15%"; "two thousand and four" for a year; "zero point eight five" or "point
  eight five" for 0.85.
- No symbols: no `$`, `=`, `+`, `*`, `^`, `|`, `->`, no Markdown, no code fences, no inline code. Code is read
  as words ("the function posterior takes a rate, a catch rate and a flag rate").
- Abbreviations expanded or respelled: "for example", "that is", "versus"; "P D F" with spaces when it must be
  spelled; "SQL" as "sequel" only if the listener says it that way.
- Units in words: "ten centimetres", "three milliseconds", "twenty degrees".
- Names and foreign words that a synthesiser mispronounces are respelled phonetically in the narration and
  spelled correctly in the captions.
- Punctuation the synthesiser honours: full stops and commas for pauses; question marks for rising tone; no
  semicolons, dashes, parentheses or colons (rewrite as sentences).
- Each beat's narration is plain text; captions (`video_assemble` with `captions: "storyboard"`) are generated
  from it, so it must read correctly on screen too.
- Short lines in the storyboard where a pause belongs; the renderer joins them.

## A worked beat

Written for the eye (what a chapter might say):

> Of the 990 healthy people, 5% test positive: 0.05 × 990 ≈ 50 false positives, so P(positive) = (9 + 50)/1000
= 0.059.

Spoken, for the same beat:

> Now the nine hundred and ninety who do not have it. The test is wrong about five in every hundred of them.
Five in a hundred, of nine hundred and ninety, is about fifty. So fifty healthy people test positive. Add the
nine who really have it, and fifty-nine people are positive in all.

Six short sentences, forty-nine words, about nineteen seconds at 2.6 words per second; set `seconds` to 20 and
let the fifty light up on "about fifty". Every number is in words, the percent became "five in every hundred",
the multiplication became "of", and the formula became "add the nine".

## Reading code aloud

A code walkthrough is narrated as what each line does, never as its characters:

- Name the function and what it takes: "The function posterior takes a rate, a catch rate and a flag rate."
- One sentence per line that matters, in the order the reader's eye moves; skip boilerplate and say so: "The
  next two lines are imports."
- Say the value, not the expression, when the value is the point: "true positives come to nine thousandths."
- Never read punctuation, brackets or operators; "times", "divided by", "returns", "if the rate is zero".
- End with the output: "Running it with one in a hundred prints point one five."

## Spoken tutoring turns

In a voice session the tutor turn follows the same rules plus:
- Shorter still: three to five sentences, then a question.
- Quote the learner's words back briefly before answering, so they know what was heard.
- No lists; say "two things" and then say them as sentences.
- Say the locator in words when citing ("the textbook, page twelve") and keep the written citation for the
  transcript.

## Anti-patterns

- Reading a formula symbol by symbol; reading a table row by row.
- "As you can see", "in this section", "let us now"; any reference to the medium.
- Pronouns across sentences for things on screen ("it", "this one") when two things are visible.
- Long sentences with subordinate clauses that the ear cannot hold.
- Numbers and symbols written for the eye in text that will be synthesised.
- Narration written after the animation, then squeezed to fit.

## Checklist

- [ ] Second person, present tense; sentence length for the level; one idea per sentence.
- [ ] Structure said in words (first, now, so, here is the point).
- [ ] Every formula read as a sentence with each symbol named once and pointed at the picture.
- [ ] Tables: rows and columns, the one comparison, the cell to remember. Figures: orient, direct, state the
      relationship.
- [ ] About 2.6 words per second; pauses marked; seconds set from the words.
- [ ] TTS-safe: numbers, units and abbreviations in words; no symbols, markup or code; punctuation only full
      stops, commas, question marks.
- [ ] Reads correctly as captions too.
