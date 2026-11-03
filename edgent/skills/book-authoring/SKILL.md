---
name: book-authoring
description: Drafting one chapter or section for a persona: the file contract, structure, motivation before definition, worked examples, notation, citations with locators, the checks to run.
---
# Book authoring

Use this when you draft or revise a chapter, section or lesson file in `book/chapters/`. Load `explaining`
with it: this skill is the structure and the contract; that one is how each idea gets explained.

## The chapter file contract

One Markdown file at the unit's path (`book/chapters/NN-slug.md`). Front matter first:

```
---
title: <chapter title>
unit_id: <this unit's id in book/outline.json>
persona: <who it is written for>
level: <kg | primary | middle | secondary | undergraduate | graduate | professional>
introduces: [<concept>, <concept>]
requires: [<concept introduced by an earlier unit>]
sources: [<source-id>, <source-id>]
---
```

Then the body, which the checks read:

- Headings in order: `#` title once, `##` sections, `###` sparingly. Each section ends with what the reader
  can now do.
- Math in KaTeX: `$inline$` and `$$display$$`; keep every `$` balanced; no `\begin{document}`, no packages.
- Code in fenced blocks with a language. Every ```` ```python ```` block is executed in order by
  `check_execute`, each in its own process in one shared scratch directory. Show the output in a ````
  ```output ```` block right after it when the output matters; it is compared with what the code printed. A
  block that needs an earlier block's definitions starts with the line `# edgent: continue`.
- Citations inline as `[source-id:p.12]`, `[source-id:t=00:14:32]` or `[source-id:L12]`; every one must
  resolve against `sources/text/<source-id>.md`. Never cite a source that is not there.
- Figures as `![caption](../../media/diagrams/<id>.svg)`; films as a line `{{film: <film-id>}}`; widgets as
  `{{widget: <widget-id>}}`; exercise sets as `{{exercises: <set-id>}}`. Reference only what exists or what a
  unit this one depends on has built.
- Prose at the persona's level; no concept before the outline introduces it; motivation before definition; a
  worked example per idea.

The front matter's `introduces` and `requires` are what `check_prereqs` reads; `level` is what
`check_readability` reads. Fill them honestly: an empty `requires` on a chapter that uses matrices is a lie
the checker will not catch but the reader will.

## Structure of a chapter

A chapter is a sequence of sections, each one idea, framed by an opening and a close.

### The opening (two to four paragraphs)

1. The question the chapter answers, in the reader's world. A problem they would recognise, a thing they
   cannot yet do.
2. Why the tools they have do not solve it (one paragraph, concrete: show the attempt failing).
3. What the chapter will give them, as capabilities, in the order they come.

No history, no "in this chapter we will see", no definitions yet.

### Each section (one idea)

Follow `explaining` for the idea itself. The shape of a section:

1. A motivating instance: the smallest concrete case where the idea is needed. Numbers, a picture, a short
   program.
2. The move: what you do about it, shown on the instance.
3. The name: now the term, the notation, the definition, pointed at the instance.
4. A second instance, worked, that differs from the first in the way that matters.
5. The general statement when the reader will use it, and a line on when it does not apply.
6. "What you can now do": one or two sentences, then one check question whose answer the reader can verify
   themselves (and the answer, folded or at the end).

A section runs five hundred to fifteen hundred words at undergraduate level; shorter below, rarely longer
above. When a section grows past that, it holds two ideas: split it.

### Worked examples

- One worked example per idea, minimum; two for the chapter's central idea; one of them should show the wrong
  turn a reader would take and why it fails.
- Every step shown. "Simplifying" is a defect; show the three moves once, then you may skip them later in the
  chapter, saying so.
- Numbers chosen so the idea is visible: not all ones, not all equal, not so large that the arithmetic hides
  the structure.
- Code examples run. A code block that is "illustrative" and does not run is removed or made to run. Its
  output is shown and is what the code prints.
- Examples drawn from the sources where the sources have them, with a citation; examples from the persona's
  world otherwise.

### Notation discipline

- Introduce a symbol once, in words, in the sentence where it first appears, pointing at a thing.
- One symbol, one meaning, for the whole book. Check the dependency chapters before choosing; reuse their
  notation. Never overload a letter within a chapter.
- A table of symbols at the first chapter that introduces more than five, and in the integration unit for the
  book.
- Prefer words to symbols in prose below undergraduate level; prefer symbols to prose in derivations at
  graduate level.
- Units and dimensions stated on every quantity that has them.

### Figures, films and widgets

- A figure where a picture explains a relationship; a film where motion is the idea; a widget where the reader
  should change a value and see the consequence. Prose otherwise.
- Place the reference where the reader needs the picture, not at the end. Caption says what to see in it, not
  what it is ("the positives split 9 coloured to 50 plain", not "Figure 2: the grid").
- Reference only media that exists or that a dependency unit built; otherwise write the prose so it stands
  without the picture and record the gap.

### The close (one or two paragraphs)

What the reader can now do, in the order the sections gave it. Where it leads (the next chapter's question,
one line). No new content, no summary of the text.

## Motivation before definition, always

The commonest defect a critic finds is a definition the reader did not want yet. The test for every
definition: is there, above it on the page, a concrete case in which the reader felt the lack of this word? If
not, move the case above the definition or cut the definition.

## Citations with locators

1. Find the passage: `grep` the text view; `read_file` around the hit.
2. Verify it: `sources_cite_check` with the locator. The quote and the locator must agree; fix the locator,
   not the quote.
3. Write it: the claim, then `[source-id:p.12]`. Quote verbatim when the wording matters; paraphrase with the
   citation when it does not. A paraphrase must not say more than the page says.
4. When the sources do not support a claim you need: say "unverified against the sources" in the text at that
   claim, keep it short, and record an unknown with the probe (which source would settle it).

A chapter cites what it rests on, not everything it mentions. Three to ten citations per section is usual for
source-grounded books; zero is a sign the chapter is written from memory and must say so.

## Reading level

`check_readability` strips code, math and markup and measures the prose: Flesch-Kincaid grade, sentence
length, long-word ratio. The bands (grade): kg 1-2, primary 2-5, middle 5-8, secondary 8-11, undergraduate
10-15, graduate 12-18, professional 10-16. Sentence ceilings in words: kg 10, primary 15, middle 20, secondary
25, undergraduate 35, graduate 40, professional 35. See `reading-levels` for how to write to a band. A
sentence over the ceiling is a defect on its own; split it.

## Before you return: the checks

Run them in this order on the file; fix and rerun until clean.

1. `check_render` with the path: front matter, headings, unclosed fences, unbalanced `$`, broken links,
   placeholder targets. With pandoc present it renders to HTML; without, structure only (say so).
2. `check_execute`: every python block runs, outputs match. A failing block is fixed or removed.
3. `check_cite`: every citation resolves. An unresolved one is a high defect: fix the locator or remove the
   claim.
4. `check_readability` with the level (or let it read the front matter): split the sentences it lists;
   simplify the words it counts.
5. `check_prereqs` with the unit id: a concept required before it is introduced is a forward reference; move
   the material or add the prerequisite to an earlier unit's plan (never introduce it silently here).
6. `run_checks` to see the whole ledger as `book_mark_done` will.

Read each defect's location and message; change the text, not the check. Never mark the unit done yourself;
the worker's gate does that.

## What not to do

- Do not begin with a definition, a list of prerequisites or a history paragraph.
- Do not write "it can be shown", "clearly", "trivially", "left as an exercise" (unless it is in the exercise
  set with a key).
- Do not introduce a symbol in a formula and define it afterwards.
- Do not reference a figure, film or widget that does not exist.
- Do not cite from memory; find the locator or say unverified.
- Do not use a concept the outline introduces later, even with a forward pointer.
- Do not pad a section to a target length; cut it to its idea.
- Do not end a section without saying what the reader can now do.
- Do not write code that does not run, or outputs you did not see.
- Do not touch other chapters to make this one work; record what they need instead.

## Checklist before returning

- [ ] Front matter complete and honest (unit_id, level, introduces, requires, sources).
- [ ] Opening: question, failed attempt, capabilities to come.
- [ ] Each section: instance, move, name, second instance, general statement, what you can now do, check
      question.
- [ ] One worked example per idea, every step shown, numbers that show the idea.
- [ ] Notation introduced once, consistent with dependency chapters.
- [ ] Media references point at files that exist; captions say what to see.
- [ ] Every citation verified with `sources_cite_check`; unsupported claims marked unverified.
- [ ] `check_render`, `check_execute`, `check_cite`, `check_readability`, `check_prereqs` clean, or the
      remaining defects named in the report.
- [ ] Close: capabilities and the next question; nothing new.
