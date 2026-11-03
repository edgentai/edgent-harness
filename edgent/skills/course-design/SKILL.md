---
name: course-design
description: Planning a book or course: objectives as observable behaviours, prerequisite graph, sequencing, Bloom levels, film or widget or prose, pacing per level, proposing the outline.
---
# Course design

Use this to plan: an outline for a build, a syllabus, a lesson sequence, or the objectives for an assessment.
The output of planning is `book/outline.json`, proposed through `book_outline_propose`, and the rules here are
what make it buildable and checkable.

## Objectives are observable behaviours

An objective says what the learner can do afterwards, in a form someone could watch them do. Write it as:
*can* + verb + object + condition + standard, where the condition and standard are included when they change
the task.

- Not objectives: "understand recursion", "appreciate the role of priors", "be familiar with matrices".
- Objectives: "can trace a recursive function by hand on inputs up to depth four", "can compute a posterior
  from a prior and a likelihood by building a frequency grid", "can decide whether a system of three linear
  equations has a unique solution by row reduction".

Rules:
- One verb per objective; two verbs are two objectives.
- The verb names the Bloom level (below); the assessment item kind follows from it.
- The condition says what the learner is given (a table, a program, a dataset, nothing); the standard says how
  well (without error, within a tolerance, in under ten minutes).
- Every unit in the outline has one objective; a chapter's objective is the capability its sections add up to.

## Bloom levels and what tests them

| Level | Verbs | Item kinds that test it |
|---|---|---|
| Remember | state, list, name, recall | mcq, short |
| Understand | explain, give an example of, classify, summarise, predict | short, mcq with explanation-type distractors, teach-back |
| Apply | compute, use, carry out, solve (a standard case) | numeric, code, short |
| Analyse | compare, decompose, trace, find the error in | code, proof, short with rubric |
| Evaluate | judge, justify, choose between, critique | proof, short with rubric, viva |
| Create | design, write, derive, build | code, proof, project with rubric |

A course that stays at Remember and Understand produces learners who cannot do anything; one that starts at
Analyse loses them. The usual shape of a chapter: two or three Apply objectives on one Understand objective,
one Analyse objective at the end.

## The prerequisite graph

Concepts are the nodes; units introduce and require them.

1. List the concepts the goal needs, as short lower-case phrases (`dot product`, `conditional probability`,
   `recursion`). Thirty to eighty for a book; five to fifteen for a short course.
2. For each concept, the concepts it needs. Draw the arrows; the graph must have no cycles.
3. Assign each concept to exactly one unit that introduces it (`meta.introduces`); every unit that uses it
   later lists it in `meta.requires`.
4. Order the units so every required concept is introduced earlier. `check_prereqs` enforces this on the built
   chapters; the outline is where you get it right first.
5. The persona's starting knowledge is the set of concepts no unit introduces. Write it down in the outline's
   persona string, because every forward reference check assumes it.

A unit that requires more than six concepts is probably too late in the book or too big.

## Sequencing

- Concrete before abstract: the first unit that touches a concept uses an instance; the general treatment
  comes one or two units later.
- One new thing per unit at school level; two or three at undergraduate; a graduate unit may introduce a
  cluster that only makes sense together.
- Spiral: a concept introduced early returns in a later unit at a higher Bloom level. Plan the return
  explicitly as an objective, not as a hope.
- Interleave practice: an exercise set per chapter, depending on it; a cumulative set every three or four
  chapters that requires the earlier concepts.
- Split a unit when its draft would exceed the level's length (below) or when it introduces concepts that do
  not depend on each other.
- Put the research unit first when the sources are thin, and the integration unit last, depending on
  everything.

## When a film, a widget, a figure or prose

| The idea is | Use | Because |
|---|---|---|
| a process, a transformation, something that changes over time or steps (an algorithm running, a vector being rotated, a wave propagating) | film | motion is the idea; a still frame loses it |
| a relationship between a few things (parts of a whole, a mapping, a hierarchy, a geometric fact) | figure (SVG diagram) | one picture, read at a glance, cited from the text |
| a quantity the reader should change and watch (a parameter, a sample size, a threshold) | widget | the reader's own action is what teaches |
| a definition, an argument, a derivation, a comparison in words | prose | the reader needs to re-read and stop |
| a skill (compute, prove, write) | exercise set | nothing teaches a skill but doing it, with a key that says whether it was done |

Films are expensive to make and verify: only where motion is the idea. A book of twelve chapters usually earns
three to six films, ten to twenty figures, two to five widgets.

## Pacing per level

Length per chapter, examples per idea, and session length by level. These are defaults; `STUDY.md` overrides
them.

| Level | Chapter length (words) | Sections per chapter | Examples per idea | Learning session | Assessment |
|---|---|---|---|---|---|
| kg | 150-400, read aloud | 1-2 | 3, all concrete | 10-15 min | play, oral, one item |
| primary | 400-1200 | 2-4 | 2-3 | 15-25 min | 3-5 items, mostly apply |
| middle | 1000-2500 | 3-5 | 2 | 25-40 min | 5-8 items |
| secondary | 2000-4000 | 4-6 | 1-2 | 40-60 min | 8-12 items, one analyse |
| undergraduate | 4000-8000 | 5-8 | 1-2 | 60-90 min | 10-15 items, proofs or code |
| graduate | 6000-12000 | 5-8 | 1 | 90-120 min | problems, a project |
| professional | 2000-6000, task-shaped | 3-6 | 1, from their domain | 30-60 min | a task with a rubric |

Readability bands the checker uses (Flesch-Kincaid grade): kg 1-2, primary 2-5, middle 5-8, secondary 8-11,
undergraduate 10-15, graduate 12-18, professional 10-16. Set `meta.level` on every text unit so the band
applies.

Sizing a build: a "300-page book" is about 90,000 words; at undergraduate level that is twelve to fifteen
chapters of 6,000-8,000 words, each with five to eight sections, an exercise set, and media where the table
above says so.

## Sizing a build, worked

Goal: "a 300-page undergraduate book on linear algebra for a reader who knows high-school algebra, with films
for the ideas that move".

- Words: about 90,000. At 6,000 to 8,000 words per chapter, thirteen chapters; each five to seven sections.
- Concepts: about sixty, from "vector" to "singular value"; the persona's starting set is high-school algebra
  and coordinates, so "vector" is introduced, not assumed.
- Media: films where motion is the idea: a linear map as a stretching of the plane, row reduction as moves, an
  eigenvector staying on its line, a projection falling onto a subspace, iteration converging (five films).
  Figures for relationships: span, basis, the four subspaces, orthogonal complements (about fifteen). Widgets
  where the reader should push on a value: a 2 x 2 matrix with sliders, a least-squares fit with movable
  points (two or three).
- Assessment: an exercise set per chapter, cumulative sets after chapters 4, 8 and 12.
- Order: a research unit first only if `sources/INDEX.md` lacks a reference text; the integration unit last.

That is roughly thirteen chapters, five films, fifteen figures, three widgets, sixteen exercise sets, one
integration unit: about fifty-three units, each with an objective, a level, its concepts and its dependencies.

## Proposing the outline

`book_outline_propose` takes a title, a persona and a list of units in reading order. One unit:

```
{"id": "chapter-vectors", "kind": "chapter", "title": "Vectors", "objective": "can add and scale vectors and draw the result",
 "depends_on": [], "path": "book/chapters/01-vectors.md",
 "meta": {"level": "undergraduate", "introduces": ["vector", "scalar multiple"], "requires": [], "estimated_words": 6000}}
```

Rules the proposal must follow:
- Kinds: `chapter`, `section`, `figure`, `film`, `widget`, `exercise_set`, `research`, `integration`,
  `assessment`, `lesson`.
- Ids are short slugs, stable for the life of the book; existing units are never replaced, so choose ids you
  will not want to rename.
- Paths: `book/chapters/NN-slug.md` for chapters, sections and lessons; `media/films/<id>` for films;
  `media/diagrams/<id>.svg` for figures; `media/widgets/<id>.html` for widgets; `assessments/<id>/items.json`
  for exercise sets and assessments.
- `depends_on`: every unit whose concepts this one requires, plus the media units it embeds. A chapter that
  embeds a film depends on the film; the film depends on the chapter that introduces the concept it animates
  only if it uses that chapter's notation (usually it does not; it depends on nothing).
- `meta.introduces` and `meta.requires` on every text unit; `meta.level` on every unit; `meta.estimated_words`
  on text units; `meta.seconds` on films (60-240 is usual).
- An exercise set per chapter, depending on that chapter, with `meta.objectives` listing the chapter's
  objectives it tests.
- A research unit first when `sources/INDEX.md` is thin for the goal, with `meta.questions`; one integration
  unit last with `depends_on` naming every other unit.
- Objectives as observable behaviours (above); the persona string names the starting knowledge.

Present the proposal to the user before any unit is built: the unit list with objectives, the media choices
with a line of justification each, the assumptions about the persona. Revise on comments; add, never delete.
No tool removes or blocks a unit from here: note a dropped unit's reason in its `meta` through
`book_outline_update` and tell the user, so the build operator can block it; a pending unit is otherwise still
built.

## Checklist for an outline

- [ ] Every objective is an observable behaviour with one verb.
- [ ] Every concept is introduced by exactly one unit before any unit requires it; no cycles.
- [ ] Units ordered concrete to abstract, one or few new things each, spiral returns planned as objectives.
- [ ] Media kinds chosen by the table: films only where motion is the idea.
- [ ] Lengths, sections and examples match the level's pacing; `meta.level` set everywhere.
- [ ] Exercise set per chapter; research first when sources are thin; integration last.
- [ ] Paths follow the layout; ids are stable slugs; `depends_on` includes embedded media.
- [ ] Persona string states the starting knowledge; assumptions reported in one line.
