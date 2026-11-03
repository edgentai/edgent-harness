---
name: reading-levels
description: Pitching prose at a level from kindergarten to professional: vocabulary, sentence length, example choice per level and language, the readability bands check_readability uses.
---
# Reading levels

Use this when writing or revising prose for a stated level, and when `check_readability` reports a grade
outside the band or sentences over the ceiling. The level comes from `STUDY.md`, the chapter's front matter
(`level`), the outline unit or the persona. The checker measures the prose only: code, math, citations and
markup are stripped first.

## The bands the checker uses

| Level | Flesch-Kincaid grade band | Sentence ceiling (words) | Who |
|---|---|---|---|
| kg | 1-2 | 10 | pre-readers and first readers; read aloud |
| primary | 2-5 | 15 | ages about six to eleven |
| middle | 5-8 | 20 | ages about eleven to fourteen |
| secondary | 8-11 | 25 | high school, sixth form, A-level |
| undergraduate | 10-15 | 35 | university, first degree |
| graduate | 12-18 | 40 | master's, doctoral, researchers |
| professional | 10-16 | 35 | practitioners reading to do a task |

The grade is computed from average sentence length and syllables per word; a text under fifty words gets only
the sentence-length judgement. A single sentence over the ceiling is a defect on its own, listed with its
line, because that is what a reader at that level trips on first. The level is recognised from words such as
"high school", "undergrad", "PhD", "practitioner" when it is not given exactly.

Two levers move the grade: sentence length and long words (three syllables or more). Everything below is about
using them on purpose.

## Vocabulary

- Technical terms are allowed at every level; what changes is how they are introduced and how many. A term is
  introduced once, with an example, in the sentence where it first appears, and then used consistently. Below
  secondary level, one new term per section; at undergraduate, three or four; at graduate, a cluster when they
  belong together.
- Everyday words for everything that is not a technical term. "Use" not "utilise", "show" not "demonstrate",
  "about" not "approximately", "so" not "therefore" below undergraduate.
- Long non-technical words are the main cause of a grade above the band; replace them before touching the
  technical terms.
- Avoid words with a plain meaning that differs from the technical one (significant, normal, random, bias,
  work) unless the technical meaning has just been introduced.
- Pronouns refer to the nearest noun; below secondary level, repeat the noun.

By level:
- kg: words a five-year-old says; no abstract nouns; numbers to twenty as words; everything is a thing or an
  action.
- primary: concrete nouns and verbs; one technical term per section; comparisons to things at home and school.
- middle: abstract nouns allowed once anchored ("a rule, like the rule for odd numbers"); technical terms
  defined in the text, not in a glossary.
- secondary: the subject's standard terms, introduced with examples; Latin and Greek roots explained once
  ("iso-, the same").
- undergraduate: the field's vocabulary assumed after introduction; symbols read aloud on first use.
- graduate: the field's vocabulary assumed; precision over plainness when they conflict; hedges carry meaning
  ("suggests", "is consistent with").
- professional: the trade's vocabulary assumed; task verbs ("configure", "verify", "roll back"); no history,
  no theory not needed for the task.

## Sentence length and shape

- Average sentence length roughly: kg 5-7 words, primary 8-11, middle 12-15, secondary 15-18, undergraduate
  18-22, graduate 20-25, professional 15-20. The ceilings above are the maximum for any one sentence.
- One clause per sentence below middle level; one subordinate clause at secondary; two at undergraduate and
  above, never nested.
- Subject, verb, object, early in the sentence. The thing the sentence is about comes first.
- Active voice unless the actor is unknown or irrelevant. "The test flags fifty healthy people", not "fifty
  healthy people are flagged by the test".
- Lists of three or more things become a bulleted list below undergraduate level, and a sentence with "three
  things:" above it.
- A sentence that needs a semicolon is two sentences.

## Example choice

- kg and primary: the child's own world (toys, pets, snacks, the playground); counts of small whole things;
  stories with a character.
- middle: school, sport, games, money in small amounts; a puzzle with a surprise.
- secondary: everyday science and technology, phones, music, sport statistics, simple experiments; one example
  from the subject's history when it carries the idea.
- undergraduate: the discipline's standard examples plus one from current practice; data they could obtain;
  code they could run.
- graduate: examples from the literature and from the reader's likely research; the edge case before the
  general case.
- professional: the reader's own task, their tools, their data; the example is the thing they will do next.

Across all levels: numbers chosen to be countable (one in a hundred, a thousand people) before rates; the same
example carried through a section rather than a new one per paragraph.

## Language

- Write in the language `STUDY.md` names; keep technical terms in that language when the field uses them, with
  the English term in parentheses once if the learner will meet it in the sources.
- The grade formula is calibrated for English; for other languages treat the sentence ceilings as the binding
  rule and the grade band as a guide. Say so in the report when the level was checked on non-English prose.
- Sources in another language: quote in the original with a translation, and cite the locator of the original.
- Idioms, humour and cultural references do not travel; use plain statements.
- For a learner reading in a second language: one level below their subject level for sentence shape, the
  subject level for vocabulary.

## Moving a text between levels

To bring a text down a level:
1. Split every sentence over the ceiling at its conjunction or relative pronoun.
2. Replace long non-technical words with short ones; keep the technical terms.
3. Add a concrete example before each abstraction; cut the second abstraction in a paragraph.
4. Repeat nouns in place of pronouns; put the subject first.
5. Re-run `check_readability`; read the sentences it still lists aloud and shorten them.

To bring a text up a level:
1. Merge pairs of short sentences that share a subject, with one subordinate clause.
2. Use the field's terms once introduced, and drop the re-explanations.
3. Replace the second and third example of an idea with the general statement.
4. Add the conditions and the hedges a precise reader expects.

Do not change the content when changing the level; a simpler text is not a less true one.

## The same idea at three levels

The claim: a rare condition makes a positive test weak evidence.

- primary (grade band 2-5, ceiling 15 words): "Ten kids in the school have a cold. The nurse's test finds nine
  of them. But the test also says fifty healthy kids have a cold. So most kids the test picks are fine." Short
  words, a character, whole kids, four sentences under twelve words.
- secondary (band 8-11, ceiling 25): "Suppose ten people in a thousand have the condition and the test catches
  nine of them. If it also wrongly flags five percent of the healthy, that is fifty people. Of the fifty-nine
  positives, only nine are real, so a positive result is weaker evidence than the ninety percent suggests."
  One subordinate clause per sentence, percent after the count, the technical word "evidence" used plainly.
- graduate (band 12-18, ceiling 40): "With prevalence 0.01, sensitivity 0.9 and false-positive rate 0.05, the
  positive predictive value is 0.009/(0.009 + 0.0495), about 0.15; the result is dominated by the
  false-positive mass because the prior is small, which is why screening tests for rare conditions are
  followed by a confirmatory test." The field's terms assumed, one long sentence with two clauses, the
  mechanism and its consequence.

Same numbers, same claim, three vocabularies and three sentence shapes. The content did not change; the reader
did.

## Reading check_readability output

- `fk_grade` outside `band`: the direction tells you which lever; "above" means shorter sentences and shorter
  words, "below" (rare, and usually fine) means the text may be patronising for the level.
- `sentences_over_ceiling` with lines: fix those lines first; they are the defects that hold a unit.
- `level_source`: where the level came from. "none" means set `level` in the front matter so the band applies.
- `words` under fifty: the grade is noise; only the sentence lengths were judged.

## Anti-patterns

- Simplifying by removing the technical term instead of introducing it.
- Hitting the band with short sentences full of long words, or long sentences full of short ones.
- Explaining a kindergarten idea in undergraduate syntax because the subject is hard.
- Examples from the author's world rather than the reader's.
- Changing the meaning while changing the level.
- Ignoring the listed sentences because the average is inside the band.

## Checklist

- [ ] Level known (front matter, outline or STUDY.md); band and ceiling noted.
- [ ] Technical terms introduced once with an example; everyday words elsewhere; long non-technical words
      replaced.
- [ ] Sentence lengths at the level's average; none over the ceiling; one clause per sentence below middle.
- [ ] Examples from the reader's world at this level; one example carried through a section.
- [ ] Language from STUDY.md; non-English checked by ceilings, said so.
- [ ] `check_readability` inside the band with no listed sentences, or the remaining ones justified.
