# Example item set

A complete `key.json` for a short set on Bayes' theorem at undergraduate level. Every key computes; the one
proof item carries a rubric. The code key receives `answer` as the submitted source text, so its recipe `exec`s
it before asserting. The learner view (`items.json`) is the same without `answer`, `solution`,
`tolerance` and `verify`.

```json
{
  "set_id": "ch03-bayes",
  "title": "Bayes on a grid",
  "objectives": [
    "can compute the fraction of positives that have a condition from a rate, a catch rate and a false-flag rate",
    "can explain why a rare condition makes a positive test weak evidence"
  ],
  "level": "undergraduate",
  "items": [
    {
      "id": "q1",
      "kind": "numeric",
      "stem": "Of 1000 people, 10 have the condition. The test catches 9 of the 10 and wrongly flags 50 of the other 990. What fraction of the people who test positive have the condition? Give a decimal to two places.",
      "answer": 0.15,
      "tolerance": 0.01,
      "solution": "Positives: 9 + 50 = 59. Fraction with the condition: 9/59 = 0.1525, about 0.15 [textbook:p.12].",
      "concept": "posterior from counts",
      "difficulty": 2,
      "verify": {"type": "numeric", "expression": "9 / (9 + 50)"}
    },
    {
      "id": "q2",
      "kind": "mcq",
      "stem": "The condition becomes ten times more common: 100 of 1000 have it, the test still catches 90 percent and still flags 5 percent of the healthy. The fraction of positives who have it is closest to",
      "options": ["0.15", "0.67", "0.90", "0.05"],
      "answer": 1,
      "solution": "Positives: 90 + 45 = 135; 90/135 = 0.67. Option 0 keeps the old answer (ignores the change in rate); option 2 is the catch rate (the two conditionals confused); option 3 is the false-flag rate.",
      "concept": "effect of the prior",
      "difficulty": 3,
      "verify": {"type": "none"}
    },
    {
      "id": "q3",
      "kind": "code",
      "stem": "Write a function posterior(rate, catch, flag) that returns the fraction of positives who have the condition, for a rate of the condition, a catch rate on those who have it, and a flag rate on those who do not. Example: posterior(0.01, 0.9, 0.05) is about 0.154.",
      "answer": "def posterior(rate, catch, flag):\n    true_pos = rate * catch\n    false_pos = (1 - rate) * flag\n    return true_pos / (true_pos + false_pos)\n",
      "solution": "True positives are rate times catch (0.009); false positives are (1 - rate) times flag (0.0495); divide the first by their sum: 0.009 / 0.0585, about 0.154.",
      "concept": "posterior from counts",
      "difficulty": 3,
      "verify": {"type": "code", "code": "ns = {}\nexec(answer, ns)\nposterior = ns[\"posterior\"]\nassert abs(posterior(0.01, 0.9, 0.05) - 0.1538) < 0.001\nassert abs(posterior(0.1, 0.9, 0.05) - 2 / 3) < 0.001\nassert posterior(1.0, 0.9, 0.05) == 1.0\nassert posterior(0.5, 1.0, 1.0) == 0.5\n"}
    },
    {
      "id": "q4",
      "kind": "short",
      "stem": "In one word: which quantity, when it is small, makes a positive test weak evidence?",
      "answer": "rate",
      "solution": "The rate of the condition (the prior). When it is small, most positives come from the many healthy people [textbook:p.12].",
      "concept": "effect of the prior",
      "difficulty": 1,
      "verify": {"type": "regex", "pattern": "(the )?(base )?(rate|prior|prevalence)( of the condition)?"}
    },
    {
      "id": "q5",
      "kind": "proof",
      "stem": "Show that if the test never flags a healthy person, every positive has the condition, whatever the rate. Then say what happens to the fraction as the rate approaches zero when the flag rate is positive.",
      "answer": "With flag = 0 the false positives are 0, so the fraction is true_pos / true_pos = 1 for any rate > 0. With flag > 0, true_pos = rate * catch goes to 0 while false_pos approaches flag, so the fraction goes to 0.",
      "solution": "See rubric.md, q5.",
      "concept": "limits of the posterior",
      "difficulty": 4,
      "verify": {"type": "none"}
    }
  ]
}
```

The matching `rubric.md` rows for q5:

```
## q5 (proof, difficulty 4)
- [2] Writes the fraction as true positives over (true positives + false positives).
- [2] Sets the false positives to zero and concludes the fraction is 1 for any positive rate.
- [3] Shows the numerator goes to zero and the denominator stays positive as the rate goes to zero.
- [1] States the conclusion in words: a rare condition makes a positive weak evidence.
```
