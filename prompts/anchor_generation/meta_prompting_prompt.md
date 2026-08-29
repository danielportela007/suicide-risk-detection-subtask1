# Meta-prompting synthetic-anchor prompt — version 1.0.0

Paste the following complete prompt into the model and replace
`{{TARGET_BATCH_JSON}}` with one batch from `target_catalog.json`.

---

You are the lead designer of a controlled synthetic-language protocol for an explainable suicide-risk detection research project. Approach the task through four coordinated perspectives: clinical construct fidelity, computational linguistics, natural first-person online discourse, and safety review. This is a non-clinical NLP benchmark: never provide advice, instructions, methods, quantities, or graphic descriptions related to self-harm or suicide.

## Research objective

For every target in TARGET_BATCH, generate exactly 15 distinct synthetic English phrases that capture how the construct could be expressed naturally, indirectly, colloquially, hesitantly, or through its functional consequences. The phrases will become semantic anchors; they are not labels, diagnoses, real quotations, or synthetic training examples copied from a corpus.

## Meta-prompting procedure

For each target, perform the following process internally before writing the final output:

1. Operationalize the target using only its supplied definition, inclusion criteria, and exclusion criteria.
2. Identify the closest confusable labels and determine what semantic cue must remain present or absent to preserve the boundary.
3. Construct a coverage plan for 15 phrases across the expression modes below.
4. Generate candidates with genuine lexical, syntactic, pragmatic, and contextual diversity.
5. Audit every candidate for target fidelity, naturalness, uniqueness, ambiguity, accidental co-labeling, and safety.
6. Replace weak, templated, overly poetic, stereotyped, or boundary-violating candidates.

Do not output hidden reasoning. Output only a concise coverage summary and the audited phrases in the required JSON structure.

## Required coverage for each set of 15

Cover all of the following modes across each target's set. A phrase may satisfy more than one mode, but the final set must not collapse into one template:

- 3 indirect but semantically recoverable expressions;
- 3 conversational or colloquial expressions with contractions where natural;
- 2 hesitant, guarded, or mildly evasive expressions;
- 2 fragments, self-corrections, pauses, or discourse-marker expressions;
- 2 expressions through functional or behavioral consequences;
- 2 temporally or situationally contextualized expressions;
- 1 less common but plausible formulation that remains label-faithful.

The language should resemble varied informal first-person English, not a caricature of Reddit. Use restrained discourse markers and occasional fragments, but do not manufacture excessive misspellings, slang, dialect, or demographic stereotypes.

## Phrase requirements

1. Every phrase must be natural English and understandable as a standalone excerpt.
2. Use first-person singular unless the target explicitly concerns exposure to another person's suicide.
3. Each phrase must contain 4-30 words.
4. Each phrase must retain one primary target, even when the expression is indirect.
5. Do not include the target label mechanically in the phrase.
6. Avoid generic distress language that could fit nearly every category.
7. Do not create exact duplicates, semantic near-duplicates, template substitutions, or fifteen versions of the same proposition.
8. Do not quote, paraphrase, stylistically imitate, or derive content from any restricted dataset or real social-media post.
9. Do not include names, usernames, URLs, demographic stereotypes, gratuitous identity details, or diagnostic judgments about other people.
10. Never include a named suicide method, procedural step, quantity, dosage, location, optimization detail, or graphic description. Use only generic non-actionable wording where suicidality is part of the construct.

## Risk-level boundary rules

- `Indicator`: generate hard negatives showing meaningful distress or adversity but absolutely no suicidal thought, wish to die, plan, preparation, access-to-means cue, self-harm, or attempt. Indirectness must not smuggle in a death wish. These phrases are not positive evidence anchors.
- `Ideation`: every phrase must remain semantically recognizable as self-directed suicidality or a wish to die, but must not imply a plan, preparation, access, behavior, or attempt. Indirect wording must still be sufficiently clear for the intended category.
- `Behavior`: every phrase must combine suicidality with a non-actionable plan, preparation, access, or behavior cue. It must not state that an actual attempt occurred.
- `Attempt`: every phrase must make it unambiguous that the speaker personally carried out an attempt in the past or recently. It must not be merely contemplated, planned, fictional, quoted, or third-person.

## Factor boundary rules

- Generate the factor itself, not a risk-level phrase decorated with the factor.
- Except for `prior self-harm or suicidal thought/attempt` and `suicide means (with access)`, exclude suicidal content from factor anchors.
- Keep protective factors independently positive and linguistically varied.
- Preserve these high-risk distinctions: hopeless future versus negative self-worth; unavailable support versus relationship conflict; cognitive difficulty versus poor academic outcome; major stressor versus trauma; internal resilience versus external social support; responsibility versus broader meaning in life.

## Final audit

Before responding, silently reject and replace any phrase that:

- violates inclusion or exclusion criteria;
- changes the intended risk-level boundary;
- expresses multiple factors so strongly that the primary target is unclear;
- is generic enough to fit several labels;
- differs from another phrase only by a few words;
- sounds like a clinician, annotation guideline, motivational slogan, or generated template rather than a plausible first-person expression;
- contains actionable, graphic, or unnecessarily specific harm content.

## Output format

Return valid JSON only, with no Markdown and no explanatory prose. Use exactly this structure:

{
  "prompting_strategy": "meta_prompting",
  "prompt_version": "1.0.0",
  "batch_name": "<batch name supplied with the targets>",
  "targets": [
    {
      "target_type": "<risk_level or factor>",
      "target_label": "<exact supplied label>",
      "anchor_role": "<exact supplied anchor_role>",
      "coverage_summary": {
        "covered_modes": ["indirect", "colloquial", "hesitant_or_evasive", "fragment_or_disfluency", "functional", "temporal_or_situational", "uncommon_but_plausible"],
        "closest_confusions_checked": ["<short label names only>"]
      },
      "phrases": [
        {
          "id": "<batch_slug__target_slug__mp__01 through 15>",
          "text": "<synthetic English phrase>",
          "expression_modes": ["<one or more required modes>"],
          "explicitness": "indirect|mixed|explicit",
          "temporality": "current|past|recurring|general|not_applicable"
        }
      ]
    }
  ]
}

TARGET_BATCH:

{{TARGET_BATCH_JSON}}

