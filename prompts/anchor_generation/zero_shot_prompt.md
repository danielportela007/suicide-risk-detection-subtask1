# Zero-shot synthetic-anchor prompt — version 1.0.0

Paste the following complete prompt into the model and replace
`{{TARGET_BATCH_JSON}}` with one batch from `target_catalog.json`.

---

You are generating a controlled research artifact for an explainable suicide-risk detection study. Generate synthetic English phrases from the operational definitions supplied below. This is a non-clinical NLP benchmark: do not give advice, instructions, methods, quantities, or graphic descriptions related to self-harm or suicide.

## Task

For every target in TARGET_BATCH, generate exactly 15 distinct synthetic English phrases that directly and prototypically express the target definition.

This is a zero-shot generation condition:

- Work only from the supplied label, definition, inclusion criteria, and exclusion criteria.
- Do not request examples and do not imitate any dataset, author, platform user, or real person.
- Do not add an expert persona, imagined corpus analysis, or hidden external taxonomy.
- Do not quote or paraphrase restricted social-media posts.

## Phrase requirements

1. Every phrase must be natural English and understandable without surrounding context.
2. Use first-person singular language unless the target definition explicitly concerns another person's suicide exposure.
3. Each phrase must contain 4-25 words.
4. Each phrase must express one primary target only. Minimize accidental expression of other labels.
5. Prefer short clauses or single sentences suitable for semantic embedding and local evidence matching.
6. Vary wording and syntax, but keep the expression direct, explicit, and prototypical rather than literary, highly contextual, metaphorical, or ambiguous.
7. Do not place the target label itself inside a phrase unless that wording would occur naturally in ordinary English.
8. Do not produce exact duplicates, near-duplicates, or trivial tense/pronoun substitutions.
9. Do not include names, demographic stereotypes, URLs, usernames, quotations, diagnostic conclusions about another person, or references to this prompt.
10. Never include a named suicide method, procedural step, quantity, dosage, location, optimization detail, or graphic description. Generic non-actionable wording is mandatory where suicidality is relevant.

## Risk-level controls

- `Indicator` is a hard-negative class. Its phrases may show substantial distress or adversity but must contain no suicidal thought, wish to die, plan, preparation, access-to-means cue, self-harm, or attempt. Never treat an Indicator phrase as positive suicide evidence.
- `Ideation` must clearly express a self-directed suicidal thought or wish to die, while excluding plan, preparation, access, behavior, and attempt.
- `Behavior` must include suicidality plus a plan, preparation, access, or behavior cue, while excluding any statement that an attempt actually occurred.
- `Attempt` must unambiguously refer to a first-person attempt that already occurred and must not be merely hypothetical or planned.

## Factor controls

- Factor phrases should express the factor itself rather than a suicide-risk level.
- Except for `prior self-harm or suicidal thought/attempt` and `suicide means (with access)`, do not introduce suicidal language into factor anchors.
- Treat the five protective factors as positive constructs, not simply as absence of risk.
- Respect the distinction between `poor social support` and `interpersonal difficulty`, between `hopelessness` and `low self-esteem`, and between `stressful life event` and `traumatic experience`.

## Internal quality check

Before responding, silently verify for every phrase:

- target fidelity;
- inclusion-criteria satisfaction;
- exclusion-criteria satisfaction;
- non-actionability and safety;
- length compliance;
- English language;
- uniqueness within the batch.

Revise any failing phrase before producing the response.

## Output format

Return valid JSON only, with no Markdown and no explanatory text. Use exactly this structure:

{
  "prompting_strategy": "zero_shot",
  "prompt_version": "1.0.0",
  "batch_name": "<batch name supplied with the targets>",
  "targets": [
    {
      "target_type": "<risk_level or factor>",
      "target_label": "<exact supplied label>",
      "anchor_role": "<exact supplied anchor_role>",
      "phrases": [
        {
          "id": "<batch_slug__target_slug__zs__01 through 15>",
          "text": "<synthetic English phrase>",
          "temporality": "current|past|recurring|general|not_applicable"
        }
      ]
    }
  ]
}

TARGET_BATCH:

{{TARGET_BATCH_JSON}}

