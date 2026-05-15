# Evaluation

Evidence supporting the SAE Steering project lives here. The directory has two sources of evaluation data:

1. a **primary 200-participant user study** reported in a paper currently under review (raw data in a private repository), and
2. a **supplementary in-house sanity check** with five participants that exercises each steering modality on default settings.

## Contents

1. [Primary evaluation — submitted paper (n = 200)](#1-primary-evaluation--submitted-paper-n--200)
2. [Supplementary in-house evaluation — 5 participants × 3 modalities](#2-supplementary-in-house-evaluation--5-participants--3-modalities)
3. [Files](#files)
4. [How to read `responses.csv`](#how-to-read-responsescsv)
5. [Headline observations from the 15 supplementary responses](#headline-observations-from-the-15-supplementary-responses)

## 1. Primary evaluation — submitted paper (n = 200)

The primary evaluation is a **200-participant user study** conducted on Prolific. Participants compared a no-steering baseline against a slider-steering variant of the SAE Steering plugin built in this repository. The study, the analysis pipeline, the manuscript, and the anonymised raw data live in the private [OfflineEasyStudy](https://github.com/vaclavstibor/OfflineEasyStudy) repository because the paper is currently in review and the bundle contains raw participant data. Reviewers and collaborators can request access (see the contact line in the root `README.md`).

When the paper clears review, the anonymised analysis bundle (CSVs, notebooks, headline plots) will be linked from this README.

## 2. Supplementary in-house evaluation — 5 participants × 3 modalities

In addition to the formal study, five participants ran the steering loop on the bundled `ml-32m-filtered` dataset with the three steering modalities — **toggle**, **slider**, and **text (only)** — kept at the default study configuration:

- the default ELSA + Top‑K SAE checkpoint shipped via the GitHub Releases bootstrap (`TopKSAE-1024`),
- `num_iterations = 3`, `n_items_to_show = 12` per iteration,
- per-mode default reranking strategy (`feature-conditioned` additive blend for sliders/toggles, `feature-conditioned` for text),
- no domain-specific guidance: participants explored the controls freely and rated the experience after each modality.

## Files

- [`data/questionnaire.csv`](data/questionnaire.csv) — the five-question Likert instrument (1–5 scale) applied per modality.
- [`data/responses.csv`](data/responses.csv) — long-format responses, **15 rows** (5 participants × 3 modalities).
- `data/anonymization_map.csv` — mapping `anonym_1 … anonym_5` to the real participant identities. Kept locally only (see `.gitignore`) and shared on request with the supervisor / reviewer so the raw responses can be attributed during the review process; in all other contexts treat the `anonym_*` ids as canonical.

## How to read `responses.csv`


| Column           | Meaning                                                                  |
| ---------------- | ------------------------------------------------------------------------ |
| `participant_id` | `anonym_1` … `anonym_5`; resolves via `anonymization_map.csv`            |
| `mode`           | `toggle`, `slider`, or `text`                                            |
| `q1_overall`     | Interface understandability (1 = not at all; 5 = very)                   |
| `q2_controls`    | Ease of using steering controls (1 = very difficult; 5 = very easy)      |
| `q3_control`     | Felt in control of recommendations (1 = not at all; 5 = very much)       |
| `q4_trust`       | System reacted as intended (1 = not at all; 5 = very much)               |
| `q5_would_use`   | Would use this in a real system (1 = definitely not; 5 = definitely yes) |
| `comment`        | Free-form qualitative note (Czech)                                       |


## Headline observations from the 15 supplementary responses

These are descriptive notes about the five-person sample, not claims about the wider population:

- All three modalities scored ≥ 3 on every dimension; no participant flagged the interface as unusable on default settings.
- **Toggles** scored highest on `q2_controls` (ease of use, mean 4.8) but lowest on `q3_control` (perceived granularity, mean 2.8).
- **Sliders** scored highest on `q3_control` (mean 4.8) and `q5_would_use` (mean 4.8); the comments consistently mention immediate feedback from default values.
- **Text** had the widest spread on `q4_trust` (mean 2.8, range 2–3); participants liked the expressiveness but flagged occasional surprising matches, which matches the lexical-resolver discussion in [`docs/tech-docs.md` Section 11](../docs/tech-docs.md#11-limitations-and-future-work).

The paper study (Section 1) is the authoritative source for any claim about effect sizes, comparisons across modalities, or generalisation.
