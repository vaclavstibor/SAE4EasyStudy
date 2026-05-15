# Participant Manual

This is the participant experience for an SAE Steering study. It is the user-facing companion to [`admin-manual.md`](admin-manual.md). Times are approximate and depend on study configuration.

## Contents

1. [Join the study](#1-join-the-study--1-minute)  
   Consent + intro, session starts.
2. [Preference elicitation](#2-preference-elicitation--3-minutes)  
   Pick baseline movies and search titles.
3. [Resolution gate](#3-resolution-gate)  
   Minimum screen-size check (if configured).
4. [Steering iterations](#4-steering-iterations--10-minutes)  
   Review → (optional) steer → approve → refresh.
5. [Per-approach questionnaire](#5-per-approach-questionnaire--1-minute-optional)  
   Optional questionnaire between approaches.
6. [Final questionnaire](#7-final-questionnaire--2-minutes)  
   End-of-study questionnaire and finish.
7. [Edge cases](#edge-cases-the-ui-handles)  
   No-match, refresh, resume behaviour.

## 1. Join the study — ~1 minute

- The invitation link points at `/sae_steering/join?guid=<study-guid>`.
- The participant lands on a consent + study-intro page describing what the next ~15 minutes will look like.
- A unique participation `uuid` is assigned and stored in the session.
- If a Prolific ID is supplied via URL parameters, it is recorded for later payment reconciliation.

## 2. Preference elicitation — ~3 minutes

- The participant picks at least N movies they like from a paginated grid of popular titles.
- Each pick/deselect is recorded.
- A free-text search lets the participant find specific titles (`/movie-search`).
- Selected movies are used as the recommender's seed.

## 3. Resolution gate

If the study requires a minimum screen resolution, an overlay blocks the study UI until the participant resizes their window. The minimum is set by the researcher in the study config.

## 4. Steering iterations — ~10 minutes

For each approach the participant sees `num_iterations` rounds (typically 3). Each round has four phases:

### 4.1 Review recommendations

The participant browses the recommended movies and may click a like / dislike thumb on each card they want to flag.

### 4.2 (Optional) Steer

Depending on the approach, the participant may use one or more of:

- **Sliders / toggles.** Adjust feature strengths in either direction. Continuous sliders or three-state toggles (boost / off / suppress) are available depending on the approach.
- **Text prompt** (FR-09). Describe the desired change in natural language (max 200 characters). The previous prompt is shown above the input as a "You said before" hint so the participant can extend their intent across iterations.
- **Feature search.** Type a keyword to find specific concepts. Matching feature clusters can be added to the slider grid.
- **Reset** (FR-12). A dedicated "Reset all controls" button. One click clears all adjustments.

### 4.3 Approve preferences

A single click confirms the current preference set and unlocks the "Get next recommendations" button.

### 4.4 Get the next recommendations

Fetches a new set of recommendations using the cumulative adjustments.

When the configured number of iterations is reached, the iteration controls lock and the participant moves on.

## 5. Per-approach questionnaire — ~1 minute (optional)

If the study is configured with a phase questionnaire, it is shown between approaches.

## 6. Approach switch (sequential studies only)

If the study has multiple approaches, the participant sees a short transition page and the next approach starts.

## 7. Final questionnaire — ~2 minutes

A single final questionnaire is shown at the end. Typical content: comparative preference, SUS-like usability questions, free-text feedback.

## 8. Finish

- The participant clicks "Finish".
- The participation is marked completed.
- If a Prolific completion code is configured, it is shown for the participant to paste back into Prolific.

## Edge cases the UI handles

- **Ambiguous text prompt** (NFR-12). When the parser cannot match any cluster, the UI shows "We could not match your text to any feature, try different wording" and the participant's existing adjustments are preserved. The query is still recorded for offline analysis. See [`design-decisions.md` Section 7](design-decisions.md#7-nfr-12-text-steering-ambiguity-degrades-gracefully).
- **Browser refresh mid-iteration.** Iteration state is autosaved on every meaningful interaction. A refresh resumes where the participant left off.
- **Tab close mid-study.** The participation row stays open. The participant can resume from the join link as long as the study is still active.
