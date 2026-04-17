#!/usr/bin/env python3
"""
Reconstruct a participant's chronological journey from an exported study JSON.

Usage:
    python scripts/reconstruct_journey.py <export.json> [--participant 0] [--verbose]

This thin CLI wraps the shared `journey` module used by the admin UI so the
on-screen journey view and the offline reconstruction stay perfectly in sync.
"""
import argparse
import json
import os
import sys

# Allow running both from the repo root and from inside server/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "server"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "server", "plugins"))

from sae_steering.journey import (  # noqa: E402
    NOISE_TYPES,
    PHASE_LABELS,
    build_journey,
    describe_interaction,
    fmt_time,
)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct participant journey")
    parser.add_argument("export_json", help="Path to exported study JSON")
    parser.add_argument("--participant", "-p", type=int, default=0, help="Participant index (default: 0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Include mouse/viewport noise events")
    args = parser.parse_args()

    with open(args.export_json) as f:
        data = json.load(f)

    participants = data.get("participants", [])
    if args.participant >= len(participants):
        print(f"Only {len(participants)} participant(s) in export.", file=sys.stderr)
        sys.exit(1)

    p = participants[args.participant]
    interactions = p.get("interactions", [])
    journey = build_journey(interactions, include_noise=args.verbose)

    conf = data.get("study_config", {})
    models = conf.get("models", [])

    print("=" * 80)
    print("STUDY JOURNEY RECONSTRUCTION")
    print("=" * 80)
    print(f"Study GUID:       {data.get('study_guid', '?')}")
    print(f"Exported at:      {data.get('exported_at', '?')}")
    print(f"Participant UUID: {p.get('uuid', '?')}")
    prolific = p.get("prolific") or {}
    if prolific.get("pid"):
        print(f"Prolific PID:     {prolific.get('pid', '?')}")
        print(f"Prolific Study:   {prolific.get('study_id', '?')}")
        print(f"Prolific Session: {prolific.get('session_id', '?')}")
    print(f"Joined:           {p.get('time_joined', '?')}")
    print(f"Finished:         {p.get('time_finished', '?')}")
    print(f"Total interactions: {journey['summary']['total_interactions']}")
    print()

    print("STUDY CONFIG:")
    print(f"  Comparison mode: {conf.get('comparison_mode', '?')}")
    print(f"  Iterations:      {conf.get('num_iterations', '?')}")
    print(f"  Recommendations: {conf.get('num_recommendations', '?')}")
    for i, m in enumerate(models):
        print(
            f"  Model {chr(65 + i)}: {m.get('name', '?')} | steering={m.get('steering_mode', '?')} | "
            f"sae={m.get('sae', '?')} | questionnaire={m.get('phase_questionnaire_file', 'none')}"
        )
    print(f"  Final questionnaire: {conf.get('questionnaire_file', 'none')}")
    print()

    print("CHRONOLOGICAL TIMELINE:")
    print("-" * 80)
    prev_section = None
    for entry in journey["timeline"]:
        if entry["section"] != prev_section:
            print()
            print(f"  --- {entry['section']} ---")
            prev_section = entry["section"]
        print(f"  [{entry['ts_short']}]  {entry['summary']}")

    print()
    print("=" * 80)
    print("SUMMARY OF INTERACTION COUNTS:")
    print("-" * 40)
    for t, c in journey["summary"]["type_counts"].items():
        print(f"  {t:40s} {c:>4}")

    if not args.verbose and journey["summary"]["noise_hidden"]:
        print(
            f"\n  ({journey['summary']['noise_hidden']} mouse/viewport noise events hidden, "
            f"use --verbose to show)"
        )

    print()
    print("=" * 80)
    print("PHASE-BY-PHASE SUMMARY:")
    print("-" * 80)
    for ph in journey["summary"]["phases"]:
        models_label = ", ".join(ph["models"]) if ph["models"] else "?"
        print(f"  Phase {ph['phase']} ({models_label}):")
        print(f"    Iterations used: {ph['iterations']}")
        print(f"    Likes: {ph['likes']}, Dislikes: {ph['dislikes']}")
        print(f"    Slider adjustments (total values): {ph['slider_adjustments']}")
        if ph["searches"]:
            print(f"    Feature searches: {ph['searches']}")
        print()


if __name__ == "__main__":
    main()
