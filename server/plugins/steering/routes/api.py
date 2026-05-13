"""Participant-facing elicitation API endpoints."""

import traceback

from flask import jsonify, request, session

from server.platform.shared.common import get_tr

from ..plugin import bp, get_lang, languages
from ..service import audit


@bp.route("/get-initial-data", methods=["GET"])
def get_initial_data():
    try:
        if "elicitation_movies" not in session:
            session["elicitation_movies"] = []
        el_movies = session["elicitation_movies"]

        from server.plugins.utils.preference_elicitation import load_data_2

        rows = load_data_2(el_movies)
        tr = get_tr(languages, get_lang())
        for row in rows:
            if "genres" in row:
                row["movie"] = row["movie"] + " " + "|".join(
                    [tr(f"genre_{genre.lower()}") for genre in row["genres"]]
                )
        el_movies.extend(rows)
        session["elicitation_movies"] = el_movies
        return jsonify(el_movies)
    except Exception as exc:
        print(f"Error in get_initial_data: {exc}")
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@bp.route("/item-search", methods=["GET"])
def item_search():
    pattern = request.args.get("pattern")
    if not pattern:
        return jsonify([])
    try:
        from server.plugins.utils.preference_elicitation import search_for_movie

        lang = get_lang()
        tr = get_tr(languages, lang) if lang != "en" else None
        results = search_for_movie("movie", pattern, tr)
        participation_id = session.get("participation_id")
        if participation_id:
            audit.record_event(
                "elicitation-search",
                participation_id=participation_id,
                allow_no_approach=True,
                source="search",
                search_query=pattern,
                raw_payload={
                    "query": pattern,
                    "result_count": len(results),
                    "results": results[:10],
                },
                approach_order=session.get("approach_order"),
            )
        return jsonify(results)
    except Exception as exc:
        print(f"Error in item_search: {exc}")
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
