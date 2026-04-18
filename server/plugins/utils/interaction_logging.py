import datetime
import json
from models import Interaction, Participation, Message
from app import db

def log_interaction(participation_id, interaction_type, **kwargs):
    # Defensive: the participation row may have been deleted (e.g. admin
    # removed a test run) while the user's browser still holds a session
    # cookie referencing its primary key.  Previously this path exploded
    # with AttributeError: 'NoneType' object has no attribute 'id' and
    # returned a 500 to the participant.  Drop the log silently instead —
    # the request handler above is expected to recover (create a fresh
    # participation, clear stale session keys) once it notices the loss.
    if participation_id is None:
        return
    participation = Participation.query.filter(Participation.id == participation_id).first()
    if participation is None:
        print(
            f"[log_interaction] Skipping '{interaction_type}': participation "
            f"id={participation_id} no longer exists."
        )
        return
    x = Interaction(
        participation = participation.id,
        interaction_type = interaction_type,
        time = datetime.datetime.utcnow(),
        data = json.dumps(kwargs, ensure_ascii=False)
    )
    db.session.add(x)
    db.session.commit() 

def log_message(participation_id, **kwargs):
    x = Message(
        time = datetime.datetime.utcnow(),
        data = json.dumps(kwargs, ensure_ascii=False),
        participation = participation_id
    )
    db.session.add(x)
    db.session.commit()

def study_ended(participation_id, **kwargs):

    p = Participation.query.filter(Participation.id == participation_id).first()
    if p is None:
        print(
            f"[study_ended] Skipping: participation id={participation_id} no longer exists."
        )
        return
    if p.time_finished:
        # This one was already marked as finished
        return

    x = Interaction(
        participation = p.id,
        interaction_type = "study-ended",
        time = datetime.datetime.utcnow(),
        data = json.dumps(kwargs, ensure_ascii=False)
    )

    db.session.add(x)
    db.session.commit()

    Participation.query.filter(Participation.id == participation_id).update({"time_finished": datetime.datetime.utcnow()})
    db.session.commit()