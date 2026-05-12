from flask_wtf import FlaskForm
from wtforms import EmailField, PasswordField
from wtforms.validators import DataRequired, Length

from .db import db


class User(db.Model):
    __tablename__ = "user"

    email = db.Column(db.String, primary_key=True)
    password = db.Column(db.String)
    authenticated = db.Column(db.Boolean, default=False)
    admin = db.Column(db.Boolean, default=False)

    def is_active(self):
        return True

    def get_id(self):
        return self.email

    @property
    def is_authenticated(self):
        return self.authenticated

    def is_anonymous(self):
        return False

    def is_admin(self):
        return self.admin


class UserStudy(db.Model):
    __tablename__ = "userstudy"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    creator = db.Column(db.String, db.ForeignKey("user.email"))
    guid = db.Column(db.String)
    parent_plugin = db.Column(db.String)
    settings = db.Column(db.String)
    time_created = db.Column(db.DateTime)
    active = db.Column(db.Boolean)
    initialized = db.Column(db.Boolean)
    initialization_error = db.Column(db.String, default=None)

    def __str__(self):
        return (
            f"id={self.id},creator={self.creator},guid={self.guid},"
            f"time_created={self.time_created},settings={self.settings}"
        )


class Participation(db.Model):
    __tablename__ = "participation"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    participant_email = db.Column(db.String)
    age_group = db.Column(db.String)
    gender = db.Column(db.String)
    education = db.Column(db.String)
    ml_familiar = db.Column(db.Boolean)
    user_study_id = db.Column(
        db.Integer,
        db.ForeignKey("userstudy.id", ondelete="CASCADE"),
    )
    time_joined = db.Column(db.DateTime)
    time_finished = db.Column(db.DateTime)
    uuid = db.Column(db.String)
    language = db.Column(db.String)
    extra_data = db.Column(db.String)


class LoginForm(FlaskForm):
    email = EmailField("email", validators=[DataRequired("missing mail")])
    password = PasswordField(
        "password",
        validators=[DataRequired("missing password"), Length(6, 128, "short password")],
    )


class SignupForm(FlaskForm):
    email = EmailField("email", validators=[DataRequired("missing mail")])
    password = PasswordField(
        "password",
        validators=[DataRequired("missing password"), Length(6, 128, "short password")],
    )


class Interaction(db.Model):
    __tablename__ = "interaction"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    participation = db.Column(
        db.Integer,
        db.ForeignKey("participation.id", ondelete="CASCADE"),
    )
    interaction_type = db.Column(db.String)
    time = db.Column(db.DateTime)
    data = db.Column(db.String)


class Message(db.Model):
    __tablename__ = "message"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    participation = db.Column(
        db.Integer,
        db.ForeignKey("participation.id", ondelete="CASCADE"),
        nullable=True,
    )
    time = db.Column(db.DateTime)
    data = db.Column(db.String)
