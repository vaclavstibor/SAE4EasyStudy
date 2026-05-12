import flask
from flask_login import current_user, login_required, login_user, logout_user
from is_safe_url import is_safe_url
from werkzeug.security import check_password_hash, generate_password_hash

from server.platform.persistence.base_models import LoginForm, SignupForm, User
from server.platform.persistence.db import db

auth = flask.Blueprint("auth", __name__)


@auth.route("/login", methods=["POST"])
def login_post():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.get(form.email.data)
        if user:
            if check_password_hash(user.password, form.password.data):
                user.authenticated = True
                db.session.add(user)
                db.session.commit()
                login_user(user, remember=True)
            else:
                return "Invalid username or password"

            next_url = flask.request.args.get("next")
            print(f"next={next_url}")
            if next_url and not is_safe_url(next_url, {}):
                return flask.abort(400)

            return flask.redirect(next_url or flask.url_for("main.administration"))
        return "Invalid username or password"
    return "Invalid username or password"


@auth.route("/login")
def login():
    return flask.render_template("login.html", authenticated=current_user.is_authenticated)


@auth.route("/logout")
@login_required
def logout():
    user = current_user
    user.authenticated = False
    db.session.add(user)
    db.session.commit()
    logout_user()
    return flask.redirect(flask.url_for("auth.login"))


@auth.route("/signup")
def signup():
    return flask.render_template("signup.html")


@auth.route("/signup", methods=["POST"])
def signup_post():
    form = SignupForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            return "The user already exists"

        new_user = User(
            email=form.email.data,
            password=generate_password_hash(form.password.data),
        )
        db.session.add(new_user)
        db.session.commit()
        return flask.redirect(flask.url_for("auth.login"))
    print(form.errors)
    return "The password is too short"
