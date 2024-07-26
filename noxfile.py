import nox

SUPPORTED_PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]
VENV_BACKEND = "uv|conda|venv"
REUSE_VENV = True
# specify tests under a single session, otherwise "compatibility" and "unit_tests" will use different venvs
VENV_NAME = "tests_session"

nox.options.reuse_existing_virtualenvs = True


@nox.session(
    python=SUPPORTED_PYTHON_VERSIONS,
    venv_backend=VENV_BACKEND,
    reuse_venv=REUSE_VENV,
    name=VENV_NAME,
)
def compatibility(session):
    session.install("-e", ".[test]", silent=False)
    pip_show_output = session.run("pip", "show", "dojo", silent=True)
    if "not found" in pip_show_output:
        raise Exception(
            "Missing dojo package, this means installation probably failed."
        )


@nox.session(
    python=SUPPORTED_PYTHON_VERSIONS,
    venv_backend=VENV_BACKEND,
    reuse_venv=REUSE_VENV,
    name=VENV_NAME,
)
def unit_tests(session):
    session.install("-e", ".[test]", silent=False)
    session.run("pytest", "-s", "-v", "tests/unit_testing")
