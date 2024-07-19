import nox

SUPPORTED_PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]
VENV = "conda"
REUSE_VENV = True


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, venv_backend=VENV, reuse_venv=REUSE_VENV)
def compatibility(session):
    session.install("-e", ".", silent=True)
    pip_show_output = session.run("pip", "show", "dojo", silent=True)
    if "not found" in pip_show_output:
        raise Exception(
            "Missing dojo package, this means installation probably failed."
        )


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, venv_backend=VENV, reuse_venv=REUSE_VENV)
def unit_tests(session):
    session.install("-e", ".", silent=True)
    session.install("pytest", silent=True)
    session.run("pytest")
