import nox


@nox.session(venv_backend="conda", python=["3.10", "3.11", "3.12"], reuse_venv=False)
def test_install(session):
    session.install("-e", ".", silent=True)
    pip_show_output = session.run("pip", "show", "dojo", silent=True)
    if "not found" in pip_show_output:
        raise Exception(
            "Missing dojo package, this means installation probably failed."
        )
