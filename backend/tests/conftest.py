import pytest

from backend.common.config import load_config
from backend.tools.synth import DEFAULT_GESTURES, generate_session


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synth_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    return generate_session(root, DEFAULT_GESTURES, reps=4, free_s=40.0, seed=123)


@pytest.fixture(scope="session")
def sess(synth_path):
    from backend.data.session import load_session
    return load_session(synth_path)
