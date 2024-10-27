import pytest

@pytest.fixture(autouse=True)
def delete_imports():
    yield
    import sys
    for key in list(sys.modules.keys()):
        if key.startswith("jcm"):
            del sys.modules[key]