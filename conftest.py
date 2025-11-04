import pytest
import subprocess
import sys

@pytest.fixture(autouse=True)
def clean_jcm_imports():
    yield
    keys_to_delete = {key for key in sys.modules if key == "jcm" or key.startswith("jcm.")}
    for key in keys_to_delete:
        del sys.modules[key]

@pytest.fixture(autouse=True)
def memory_before_marked_tests(request):
    if "memorycheck" in request.keywords:
        subprocess.run(["free", "-h"], check=False)
    yield