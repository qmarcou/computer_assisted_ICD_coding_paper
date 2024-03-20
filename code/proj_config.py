from pathlib import Path
import sys
import json

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_datadir() -> Path:
    return get_project_root() / "data"

def get_codedir() -> Path:
    return get_project_root() / "code"

def get_libdir() -> Path:
    return get_codedir() / "libs"

def get_modelsdir() -> Path:
    return get_project_root() / "models"

def get_APkeyfile() -> Path:
    return get_codedir() / "APcreds.key"

# Add the libs folder to python path
sys.path.insert(0,str(get_libdir()))
# Add keras_utils path
sys.path.insert(0,str(get_libdir())+"/keras_extra")

