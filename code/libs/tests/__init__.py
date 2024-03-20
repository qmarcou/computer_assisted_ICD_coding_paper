import sys
from pathlib import Path
import os
testdir = os.path.dirname(__file__)
srcdir = testdir+'/..'
sys.path.insert(0, str(Path(os.path.abspath(srcdir))))