import os

PILLARX_PACKAGE_FOLDER = os.path.dirname(os.path.abspath(__file__))
_LD_LIBRARY_PATH = os.environ['LD_LIBRARY_PATH'].split(":")
_LD_LIBRARY_PATH.insert(0, PILLARX_PACKAGE_FOLDER)

os.environ['LD_LIBRARY_PATH'] = ":".join(_LD_LIBRARY_PATH)

from .pillarx import *
