"""Cloud entrypoint: sets safe defaults before importing the original UI."""
import os
import runpy
from pathlib import Path
os.environ["COURTSIDE_IQ_CLOUD"] = "1"
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
