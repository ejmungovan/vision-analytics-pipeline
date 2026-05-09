import runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
runpy.run_path(str(Path(__file__).parent / "app" / "streamlit_app.py"), run_name="__main__")
