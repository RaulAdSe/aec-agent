"""Streamlit app entry point."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the app
from src.app import main

if __name__ == "__main__":
    main()