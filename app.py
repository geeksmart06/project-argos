import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    # Pointing directly to the root dashboard folder seen in your Finder
    sys.argv = ["streamlit", "run", "dashboard/main.py"]
    sys.exit(stcli.main())
