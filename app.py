import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    # Pointing to the app.py file inside the dashboard folder
    sys.argv = ["streamlit", "run", "dashboard/app.py"]
    sys.exit(stcli.main())
