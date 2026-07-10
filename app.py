import os
import sys

# Tell Python it's allowed to look inside the dashboard folder for files
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'dashboard')))

# Import the dashboard app code directly to execute it without starting a new server instance
if __name__ == "__main__":
    import dashboard.app
