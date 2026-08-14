"""
Pytest Configuration File
Adds the workspace root directory to sys.path to enable imports of the ai_models package during testing.
"""
import os
import sys

# Locate and append workspace root directory dynamically
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
