"""
MCP Server Package
Multi-user compatible Python package initialization.
Handles the mcp-server folder with relative internal imports.
"""

import sys
import os

# Get the directory of this module
module_dir = os.path.dirname(os.path.abspath(__file__))

# Add the module directory to sys.path temporarily to handle relative imports
sys.path.insert(0, module_dir)

try:
    # Import app from server module
    # The server.py uses relative imports like 'from model_backends import...'
    # So we need to be in the right directory
    from server import app
finally:
    # Remove from path to clean up
    if module_dir in sys.path:
        sys.path.remove(module_dir)

__all__ = ['app']
