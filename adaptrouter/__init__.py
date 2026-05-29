# adaptrouter/__init__.py
# This file makes adaptrouter a Python package.
# It exports the main class so developers can write:
#   from adaptrouter import AdaptRouter
# instead of:
#   from adaptrouter.core import AdaptRouter

from adaptrouter.core import AdaptRouter

__version__ = "0.1.0"
__author__  = "Shreya"
__all__     = ["AdaptRouter"]