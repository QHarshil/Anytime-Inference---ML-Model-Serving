"""Model loading utilities. Submodules are imported lazily to avoid pulling in
torch at package-import time for callers that only need a single helper.
"""
