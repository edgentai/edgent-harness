"""Edgent: a general-purpose agent harness with a deep education layer.

The model is the engine. This package is everything around it: the loop,
the context, the tools, the workspace, permissions, sessions, verification,
and the skills that hold the pedagogy. Dependencies point inward:
``domain`` imports nothing of ours, ``application`` imports ``domain``,
``adapters`` and ``tools`` import both, ``ui`` imports ``application``.
"""

__version__ = "0.1.0"
