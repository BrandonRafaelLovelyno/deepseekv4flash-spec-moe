"""Base class for the analysis studies.

Each study lives in its own module under ``analyses/`` and subclasses
``Analysis``:

    class MyStudy(Analysis):
        name = "my_study"

        def run(self, ctx):
            logs = []
            out_dir = ctx.out_dir(self.name)
            ...  # read ctx, write artifacts under out_dir, yield events
            yield _emit(logs, "my_study did a thing")

A study is a Modal generator: it yields ``log`` / ``image`` / ``file`` events
(see ``helper.Event``) and writes its own artifacts into its own remote
subdirectory. To register it, append an instance to ``ANALYSES`` in
``4_analysis/main.py``; order matters only when a study reads state an earlier
one wrote into ``ctx``.
"""

from __future__ import annotations

import abc
from typing import ClassVar, Iterator

from helper import AnalysisContext, Event


class Analysis(abc.ABC):
    """One self-contained study over the shared ``AnalysisContext``."""

    name: ClassVar[str]

    @abc.abstractmethod
    def run(self, ctx: AnalysisContext) -> Iterator[Event]:
        """Replay/measure this study and stream its events.

        Args:
            ctx: Shared context; read earlier studies' results and write this
                study's own.
        """
        raise NotImplementedError
