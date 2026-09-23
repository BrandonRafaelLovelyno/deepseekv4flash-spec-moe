"""Analysis studies, one per module.

To add a study: create ``<name>.py`` here with an ``Analysis`` subclass, then
append an instance to ``ANALYSES`` in ``4_analysis/main.py``. Order matters only
when a study reads state an earlier one left in the shared ``AnalysisContext``.
"""

from analyses.base import Analysis
from analyses.coverage import CoverageAnalysis
from analyses.decode_miss import DecodeMissAnalysis
from analyses.expert_ranking import ExpertRankingAnalysis
from analyses.prefill_miss import PrefillMissAnalysis
from analyses.token_miss import TokenMissAnalysis

__all__ = [
    "Analysis",
    "CoverageAnalysis",
    "DecodeMissAnalysis",
    "ExpertRankingAnalysis",
    "PrefillMissAnalysis",
    "TokenMissAnalysis",
]
