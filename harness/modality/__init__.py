"""Modality layer (§5): every input becomes a citable Artifact with derived text views."""
from .artifacts import Artifact, ArtifactStore, View  # noqa: F401
from .sidecars import Sidecars  # noqa: F401
from .ingest import Ingestor, sniff_mime, IngestError  # noqa: F401
from .budgeter import ModalityBudgeter, BudgetDecision  # noqa: F401
from .index import ModalityIndex  # noqa: F401
