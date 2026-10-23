"""Modality budgeter (§5.5): inline the summary, reference the detail."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.config import ModalityLimits, ModelCapabilities
from ..core.types import Modality
from ..core.utils import estimate_tokens
from .artifacts import Artifact, ArtifactStore


@dataclass
class BudgetDecision:
    inline_views: dict[str, list[str]] = field(default_factory=dict)  # artifact_id -> view names inlined
    referenced: dict[str, list[str]] = field(default_factory=dict)  # artifact_id -> view names left as references
    native_media: list[str] = field(default_factory=list)  # artifact IDs sent as native image/audio/video parts
    downgraded: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class ModalityBudgeter:
    def __init__(self, store: ArtifactStore, caps: ModelCapabilities, limits: ModalityLimits) -> None:
        self.store = store
        self.caps = caps
        self.limits = limits

    def decide(self, artifacts: list[Artifact], context_budget_tokens: int) -> BudgetDecision:
        d = BudgetDecision()
        spent = 0
        max_native_images = max(0, (context_budget_tokens // 4) // max(1, self.caps.image_token_cost))
        native_used = 0
        for art in artifacts:
            inl, ref = [], []
            for v in self.store.views(art.id):
                if v.kind != "text" or not v.text:
                    ref.append(v.name)
                    continue
                cost = estimate_tokens(v.text)
                if v.name in ("summary", "caption") or (v.name in ("text", "ocr", "transcript") and cost <= self.limits.inline_text_tokens and spent + cost <= context_budget_tokens):
                    inl.append(v.name)
                    spent += cost
                else:
                    ref.append(v.name)
            if art.modality == Modality.IMAGE and self.caps.native_image:
                if native_used < max_native_images and spent + self.caps.image_token_cost <= context_budget_tokens:
                    d.native_media.append(art.id)
                    native_used += 1
                    spent += self.caps.image_token_cost
                else:
                    d.downgraded.append(art.id)
                    d.notes.append(f"{art.short}: image kept as reference (token budget)")
            if art.modality == Modality.VIDEO and not self.caps.native_video:
                d.notes.append(f"{art.short}: video represented by keyframes + transcript; fetch frames on demand")
            if art.modality == Modality.AUDIO and not self.caps.native_audio:
                d.notes.append(f"{art.short}: audio represented by transcript; re-transcribe spans on demand")
            d.inline_views[art.id] = inl
            d.referenced[art.id] = ref
        if spent > context_budget_tokens:
            d.notes.append("budget exceeded even after downgrades; ask the user to narrow the task")
        d.estimated_tokens = spent
        return d
