"""Trust gate (§12.2): the final answer is released only against the contract."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.types import Check, SubAgentResult, Verdict, VerdictStatus
from .contract import DefinitionOfDone
from .verifiers import VerifierRegistry, VerifyContext, resolve_evidence


@dataclass
class EvidenceBundle:
    items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"items": self.items}


class TrustGate:
    ORDER = ["schema", "evidence", "numbers_grounded", "computation", "tests", "consistency", "critic", "judge", "policy"]

    def __init__(self, verifiers: VerifierRegistry) -> None:
        self.verifiers = verifiers

    def evaluate(self, result: SubAgentResult, contract: DefinitionOfDone, ctx: VerifyContext) -> tuple[Verdict, EvidenceBundle]:
        checks: list[Check] = contract.check_outputs(result.answer)
        wanted = [c for c in self.ORDER if c in contract.checks] + [c for c in contract.checks if c not in self.ORDER]
        for name in wanted:
            checks.extend(self.verifiers.run(name, result, ctx))
        bundle = self._bundle(result, ctx)
        verdict = self._decide(checks, result, contract)
        return verdict, bundle

    def _bundle(self, result: SubAgentResult, ctx: VerifyContext) -> EvidenceBundle:
        items = []
        for i, c in enumerate(result.claims):
            for ev in c.evidence:
                ok, detail, snippet = resolve_evidence(ev, ctx.store)
                items.append({"claim": c.text, "claim_index": i, "citation": ev.cite(), "artifact_id": ev.artifact_id, "kind": ev.kind,
                              "locator": ev.locator, "resolved": ok, "detail": detail, "content": snippet})
        return EvidenceBundle(items)

    @staticmethod
    def _decide(checks: list[Check], result: SubAgentResult, contract: DefinitionOfDone) -> Verdict:
        failed = [c for c in checks if not c.passed]
        hard_fail = [c for c in failed if c.verifier in ("contract", "policy", "schema", "tests", "computation")]
        claim_checks = [c for c in checks if c.verifier == "evidence" and c.name.startswith("evidence:claim")]
        verified_rate = (sum(1 for c in claim_checks if c.passed) / len(claim_checks)) if claim_checks else 1.0
        unverified = [c.detail for c in failed]
        unverified += [f"gap: {g}" for g in result.gaps]
        if hard_fail:
            return Verdict(VerdictStatus.FAILED, checks, unverified, f"{len(hard_fail)} hard check(s) failed: " + "; ".join(c.name for c in hard_fail))
        if not failed and not result.gaps:
            return Verdict(VerdictStatus.VERIFIED, checks, [], f"all {len(checks)} checks passed")
        if verified_rate >= contract.min_claim_verification_rate and all(c.verifier in ("evidence", "numbers_grounded", "consistency", "critic", "judge") for c in failed):
            return Verdict(VerdictStatus.PARTIAL, checks, unverified, f"{len(failed)} soft check(s) failed or gaps declared; claim verification rate {verified_rate:.0%}")
        if verified_rate > 0 and verified_rate >= contract.min_claim_verification_rate * 0.5:
            return Verdict(VerdictStatus.PARTIAL, checks, unverified, f"claim verification rate {verified_rate:.0%} below contract target {contract.min_claim_verification_rate:.0%}")
        return Verdict(VerdictStatus.FAILED, checks, unverified, f"claim verification rate {verified_rate:.0%}; evidence does not support the answer")
