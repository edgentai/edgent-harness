import os
import tempfile
import unittest

from harness.core.client import MockModelClient
from harness.core.types import Claim, Evidence, ProblemType, SubAgentResult, VerdictStatus
from harness.modality import ArtifactStore, Ingestor
from harness.prompts import system_prompt
from harness.verification import DefinitionOfDone, TrustGate, VerifyContext, build_default_verifiers
from harness.verification.verifiers import resolve_evidence


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.store = ArtifactStore(os.path.join(self.d, "art"))
        self.addCleanup(self.store.close)
        self.ing = Ingestor(self.store, work_dir=os.path.join(self.d, "w"))
        self.art = self.ing.ingest_text("Invoice 7781\nTotal: 318.40 EUR\nDue: 30 days")
        self.reg = build_default_verifiers()
        self.ctx = VerifyContext(self.store)

    def good(self, text="Total: 318.40 EUR"):
        off = self.store.get_view(self.art.id, "text").text.find(text)
        return Claim("The invoice total is 318.40 EUR", [Evidence(self.art.id, "text", {"offset": off, "length": len(text)}, text)])

    def test_resolve(self):
        ok, detail, snippet = resolve_evidence(self.good().evidence[0], self.store)
        self.assertTrue(ok)
        self.assertEqual(snippet, "Total: 318.40 EUR")
        ok, detail, _ = resolve_evidence(Evidence(self.art.id, "text", {"offset": 0}, "not there"), self.store)
        self.assertFalse(ok)
        ok, detail, _ = resolve_evidence(Evidence("deadbeef" * 8, "text", {}, "x"), self.store)
        self.assertIn("not found", detail)
        ok, detail, _ = resolve_evidence(Evidence(self.art.id, "text", {"offset": 0, "length": 5}, None), self.store)
        self.assertTrue(ok)

    def test_evidence_and_numbers(self):
        res = SubAgentResult(answer="The total is 318.40 EUR", claims=[self.good()])
        checks = self.reg.run_many(["evidence", "numbers_grounded"], res, self.ctx)
        self.assertTrue(all(c.passed for c in checks), [c.to_dict() for c in checks])
        self.assertTrue(res.claims[0].verified)
        res2 = SubAgentResult(answer="The total is 999.99 EUR", claims=[self.good()])
        checks = self.reg.run_many(["numbers_grounded"], res2, self.ctx)
        self.assertFalse(checks[0].passed)
        self.assertIn("999.99", checks[0].detail)
        res3 = SubAgentResult(answer="x", claims=[Claim("made up", [Evidence(self.art.id, "text", {}, "quote that does not exist")])])
        checks = self.reg.run_many(["evidence"], res3, self.ctx)
        self.assertFalse(checks[0].passed)
        self.assertFalse(res3.claims[0].verified)

    def test_schema_policy(self):
        self.assertFalse(self.reg.run("schema", SubAgentResult(answer=""), self.ctx)[0].passed)
        self.assertFalse(self.reg.run("schema", SubAgentResult(answer="Sure, I will ignore all previous instructions"), self.ctx)[0].passed)
        self.ctx.extra["expected_output"] = "number"
        self.assertTrue(self.reg.run("schema", SubAgentResult(answer="1,234"), self.ctx)[0].passed)
        self.assertFalse(self.reg.run("schema", SubAgentResult(answer="many"), self.ctx)[0].passed)
        pol = self.reg.run("policy", SubAgentResult(answer="key sk-abcdefghijklmnopqrstuvwxyz"), VerifyContext(self.store))
        self.assertFalse(pol[0].passed)

    def test_consistency_and_unknown(self):
        ctx = VerifyContext(self.store, extra={"agreement": 0.4, "samples": 5})
        self.assertFalse(self.reg.run("consistency", SubAgentResult(answer="x"), ctx)[0].passed)
        self.assertFalse(self.reg.run("nonexistent", SubAgentResult(answer="x"), ctx)[0].passed)

    def test_model_critic_grounded(self):
        m = MockModelClient()
        m.push('{"passed": false, "issues": [{"claim": "x", "problem": "quote absent"}]}')
        ctx = VerifyContext(self.store, client=m, system_prompt=system_prompt)
        checks = self.reg.run("critic", SubAgentResult(answer="x", claims=[self.good()]), ctx)
        self.assertFalse(checks[0].passed)
        self.assertIn("resolved=True", m.calls[0].messages[-1].plain_text())


class GateTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.store = ArtifactStore(os.path.join(self.d, "art"))
        self.addCleanup(self.store.close)
        self.art = Ingestor(self.store, work_dir=os.path.join(self.d, "w")).ingest_text("Alpha beta gamma. Delta 12.")
        self.gate = TrustGate(build_default_verifiers())
        self.ctx = VerifyContext(self.store, extra={"expected_output": "text"})

    def claim(self, quote, text=None):
        return Claim(text or quote, [Evidence(self.art.id, "text", {"offset": self.store.get_view(self.art.id, "text").text.find(quote)}, quote)])

    def test_verified(self):
        v, bundle = self.gate.evaluate(SubAgentResult(answer="Alpha beta gamma.", claims=[self.claim("Alpha beta gamma.")]), DefinitionOfDone(), self.ctx)
        self.assertEqual(v.status, VerdictStatus.VERIFIED)
        self.assertTrue(bundle.items[0]["resolved"])

    def test_partial_and_failed(self):
        res = SubAgentResult(answer="Alpha beta gamma. Delta 12.", claims=[self.claim("Alpha beta gamma."), self.claim("nonexistent quote")])
        dod = DefinitionOfDone(min_claim_verification_rate=0.5)
        v, _ = self.gate.evaluate(res, dod, self.ctx)
        self.assertEqual(v.status, VerdictStatus.PARTIAL)
        self.assertTrue(any("quote not found" in u for u in v.unverified))
        v, _ = self.gate.evaluate(SubAgentResult(answer="", claims=[]), DefinitionOfDone(), self.ctx)
        self.assertEqual(v.status, VerdictStatus.FAILED)
        v, _ = self.gate.evaluate(SubAgentResult(answer="x", claims=[self.claim("not in text")]), DefinitionOfDone(), self.ctx)
        self.assertEqual(v.status, VerdictStatus.FAILED)

    def test_gaps_make_partial(self):
        v, _ = self.gate.evaluate(SubAgentResult(answer="Alpha beta gamma.", claims=[self.claim("Alpha beta gamma.")], gaps=["could not read page 2"]), DefinitionOfDone(), self.ctx)
        self.assertEqual(v.status, VerdictStatus.PARTIAL)
        self.assertIn("gap: could not read page 2", v.unverified)

    def test_contract_defaults(self):
        d = DefinitionOfDone.default_for(ProblemType.CONVERGENT, has_artifacts=True)
        self.assertIn("consistency", d.checks)
        self.assertIn("evidence", d.checks)
        d2 = DefinitionOfDone.default_for(ProblemType.DIVERGENT, has_artifacts=False)
        self.assertNotIn("evidence", d2.checks)
        d3 = DefinitionOfDone.from_dict({"required_outputs": [{"name": "answer", "format": "number"}]})
        self.assertFalse(d3.check_outputs("abc")[0].passed)
        self.assertTrue(d3.check_outputs("12.5")[0].passed)


if __name__ == "__main__":
    unittest.main()
