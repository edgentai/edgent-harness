import os
import tempfile
import unittest

from harness.core.config import ModalityLimits, ModelCapabilities
from harness.core.types import Modality
from harness.modality import ArtifactStore, Ingestor, ModalityBudgeter, ModalityIndex, sniff_mime
from harness.modality import audio as audio_mod
from harness.modality import image as image_mod
from harness.modality import text as text_mod
from harness.modality.sidecars import Sidecars
from tests.helpers import have_ffmpeg, png_bytes, write_mp4, write_png, write_text, write_wav


class SniffTests(unittest.TestCase):
    def test_magic(self):
        self.assertEqual(sniff_mime(png_bytes()[:64]), ("image/png", Modality.IMAGE))
        self.assertEqual(sniff_mime(b"RIFF\x00\x00\x00\x00WAVEfmt ")[0], "audio/wav")
        self.assertEqual(sniff_mime(b"\x00\x00\x00\x18ftypmp42")[1], Modality.VIDEO)
        self.assertEqual(sniff_mime(b"hello.png is not an image")[1], Modality.TEXT)
        self.assertEqual(sniff_mime(b'{"a":1}')[0], "application/json")
        self.assertEqual(sniff_mime(b"\x00\x01\x02binary")[1], Modality.UNKNOWN)


class TextTests(unittest.TestCase):
    def test_structure_chunks_locate(self):
        t = "# H1\n\npara one.\n\n| a | b |\n|---|---|\n\n## H2\n" + "word " * 1000
        s = text_mod.structure(t)
        self.assertEqual([h["title"] for h in s["headings"]], ["H1", "H2"])
        self.assertEqual(len(s["tables"]), 1)
        chunks = text_mod.chunk(t, size=500, overlap=50)
        self.assertGreater(len(chunks), 3)
        for c in chunks:
            self.assertEqual(t[c["offset"] : c["offset"] + c["length"]], c["text"])
        self.assertEqual(text_mod.locate(t, "para one."), t.find("para one."))
        self.assertNotEqual(text_mod.locate(t, "PARA   ONE."), -1)
        self.assertEqual(text_mod.locate(t, "absent phrase"), -1)


class ImageTests(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(image_mod.dimensions(png_bytes(7, 5)), (7, 5))
        gif = b"GIF89a" + (300).to_bytes(2, "little") + (200).to_bytes(2, "little")
        self.assertEqual(image_mod.dimensions(gif), (300, 200))
        jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\xff\xc0\x00\x11\x08" + (120).to_bytes(2, "big") + (160).to_bytes(2, "big")
        self.assertEqual(image_mod.dimensions(jpeg), (160, 120))
        self.assertIsNone(image_mod.dimensions(b"junk"))


class AudioTests(unittest.TestCase):
    def test_probe_and_split(self):
        d = tempfile.mkdtemp()
        p = write_wav(d, seconds=3.0)
        info = audio_mod.probe_wav(p)
        self.assertAlmostEqual(info["duration_s"], 3.0, places=2)
        chunks = audio_mod.split_on_silence(p, max_chunk_s=1.2, min_silence_s=0.3)
        self.assertGreaterEqual(len(chunks), 2)
        # a cut must land inside the silent region (1s..2s), not mid-tone
        cuts = [c["end"] for c in chunks[:-1]]
        self.assertTrue(any(1.0 <= c <= 2.0 for c in cuts), cuts)
        self.assertIn("(low confidence)", audio_mod.transcript_text([{"start": 0, "end": 1, "text": "hi", "conf": 0.3}]))


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.store = ArtifactStore(os.path.join(self.d, "art"))
        self.addCleanup(self.store.close)
        self.ing = Ingestor(self.store, ModalityLimits(), Sidecars(), os.path.join(self.d, "work"))

    def test_text_views_and_dedup(self):
        a = self.ing.ingest_text("# T\n\nThe total is 42.")
        b = self.ing.ingest_text("# T\n\nThe total is 42.")
        self.assertEqual(a.id, b.id)
        self.assertEqual({v.name for v in self.store.views(a.id)}, {"text", "structure", "pages", "chunks", "summary"})
        self.assertEqual(self.store.get(a.id[:12]).id, a.id)

    def test_image_views_and_provenance(self):
        a = self.ing.ingest_path(write_png(self.d, w=6, h=4))
        self.assertEqual(a.modality, Modality.IMAGE)
        self.assertEqual(self.store.get_view(a.id, "metadata").data()["width"], 6)
        self.assertIsNotNone(self.store.get_view(a.id, "text"))
        if have_ffmpeg():
            prev = self.store.find_derived(a.id, "preview", {"max_side": 768})
            self.assertIsNotNone(prev)
            self.assertEqual([x.id for x in self.store.provenance(prev.id)], [prev.id, a.id])

    def test_audio_honest_without_stt(self):
        a = self.ing.ingest_path(write_wav(self.d))
        s = self.store.get_view(a.id, "summary").data()
        self.assertFalse(s["transcript_available"])
        self.assertIn("no transcript", self.store.get_view(a.id, "text").text)

    def test_audio_with_stt_sidecar(self):
        side = Sidecars(stt=lambda p: {"available": True, "text": "hello world", "segments": [{"start": 0.0, "end": 1.0, "text": "hello world", "conf": 0.9}]})
        ing = Ingestor(self.store, ModalityLimits(), side, os.path.join(self.d, "work2"))
        a = ing.ingest_path(write_wav(self.d, "b.wav"))
        self.assertIn("hello world", self.store.get_view(a.id, "transcript").text)

    @unittest.skipUnless(have_ffmpeg(), "ffmpeg required")
    def test_video(self):
        p = write_mp4(self.d)
        self.assertIsNotNone(p)
        a = self.ing.ingest_path(p)
        s = self.store.get_view(a.id, "summary").data()
        self.assertEqual(s["modality"], "video")
        self.assertGreaterEqual(s["keyframes"], 1)
        self.assertTrue(s["has_audio"])
        kf = self.store.get_view(a.id, "keyframes").data()
        self.assertEqual(self.store.find_derived(a.id, "frame", {"t": kf[0]["t"]}).id, kf[0]["artifact_id"])
        self.assertIn("artifact_id=", self.store.get_view(a.id, "text").text)

    def test_limits(self):
        from harness.modality.ingest import IngestError

        ing = Ingestor(self.store, ModalityLimits(max_file_bytes=10), Sidecars(), os.path.join(self.d, "w3"))
        with self.assertRaises(IngestError):
            ing.ingest_path(write_text(self.d, "big.txt", "x" * 100))

    def test_budgeter_and_index(self):
        a = self.ing.ingest_text("short text")
        img = self.ing.ingest_path(write_png(self.d, "i.png"))
        dec = ModalityBudgeter(self.store, ModelCapabilities(native_image=True, image_token_cost=1000), ModalityLimits()).decide([a, img], 10_000)
        self.assertIn("text", dec.inline_views[a.id])
        self.assertIn(img.id, dec.native_media)
        tiny = ModalityBudgeter(self.store, ModelCapabilities(native_image=True, image_token_cost=1000), ModalityLimits()).decide([a, img], 500)
        self.assertIn(img.id, tiny.downgraded)
        text = ModalityIndex(self.store).render([a, img], dec)
        self.assertIn(f"artifact_id={a.id}", text)
        self.assertIn("UNTRUSTED_DATA", text)
        parts = ModalityIndex(self.store).content_parts([img], dec, True)
        self.assertEqual(parts[0].type, "image")
        self.assertTrue(parts[0].data_url.startswith("data:image/"))


if __name__ == "__main__":
    unittest.main()
