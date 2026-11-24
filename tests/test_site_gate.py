"""The site gate: parsing the site's check output and repairing the flagged labels.

Fixtures are the real CI output from the deploys that failed between 14 Oct and
22 Nov 2026.
"""
from publish import site_gate

FITCHECK_OUTPUT = """
8 graphic(s) do not fit their container:

  how-content-discovery-works-streaming-apps-signal-sources-feeding-ranking-1bz8l.svg
    horizontal: "Personalized" spans [81.3..102.3], viewBox [0..100]

  how-to-add-multi-tenancy-to-saas-app-four-isolation-models-positioned-xxkm7.svg
    truncated: "Absolute separa…" was clipped to fit

  how-to-add-multi-tenancy-to-saas-app-four-isolation-models-positioned-xxkm7.svg
    truncated: "Low operational complex…" was clipped to fit

Shorten the offending label, or grow the shape in src/lib/graphics/.
"""

COVERS_OUTPUT = """
  ✗ how-to-add-read-receipts-to-chat-app-cover.svg
      "100 writes per 100 messages" collides with "500 writes per 100 messages"

test-covers: 1 failure(s) across 11 fixture case(s) and 38 published cover(s)
"""

QUADRANT = '''<QuadrantMap
  caption="The four isolation models positioned by operational cost"
  xAxis={{ low: "Low operational complexity", high: "High operational complexity" }}
  quadrants={{
    topRight: "DB-per-tenant (dedicated): Absolute separation, highest ops cost",
    bottomLeft: "Shared schema + RLS"
  }}
/>

Low operational complexity is what most teams want on day one.
'''


def test_parse_failures_groups_strings_under_their_file():
    found = site_gate.parse_failures(FITCHECK_OUTPUT)
    mt = "how-to-add-multi-tenancy-to-saas-app-four-isolation-models-positioned-xxkm7.svg"
    assert found[mt] == ["Absolute separa", "Low operational complex"]
    assert "Personalized" in found[
        "how-content-discovery-works-streaming-apps-signal-sources-feeding-ranking-1bz8l.svg"]


def test_parse_failures_reads_cover_collisions():
    found = site_gate.parse_failures(COVERS_OUTPUT)
    assert found["how-to-add-read-receipts-to-chat-app-cover.svg"] == [
        "100 writes per 100 messages", "500 writes per 100 messages"]


def test_offending_literals_match_clipped_and_joined_text():
    values = site_gate.offending_literals(QUADRANT, ["Absolute separa", "Low operational complex"])
    assert "DB-per-tenant (dedicated): Absolute separation, highest ops cost" in values
    assert "Low operational complexity" in values
    assert "Shared schema + RLS" not in values
    # value joined with its unit by the renderer
    body = '<BarChart unit="writes per 100 messages" data={[{ label: "10 msgs", value: 100 }]} />'
    assert site_gate.offending_literals(body, ["100 writes per 100 messages"]) == [
        "writes per 100 messages"]


def test_captions_are_never_offending():
    body = '<X caption="Low operational complexity wins" />'
    assert site_gate.offending_literals(body, ["Low operational complex"]) == []


def test_repair_without_model_shortens_only_the_props():
    report = site_gate.GateReport(ran=True, own={"x.svg": ["Low operational complex"]})
    fixed = site_gate.repair(QUADRANT, report, llm=None)
    assert fixed is not None
    assert 'low: "Low operational complexity"' not in fixed
    # prose and captions untouched
    assert "Low operational complexity is what most teams want" in fixed
    assert 'caption="The four isolation models positioned by operational cost"' in fixed


def test_repair_uses_model_text_when_it_fits():
    class FakeLLM:
        def complete_json(self, **_kw):
            return [{"id": 1, "text": "Low ops cost"}]

    report = site_gate.GateReport(ran=True, own={"x.svg": ["Low operational complex"]})
    fixed = site_gate.repair(QUADRANT, report, llm=FakeLLM())
    assert 'low: "Low ops cost"' in fixed


def test_repair_returns_none_when_nothing_maps():
    report = site_gate.GateReport(ran=True, own={"x.svg": ["Nowhere in the body"]})
    assert site_gate.repair(QUADRANT, report, llm=None) is None


def test_blocking_only_counts_this_posts_graphics(tmp_path, monkeypatch):
    # No node/scripts in tmp_path: the gate must skip rather than block.
    report = site_gate.run_checks(tmp_path, "some-slug")
    assert report.ran is False and report.blocking is False
