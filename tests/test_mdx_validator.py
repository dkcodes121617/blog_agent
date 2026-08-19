"""The MDX contract, and the component-prop contract underneath it.

The validator is the last thing standing between an unattended model and the site's
`main` branch. Everything here is a rule some published post already broke.
"""
from __future__ import annotations

import pytest

from seo.mdx_validator import (
    COMPONENT_ITEM_KEYS,
    COMPONENT_PROPS,
    component_prop_errors,
    validate_mdx,
)


# ── Component prop contracts ─────────────────────────────────────────────────
# Three live posts carry a component the site cannot draw. The exporter logged a
# warning, the page component crashed the static export, and the deploy was down for
# three days before anyone connected the two.

def test_statgrid_with_data_prop_is_rejected():
    """<StatGrid data={…}> — live in task-management-software-...-cost-guide."""
    errors = component_prop_errors('<StatGrid caption="x" data={[{ label: "A", value: "Daily" }]} />')
    assert errors
    assert "`data=`" in errors[0] and "`stats=`" in errors[0]


def test_timeline_with_milestones_prop_is_rejected():
    """<Timeline milestones={…}> — live in when-to-replace-spreadsheets-with-custom-crm."""
    errors = component_prop_errors(
        '<Timeline caption="x" milestones={[{ date: "A", title: "B" }]} />')
    assert any("`milestones=`" in e and "`events=`" in e for e in errors)


def test_wrong_item_keys_are_rejected():
    """The container prop can be right while every entry inside it is wrong."""
    errors = component_prop_errors(
        '<Timeline caption="x" events={[{ date: "A", title: "B" }]} />')
    assert any("`title:`" in e and "`label:`" in e for e in errors)


def test_missing_data_prop_is_rejected():
    errors = component_prop_errors('<BarChart caption="x" unit="%" />')
    assert any("missing its `data=`" in e for e in errors)


@pytest.mark.parametrize("mdx", [
    '<StatGrid caption="x" stats={[{ label: "A", value: "Daily" }]} />',
    '<Timeline caption="x" events={[{ date: "A", label: "B" }]} />',
    '<BarChart caption="x" unit="%" data={[{ label: "A", value: 4 }]} />',
    '<CompareDiagram caption="x" columns={[{ title: "A", points: ["p"] }]} />',
    '<FlowDiagram caption="x" steps={[{ label: "A", sub: "b" }]} />',
    '<ConceptDiagram caption="x" nodes={[{ title: "A", sub: "b" }]} />',
    '<KeyTakeaways points={["a", "b"]} />',
    '<FAQ items={[{ q: "a", a: "b" }]} />',
])
def test_correct_components_pass(mdx):
    assert component_prop_errors(mdx) == [], f"false positive on: {mdx}"


def test_every_component_in_the_table_is_a_real_component():
    """A typo here silently disables the check for that component."""
    from seo.mdx_validator import ALLOWED_COMPONENTS
    assert set(COMPONENT_PROPS) <= ALLOWED_COMPONENTS
    assert set(COMPONENT_ITEM_KEYS) <= ALLOWED_COMPONENTS


# ── Policy integration ───────────────────────────────────────────────────────

def _shell(body: str) -> str:
    """A minimally valid post, so a policy test fails on policy and nothing else."""
    return (
        "A lead paragraph long enough to look like real prose rather than a stub, "
        "written the way the studio writes for someone planning a build.\n\n"
        '<KeyTakeaways points={["one", "two", "three"]} />\n\n'
        f"{body}\n\n"
        '<FAQ items={[{ q: "a", a: "b" }, { q: "c", a: "d" }, { q: "e", a: "f" }]} />\n\n'
        "<BlogCTA />\n"
    )


def test_price_in_body_is_a_validation_error():
    report = validate_mdx(_shell("## A section\n\nThe rebuild came to $45,000 in the end."))
    assert any("editorial policy" in e for e in report.errors)


def test_split_value_and_unit_in_a_component_is_an_error():
    """The reader sees "12-24 weeks"; the source never contains that string."""
    report = validate_mdx(_shell(
        '## A section\n\n<StatGrid caption="c" stats={[{ label: "L", value: "12-24", unit: "weeks" }]} />'))
    assert any("editorial policy" in e for e in report.errors)


def test_cost_heading_is_a_warning_not_an_error():
    """By the time a body exists the subject was settled at topic time.

    Failing the whole draft over a heading would loop the writer without converging —
    it cannot change the subject from inside a section prompt. topic_violations() is
    what stops a cost post being commissioned at all.
    """
    report = validate_mdx(_shell("## What it costs to run\n\nOrdinary prose here."))
    assert any("editorial policy" in w for w in report.warnings)
    assert not any("editorial policy" in e for e in report.errors)


def test_clean_body_has_no_policy_errors():
    report = validate_mdx(_shell(
        "## How the routing layer decides\n\n"
        "It reads the workflow, not the org chart. That is the whole trick."))
    assert not any("editorial policy" in e for e in report.errors)


# ── Deterministic repair ─────────────────────────────────────────────────────
# A wrong prop name is a rename with no judgement in it, so it is fixed mechanically
# rather than by asking the model to rewrite the post around it.

def test_rename_fixes_a_wrong_container_prop():
    from seo.mdx_repair import rename_component_props
    fixed, notes = rename_component_props(
        '<StatGrid caption="x" data={[{ label: "A", value: "Daily" }]} />')
    assert 'stats={' in fixed and 'data={' not in fixed
    assert notes
    assert component_prop_errors(fixed) == []


def test_rename_fixes_wrong_entry_keys():
    from seo.mdx_repair import rename_component_props
    fixed, _ = rename_component_props(
        '<Timeline caption="x" milestones={[{ date: "A", title: "B" }, { date: "C", title: "D" }]} />')
    assert 'events={' in fixed
    assert 'label: "B"' in fixed and 'label: "D"' in fixed
    assert component_prop_errors(fixed) == []


def test_rename_leaves_correct_components_untouched():
    from seo.mdx_repair import rename_component_props
    good = '<StatGrid caption="x" stats={[{ label: "A", value: "Daily" }]} />'
    fixed, notes = rename_component_props(good)
    assert fixed == good and notes == []


def test_rename_refuses_when_both_props_are_present():
    """Renaming blindly would produce two `stats=` props and invalid MDX."""
    from seo.mdx_repair import rename_component_props
    both = '<StatGrid caption="x" stats={[{ value: "1" }]} data={[{ value: "2" }]} />'
    fixed, notes = rename_component_props(both)
    assert fixed == both and notes == []


def test_repair_is_idempotent():
    """The pass runs repeatedly inside the repair loop; a second run must be a no-op."""
    from seo.mdx_repair import rename_component_props
    once, _ = rename_component_props(
        '<FlowDiagram caption="x" stages={[{ title: "A" }]} />')
    twice, notes = rename_component_props(once)
    assert once == twice and notes == []
