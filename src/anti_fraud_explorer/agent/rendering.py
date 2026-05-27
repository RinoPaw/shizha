"""Jinja2 template rendering helpers."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "templates"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
)


def render_template(name: str, **kwargs) -> str:
    return _JINJA_ENV.get_template(name).render(**kwargs)


def build_transform_local(transform_type: str, target_item) -> str:
    """Build a template-based local answer for content transformation."""
    from ..service.item_cards import title_with_family

    title = title_with_family(target_item)
    category = target_item.ccl2023_category
    summary = target_item.summary
    features = "；".join(target_item.key_methods) or summary
    level = target_item.risk_level or ""

    return render_template(
        "transform_local.md.j2",
        transform_type=transform_type,
        title=title,
        category=category,
        summary=summary,
        features=features,
        level=level,
    ).strip()
