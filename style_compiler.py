"""Deterministic front-end for portable Skill style knowledge.

The catalogue remains inside the Skill package.  PromptHub only recognises when
the compiler is relevant, builds a provenance-aware canonical profile, and
hands that profile to the Skill's normal model adapter.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


RESOURCE_PATH = "references/obscure-style-compiler.md"
PROFILE_FIELDS = (
    "medium_and_substrate",
    "image_formation_or_mark_making",
    "palette_and_tonal_structure",
    "optical_properties",
    "surface_and_material_response",
    "ageing_and_process_artefacts",
    "composition_and_conventions",
)
COMMON_STYLES = {"cinematic", "photorealism", "watercolour", "oil painting", "anime", "pixel art"}
PARTIAL_STYLES = {"blueprint", "retro", "vintage", "folk art", "magic realism", "pictorialism"}
STYLE_INTENT = re.compile(r"\b(style|aesthetic|process|print|printmaking|photographic|photography|visual language)\b", re.I)
INVENTED_STYLE = re.compile(r"\b(?:invented|hybrid)\s+(?:style|aesthetic)\s*[:=]?\s*[\"']?([^\n\"']{2,120})", re.I)


class StyleCompilerError(ValueError):
    pass


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _aliases(name: str) -> list[str]:
    values = {name.strip()}
    parenthetical = re.findall(r"\(([^)]+)\)", name)
    values.update(parenthetical)
    values.update(part.strip() for part in re.split(r"/|\bor\b", re.sub(r"\([^)]*\)", "", name), flags=re.I))
    return sorted({_normalise(value) for value in values if _normalise(value)}, key=len, reverse=True)


def load_style_signatures(skill_root: str | Path) -> list[dict[str, Any]]:
    path = Path(skill_root) / RESOURCE_PATH
    if not path.is_file():
        return []
    category = "uncategorised"
    in_library = False
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().lower() == "## curated signature library":
            in_library = True
            continue
        if not in_library:
            continue
        if line.startswith("### "):
            category = line[4:].strip()
            continue
        match = re.match(r"^- \*\*(.+?)\*\*:\s*(.+)$", line.strip())
        if not match:
            continue
        name, signature = match.groups()
        entries.append({
            "canonical_name": re.sub(r"\s*\([^)]*\)\s*$", "", name).strip(),
            "aliases": _aliases(name),
            "category": category,
            "signature": signature.strip(),
            "confidence": "uncertain" if re.search(r"\buncertain|moderate confidence|disputed\b", signature, re.I) else "high",
        })
    return entries


def _attribute_profile(signature: str) -> dict[str, str | None]:
    groups = {
        "medium_and_substrate": r"paper|plate|glass|metal|film|print|textile|wall|raster|envelope|panel|gelatin|emulsion",
        "image_formation_or_mark_making": r"exposure|intaglio|relief|cut|stencil|pigment|bleach|developer|discharge|refraction|scanline|collage|letter|etched|carved|drawn|layer",
        "palette_and_tonal_structure": r"palette|colour|color|tone|tonal|black|white|sepia|cyan|blue|saturated|contrast|highlight|shadow",
        "optical_properties": r"focus|lens|grain|halation|irides|transparent|glow|refract|blur|reflection|luminous|pointill",
        "surface_and_material_response": r"surface|texture|tooth|sheen|relief|ink|fibre|scratch|crack|emboss|gloss|gouge",
        "ageing_and_process_artefacts": r"fade|fog|stain|leak|registration|reticulation|dust|uneven|torn|veil|artefact|artifact|offset",
        "composition_and_conventions": r"composition|framing|viewpoint|negative space|motif|typograph|annotation|portrait|poster|map|hierarchy|layout|space",
    }
    result: dict[str, str | None] = {}
    clauses = [part.strip() for part in re.split(r"(?<=[.;])\s+|,\s+(?=[a-z])", signature) if part.strip()]
    for field, pattern in groups.items():
        matches = [clause for clause in clauses if re.search(pattern, clause, re.I)]
        result[field] = "; ".join(matches[:3]) or None
    return result


def _explicit_style(parameters: dict[str, Any], request: str) -> tuple[str | None, bool]:
    for key in ("style", "visual_style", "aesthetic"):
        value = str(parameters.get(key) or "").strip()
        if value:
            return value, True
    match = INVENTED_STYLE.search(request)
    return (match.group(1).strip(" .,:;"), True) if match else (None, False)


def compile_style_request(skill_root: str | Path, request: str, parameters: dict[str, Any] | None = None) -> dict[str, Any] | None:
    parameters = dict(parameters or {})
    entries = load_style_signatures(skill_root)
    if not entries:
        return None
    request_normalised = _normalise(request)
    explicit, explicit_intent = _explicit_style(parameters, request)
    explicit_normalised = _normalise(explicit or "")
    candidates = []
    for entry in entries:
        matching_aliases = [alias for alias in entry["aliases"] if alias and (alias in request_normalised or alias == explicit_normalised)]
        if matching_aliases:
            candidates.append((max(map(len, matching_aliases)), entry))
    matched = max(candidates, key=lambda item: item[0])[1] if candidates else None
    if not matched and not explicit_intent:
        return None
    style_name = explicit or (matched["canonical_name"] if matched else None)
    if not style_name or len(style_name) > 120 or any(ord(char) < 32 for char in style_name):
        raise StyleCompilerError("Style names must be printable text between 1 and 120 characters.")
    normalised_name = _normalise(style_name)
    invented = bool(INVENTED_STYLE.search(request))
    if invented or re.search(r"\bhybrid\b|\+|\bx\b", style_name, re.I):
        classification = "hybrid_or_invented"
    elif matched:
        classification = "partially_understood" if normalised_name in PARTIAL_STYLES else "obscure_or_technical"
    elif normalised_name in COMMON_STYLES:
        classification = "common"
    else:
        classification = "partially_understood"
    signature = matched["signature"] if matched else str(parameters.get("visible_traits") or "").strip()
    attributes = _attribute_profile(signature) if signature else {field: None for field in PROFILE_FIELDS}
    supplied = {
        "medium_and_substrate": parameters.get("medium") or parameters.get("substrate"),
        "image_formation_or_mark_making": parameters.get("process") or parameters.get("mark_making"),
        "composition_and_conventions": parameters.get("composition"),
    }
    for field, value in supplied.items():
        if value:
            attributes[field] = str(value)
    context = {key: parameters.get(key) for key in ("era", "region", "references", "visible_traits") if parameters.get(key)}
    return {
        "classification": classification,
        "confidence": matched["confidence"] if matched else "uncertain",
        "canonical_name": matched["canonical_name"] if matched else style_name,
        "aliases": matched["aliases"] if matched else [normalised_name],
        "category": matched["category"] if matched else "decomposed or invented",
        "observable_signature": signature or None,
        "attributes": attributes,
        "decomposition_context": context,
        "invariants": parameters.get("invariants") or ["identity", "anatomy", "pose", "composition", "framing", "required text", "object geometry", "reference-image roles"],
        "translation_guidance": [
            "Use the style term only as a secondary anchor.",
            "Express the observable profile in the selected model's native prompt language.",
            "Preserve declared invariants and apply the target model adapter after compilation.",
            "Do not add generic negative prompts or quality-token clutter.",
        ],
        "source_resource": RESOURCE_PATH,
    }
