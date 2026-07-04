"""JSON persistence for named profiles (PRD §8).

One file per profile, named by a slug of its ``name``. Writes are atomic
(write to a temp file in the same directory, then ``os.replace``) so a crash
mid-write never leaves a corrupt profile file behind. Presets
(``funnel.profiles.models.PRESETS``) are always included in ``list_profiles``
and are never written to disk unless the user explicitly saves them (at
which point they are saved like any other profile, under their own name);
``delete_profile`` refuses to delete a preset name regardless of whether a
same-named file exists on disk.
"""

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import cast

from funnel.profiles.models import PRESETS, Profile, SliderValues

PROFILES_DIR_ENV_VAR = "FUNNEL_PROFILES_DIR"

_PRESET_NAMES = frozenset(preset.name for preset in PRESETS)


def profiles_dir() -> Path:
    """Resolve the on-disk profiles directory.

    Honors the ``FUNNEL_PROFILES_DIR`` environment variable if set;
    otherwise falls back to ``<repo root>/data/profiles`` (relative to this
    file's location in ``engine/src/funnel/profiles/store.py``, mirroring
    ``funnel.api.app._resolve_web_dir``'s ``parents[4]`` convention).
    """
    override = os.environ.get(PROFILES_DIR_ENV_VAR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "data" / "profiles"


def _slugify(name: str) -> str:
    """Lowercase, non-alphanumeric-to-hyphen slug used as a profile's filename stem."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "profile"


def _profile_path(name: str, directory: Path) -> Path:
    return directory / f"{_slugify(name)}.json"


def _to_json_dict(profile: Profile) -> dict[str, object]:
    return asdict(profile)


def _int_field(value: object) -> int:
    """Narrow a raw JSON-decoded value to ``int``."""
    assert isinstance(value, int)
    return value


def _from_json_dict(data: dict[str, object]) -> Profile:
    raw_sliders = data["sliders"]
    assert isinstance(raw_sliders, dict)
    sliders_data = cast("dict[str, object]", raw_sliders)
    sliders = SliderValues(
        capital=_int_field(sliders_data["capital"]),
        risk_tolerance=_int_field(sliders_data["risk_tolerance"]),
        time_horizon=_int_field(sliders_data["time_horizon"]),
        drawdown_tolerance=_int_field(sliders_data["drawdown_tolerance"]),
    )
    return Profile(
        name=str(data["name"]),
        sliders=sliders,
        created_at=str(data["created_at"]),
        preset=bool(data.get("preset", False)),
    )


def save_profile(profile: Profile, directory: Path | None = None) -> Path:
    """Atomically write ``profile`` to ``<directory>/<slug>.json``.

    ``directory`` defaults to ``profiles_dir()``. Writes to a temp file in
    the same directory first, then renames it into place, so a partially
    written file can never be observed by a concurrent reader.
    """
    target_dir = directory if directory is not None else profiles_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = _profile_path(profile.name, target_dir)

    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(_to_json_dict(profile), indent=2))
    os.replace(tmp_path, path)
    return path


def load_profile(name: str, directory: Path | None = None) -> Profile:
    """Load a saved profile by name, checking presets first.

    Raises ``FileNotFoundError`` if ``name`` is neither a preset nor a saved
    file in ``directory``.
    """
    for preset in PRESETS:
        if preset.name == name:
            return preset

    target_dir = directory if directory is not None else profiles_dir()
    path = _profile_path(name, target_dir)
    if not path.exists():
        raise FileNotFoundError(f"no saved profile named {name!r} in {target_dir}")
    return _from_json_dict(json.loads(path.read_text()))


def list_profiles(directory: Path | None = None) -> list[Profile]:
    """List every profile: presets first, then saved profiles from ``directory``.

    Saved profiles are read only from JSON files on disk; a saved file with
    the same name as a preset would sit alongside its own listing entry
    (this happens only if a user explicitly saved-as a preset's name, which
    ``save_profile`` allows).
    """
    profiles = list(PRESETS)

    target_dir = directory if directory is not None else profiles_dir()
    if not target_dir.exists():
        return profiles

    for path in sorted(target_dir.glob("*.json")):
        profiles.append(_from_json_dict(json.loads(path.read_text())))

    return profiles


def delete_profile(name: str, directory: Path | None = None) -> None:
    """Delete a saved profile by name.

    Refuses (raises ``ValueError``) if ``name`` is one of the shipped
    presets — presets cannot be deleted regardless of whether a same-named
    file also exists on disk.
    """
    if name in _PRESET_NAMES:
        raise ValueError(f"cannot delete preset profile {name!r}")

    target_dir = directory if directory is not None else profiles_dir()
    path = _profile_path(name, target_dir)
    if not path.exists():
        raise FileNotFoundError(f"no saved profile named {name!r} in {target_dir}")
    path.unlink()
