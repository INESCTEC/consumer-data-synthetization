"""Pseudonymisation of pre-trained model provenance.

The models shipped with this platform are trained on real smart meters. The
folder name is the meter's own identifier, and metadata.json records where the
raw series was pulled from, the calendar span it covers and when training ran.
Serving any of that alongside generated data lets a consumer work back to the
household and the period behind it - which is precisely what synthetic data
exists to avoid. The generated series carries the same problem in its time
axis: it is labelled with the training data's own start, so the timestamps
alone date the original measurements.

So for these models nothing real is published:

  * the model is published as `ResidentialConsumer_NN`, and that alias is the
    only address it answers to - accepting the real folder name would let a
    client confirm which meters exist by guessing names. The number is a
    position in an ordering keyed by a per-deployment secret, so it says
    nothing about the meter behind it, not even where it sits on disk. Models
    offering at least one verified horizon are numbered first, so every model a
    client can actually generate from falls inside the two-digit 00-99 range
    and the rest - which a picker hides anyway - run on past it. Numbers are
    positions, so adding or removing models renumbers the others;
  * device_id, the source path, the raw-fetch block and every training or
    evaluation timestamp are dropped;
  * generated data is dated from the moment of generation instead of from the
    start of the real series (see `generation_start`).

Aggregate quality figures (days trained on, resolution, verified horizons, the
per-horizon metrics) are kept: they describe how much to trust the output and
identify nobody.

Models a user trained themselves are left alone. They are recognised by having
no device_id or raw_fetch in their metadata - their name and timestamps are the
user's own data, not somebody else's, and renaming them would break the
workflow of whoever uploaded the CSV.
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Overrides the persisted secret; set it to keep aliases identical across
# deployments that do not share the models volume.
_SECRET_ENV = "SYNTHETIC_ALIAS_SECRET"
# Lives beside the models so every replica sharing the volume derives the same
# aliases, and so they survive a container restart.
_SECRET_FILE = ".alias_secret"

_ALIAS_PREFIX = "ResidentialConsumer_"

_secret_cache: dict[Path, bytes] = {}
# The two directions of the same numbering. Rebuilt on a miss, so they self-heal
# when models are added or removed under a running service.
_alias_index: dict[str, Path] = {}
_name_by_dir: dict[Path, str] = {}


def _secret(model_root: Path) -> bytes:
    """The per-deployment HMAC key the aliases are derived from.

    A plain hash of the device id would be reversible by anyone: the
    identifiers follow a short fixed pattern, so the whole space can be
    enumerated in seconds. Keying the digest is what makes the alias opaque.
    """
    cached = _secret_cache.get(model_root)
    if cached is not None:
        return cached

    from_env = os.getenv(_SECRET_ENV)
    if from_env:
        _secret_cache[model_root] = from_env.encode()
        return _secret_cache[model_root]

    path = model_root / _SECRET_FILE
    secret = secrets.token_hex(32).encode()
    try:
        # Exclusive create: two replicas starting together must not each write
        # their own key and then disagree about every alias.
        with path.open("xb") as f:
            f.write(secret)
    except FileExistsError:
        secret = path.read_bytes().strip() or secret
    except OSError as e:
        logger.warning(
            "Could not persist the alias secret to %s (%s); aliases will change "
            "when this process restarts. Set %s to pin them.",
            path,
            e,
            _SECRET_ENV,
        )

    _secret_cache[model_root] = secret
    return secret


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open(encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def is_pseudonymous(metadata: dict) -> bool:
    """True for a model trained on somebody else's real, identified meter."""
    return bool(metadata.get("device_id") or metadata.get("raw_fetch"))


def _digest(model_root: Path, model_dir: Path) -> str:
    """A keyed digest of the real folder name.

    It is never published; it only orders the models. An unkeyed hash would
    order them reproducibly for anyone who guessed a meter identifier, which is
    enough to check the guess - the identifiers follow a short fixed pattern, so
    the whole space can be enumerated in seconds.
    """
    try:
        real_name = str(model_dir.relative_to(model_root))
    except ValueError:
        real_name = model_dir.name
    return hmac.new(_secret(model_root), real_name.encode(), hashlib.sha256).hexdigest()


def _iter_model_dirs(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for model_path in sorted(root.rglob("model.pt")):
        yield model_path.parent


def _alias_for_number(number: int) -> str:
    return f"{_ALIAS_PREFIX}{number:02d}"


def _rebuild_index(model_root: Path) -> None:
    """Number every pseudonymous model under `model_root`.

    Models with a verified generation horizon are numbered first so that they
    take the two-digit numbers: they are the only ones a client can generate
    from, and a picker hides the rest. Within each group the keyed digest
    decides the order, so a number reveals nothing but the model's own quality
    tier - which `offered_hours` states outright anyway.
    """
    ranked = []
    for model_dir in _iter_model_dirs(model_root):
        metadata = load_metadata(model_dir)
        if not is_pseudonymous(metadata):
            continue
        caps = metadata.get("capabilities") or {}
        usable = 0 if caps.get("offered_hours") else 1
        ranked.append((usable, _digest(model_root, model_dir), model_dir))
    ranked.sort()

    _alias_index.clear()
    _name_by_dir.clear()
    for number, (_, _, model_dir) in enumerate(ranked):
        alias = _alias_for_number(number)
        _alias_index[alias] = model_dir
        _name_by_dir[model_dir] = alias


def public_name(model_root: Path, model_dir: Path, metadata: dict) -> str:
    """The name a client is given for a model."""
    if not is_pseudonymous(metadata):
        return model_dir.name

    name = _name_by_dir.get(model_dir)
    if name is None:
        _rebuild_index(model_root)
        name = _name_by_dir.get(model_dir)
    if name is None:
        # A pseudonymous folder the scan did not reach, because it has no
        # model.pt of its own. Number it here rather than ever let the real
        # folder name out as a fallback.
        name = _alias_for_number(len(_alias_index))
        _alias_index[name] = model_dir
        _name_by_dir[model_dir] = name
    return name


def resolve_alias(model_root: Path, name: str) -> Optional[Path]:
    """The model folder an alias stands for, or None if it is not an alias."""
    hit = _alias_index.get(name)
    if hit is not None and (hit / "model.pt").exists():
        return hit
    _rebuild_index(model_root)
    return _alias_index.get(name)


def public_summary(model_root: Path, model_dir: Path, metadata: dict) -> dict:
    """Identity plus the capability figures a model picker needs.

    Callers merge this into their own listing entry, so it is the single place
    that decides which metadata fields reach a client.
    """
    caps = metadata.get("capabilities") or {}
    pseudonymous = is_pseudonymous(metadata)
    name = public_name(model_root, model_dir, metadata)
    return {
        "name": name,
        # The model stands for exactly one meter, so the pseudonym names both.
        "device_id": name if pseudonymous else metadata.get("device_id"),
        "parameter": metadata.get("parameter"),
        "resolution_minutes": metadata.get("resolution_minutes"),
        # Withheld for a real meter: it dates the underlying measurements.
        "trained_at": None if pseudonymous else metadata.get("trained_at"),
        "trained_sequence_hours": caps.get("trained_sequence_hours"),
        "real_days_trained_on": caps.get("real_days_trained_on"),
        "offered_hours": caps.get("offered_hours"),
        "max_offered_hours": caps.get("max_offered_hours"),
        "capabilities_evaluated_at": None
        if pseudonymous
        else caps.get("evaluated_at"),
    }


# Whole blocks and single fields that place the real data in time or space.
_PROVENANCE_KEYS = ("source", "raw_fetch", "trained_at", "device_id", "model_name")


def public_metadata(model_root: Path, model_dir: Path, metadata: dict) -> dict:
    """metadata.json with everything identifying removed."""
    if not is_pseudonymous(metadata):
        return metadata

    name = public_name(model_root, model_dir, metadata)
    public = {k: v for k, v in metadata.items() if k not in _PROVENANCE_KEYS}
    public["model_name"] = name
    public["device_id"] = name

    caps = metadata.get("capabilities")
    if isinstance(caps, dict):
        public["capabilities"] = {
            k: v for k, v in caps.items() if k != "evaluated_at"
        }
    return public


def generation_start(metadata: dict) -> Optional[datetime]:
    """Where a generated series' time axis should begin, or None to keep the
    model's own.

    Generation is dated from now, so the output says when it was made and
    nothing about when the training data was recorded. It is floored to the
    model's resolution to keep the familiar clean grid (:00, :15, :30, :45).
    """
    if not is_pseudonymous(metadata):
        return None

    now = datetime.now(timezone.utc).replace(microsecond=0)
    resolution = metadata.get("resolution_minutes")
    try:
        step = int(resolution)
    except (TypeError, ValueError):
        return now
    if step <= 0:
        return now

    midnight = now.replace(hour=0, minute=0, second=0)
    elapsed = int((now - midnight).total_seconds() // 60)
    return midnight + timedelta(minutes=elapsed - elapsed % step)
