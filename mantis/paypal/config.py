"""PayPal gate configuration: one switch, two modes, one interface.

    python -m mantis.paypal.config        # print the resolved mode, never a secret

``MANTIS_PAYPAL_MODE``
    ``mock`` (the default) replays recorded PayPal responses from
    ``data/fixtures/paypal/`` and needs no keys, no network and no ``.env`` — so a
    clean clone runs the gate demo (HARD RULE 4). ``live`` talks to the PayPal
    **sandbox** with the client id and secret in ``.env``.

"live" means sandbox, and only sandbox
----------------------------------------
:data:`SANDBOX_BASE_URL` is a constant, not a setting. There is deliberately no
environment variable that points this code at ``api-m.paypal.com``: a hackathon
demo that can move real money because somebody exported the wrong variable is a
liability, and nothing here needs production. Graduating to production is a
one-line, reviewed change to this file.

Why there is a ``.env`` reader here and no ``python-dotenv``
--------------------------------------------------------------
Three ``KEY=value`` lines do not justify a dependency, and the deployed image's
``serve`` extra is kept small on purpose (see ``pyproject.toml``). Real
environment variables always win over the file, so a CI secret or a shell export
overrides a stale ``.env`` rather than being silently shadowed by it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from mantis.core.paths import DATA_DIR, REPO_ROOT

__all__ = [
    "DEFAULT_MANDATE_KEY",
    "ENV_FILE",
    "FIXTURE_DIR",
    "MODEL_DIR",
    "RECEIPT_DIR",
    "SANDBOX_BASE_URL",
    "ConfigError",
    "Mode",
    "PayPalConfig",
    "load_config",
    "read_env_file",
]

Mode = Literal["mock", "live"]

#: The only PayPal host this package will talk to. See the module docstring.
SANDBOX_BASE_URL: Final[str] = "https://api-m.sandbox.paypal.com"

ENV_FILE: Final[Path] = REPO_ROOT / ".env"

#: Recorded sandbox responses. COMMITTED: they are what mock mode replays.
FIXTURE_DIR: Final[Path] = DATA_DIR / "fixtures" / "paypal"

#: Canonical purchase receipts, one JSON per order. Gitignored, per run.
RECEIPT_DIR: Final[Path] = DATA_DIR / "receipts"

#: The live-scoring L1 artefact, built by ``scripts/train_live_model.py``.
#: Gitignored; the gate runs L0 + L3 without it and says so.
MODEL_DIR: Final[Path] = DATA_DIR / "models"

#: HMAC key for mandate signatures when ``MANTIS_MANDATE_KEY`` is unset. It is a
#: demo key and is named like one: the signature proves the mandate the gate
#: scores is byte-identical to the one the human approved, which is the property
#: the demo needs. A deployment supplies its own key.
DEFAULT_MANDATE_KEY: Final[str] = "mantis-demo-mandate-key-not-a-secret"

#: A Visa number PayPal's sandbox accepts for card payment sources. Overridable,
#: because sandbox test cards are per account and occasionally rotated.
DEFAULT_TEST_CARD: Final[str] = "4111111111111111"


class ConfigError(RuntimeError):
    """The requested mode cannot run with what the environment provides."""


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Missing file is not an error; values are not logged.

    Blank lines and ``#`` comments are skipped, surrounding quotes stripped, and a
    stray ``\\r`` from a Windows editor removed — the last one matters, because a
    trailing carriage return in a client secret is a 401 that looks like a wrong
    password.
    """
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().rstrip("\r")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


@dataclass(frozen=True, slots=True)
class PayPalConfig:
    """Resolved settings. ``client_secret`` is excluded from ``repr``."""

    mode: Mode
    client_id: str
    client_secret: str
    base_url: str = SANDBOX_BASE_URL
    fixture_dir: Path = FIXTURE_DIR
    receipt_dir: Path = RECEIPT_DIR
    model_dir: Path = MODEL_DIR
    mandate_key: str = DEFAULT_MANDATE_KEY
    test_card: str = DEFAULT_TEST_CARD

    def __repr__(self) -> str:
        shown = f"{self.client_id[:6]}…" if self.client_id else "<unset>"
        return f"PayPalConfig(mode={self.mode!r}, client_id={shown}, base_url={self.base_url!r})"

    @property
    def has_credentials(self) -> bool:
        return bool(self.client_id and self.client_secret)


def load_config(mode: str | None = None, *, env_file: Path = ENV_FILE) -> PayPalConfig:
    """Resolve the configuration. Explicit argument > environment > ``.env`` > mock.

    Raises:
        ConfigError: ``live`` was asked for without both credentials, or the mode
            is not one of ``mock`` / ``live``.
    """
    file_values = read_env_file(env_file)

    def get(key: str, default: str = "") -> str:
        return os.environ.get(key) or file_values.get(key) or default

    chosen = (mode or get("MANTIS_PAYPAL_MODE", "mock")).strip().lower()
    # "sandbox" is accepted as a spelling of "live": live *is* the sandbox here,
    # and a user who writes the environment's real name should not get an error.
    chosen = "live" if chosen == "sandbox" else chosen
    if chosen not in ("mock", "live"):
        raise ConfigError(
            f"MANTIS_PAYPAL_MODE must be 'mock' or 'live' (alias 'sandbox'), got {chosen!r}"
        )

    config = PayPalConfig(
        mode=chosen,  # type: ignore[arg-type]  # narrowed by the check above
        client_id=get("PAYPAL_CLIENT_ID"),
        client_secret=get("PAYPAL_CLIENT_SECRET"),
        mandate_key=get("MANTIS_MANDATE_KEY", DEFAULT_MANDATE_KEY),
        test_card=get("MANTIS_PAYPAL_TEST_CARD", DEFAULT_TEST_CARD),
    )
    if config.mode == "live" and not config.has_credentials:
        raise ConfigError(
            "MANTIS_PAYPAL_MODE=live needs PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET "
            "(sandbox app credentials) in the environment or in .env. "
            "Use MANTIS_PAYPAL_MODE=mock to run with no keys."
        )
    return config


def main() -> None:
    """Print the resolved configuration, never a secret."""
    config = load_config()
    print("MANTIS PayPal gate configuration")
    print(f"  mode         : {config.mode}")
    print(f"  base url     : {config.base_url}  (sandbox only, by construction)")
    print(f"  credentials  : {'set' if config.has_credentials else 'not set'}")
    print(f"  fixtures     : {config.fixture_dir}")
    print(f"  receipts     : {config.receipt_dir}")
    print(f"  live model   : {config.model_dir}")
    is_default = config.mandate_key == DEFAULT_MANDATE_KEY
    print(f"  mandate key  : {'default demo key' if is_default else 'custom'}")


if __name__ == "__main__":
    main()
