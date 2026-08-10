"""Domain-specific exceptions surfaced by the CLI."""


class ClimbTrackError(RuntimeError):
    """Base class for expected, actionable application failures."""


class ConfigurationError(ClimbTrackError):
    """Raised when configuration is invalid or unsafe."""


class ExternalToolError(ClimbTrackError):
    """Raised when a required external program is missing or fails."""


class CacheIntegrityError(ClimbTrackError):
    """Raised when a supposedly complete cache entry is invalid."""


class SchemaValidationError(ClimbTrackError):
    """Raised when canonical output records violate the schema contract."""


class DeviceUnavailableError(ClimbTrackError):
    """Raised when the explicitly configured inference device is unavailable."""


class SelectionUncertainError(ClimbTrackError):
    """Raised when automatic climber selection cannot be justified."""

    def __init__(self, reason: str, candidates: list[dict[str, object]]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.candidates = candidates


class UnknownTrackError(ClimbTrackError):
    """Raised when a manually requested track does not exist."""
