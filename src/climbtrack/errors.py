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
