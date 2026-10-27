"""iALS with a recency-weighted rating-quality preference target."""

from models.ials import ImplicitALS


class CustomSignalALS(ImplicitALS):
    """Change the preference target while retaining the base iALS confidence."""

    def __init__(self, *, half_life_days: float = 365.0, **kwargs):
        super().__init__(
            signal="quality_recency", half_life_days=half_life_days, **kwargs
        )
