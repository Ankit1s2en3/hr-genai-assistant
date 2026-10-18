"""JWT authentication. Tokens are issued by the compensation app; we only verify them."""

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings

# auto_error=False so we control the response: always 401 (not 403) when the token is missing.
bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_role(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """Verify the bearer token and return the user's role."""
    if credentials is None:
        raise _unauthorized("Missing bearer token")

    settings = get_settings()
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=["HS256"],  # pin the algorithm; never trust the token header
            options={"require": ["exp"]},
        )
    except jwt.InvalidTokenError:
        raise _unauthorized("Invalid or expired token")

    claim = payload.get(settings.jwt_role_claim)
    # The claim can be a single role ("HR") or a list of groups (["HR", "Managers"]).
    if isinstance(claim, list):
        return settings.hr_role_value if settings.hr_role_value in claim else "employee"
    return str(claim or "")


def require_hr(role: str = Depends(get_current_role)) -> str:
    if role != get_settings().hr_role_value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="HR role required")
    return role
