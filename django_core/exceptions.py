"""
Custom Exception Handler and Custom Exceptions.
Provides standardized exception handling for all API endpoints.
"""

import logging
from typing import Any

from django.core.exceptions import PermissionDenied, ValidationError
from django.http import Http404

from rest_framework import status
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import exception_handler


class BaseAPIException(APIException):
    """
    Base exception class for all custom API exceptions.
    Provides a consistent structure for error responses.
    """

    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = "An error occurred"
    default_code = "error"

    def __init__(
        self,
        detail: str | None = None,
        code: str | None = None,
        errors: dict[str, Any] | list[Any] | None = None,
    ):
        if detail is None:
            detail = self.default_detail
        if code is None:
            code = self.default_code
        super().__init__(detail, code)
        self.errors = errors


class BadRequestException(BaseAPIException):
    """Exception for 400 Bad Request errors."""

    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = "Bad request"
    default_code = "bad_request"


class UnauthorizedException(BaseAPIException):
    """Exception for 401 Unauthorized errors."""

    status_code = status.HTTP_401_UNAUTHORIZED
    default_detail = "Authentication credentials were not provided"
    default_code = "unauthorized"


class ForbiddenException(BaseAPIException):
    """Exception for 403 Forbidden errors."""

    status_code = status.HTTP_403_FORBIDDEN
    default_detail = "You do not have permission to perform this action"
    default_code = "forbidden"


class NotFoundException(BaseAPIException):
    """Exception for 404 Not Found errors."""

    status_code = status.HTTP_404_NOT_FOUND
    default_detail = "Resource not found"
    default_code = "not_found"


class ConflictException(BaseAPIException):
    """Exception for 409 Conflict errors."""

    status_code = status.HTTP_409_CONFLICT
    default_detail = "Resource conflict"
    default_code = "conflict"


class UnprocessableEntityException(BaseAPIException):
    """Exception for 422 Unprocessable Entity errors."""

    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_detail = "Unable to process the request"
    default_code = "unprocessable_entity"


class TooManyRequestsException(BaseAPIException):
    """Exception for 429 Too Many Requests errors."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    default_detail = "Too many requests. Please try again later"
    default_code = "too_many_requests"


class InternalServerException(BaseAPIException):
    """Exception for 500 Internal Server errors."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = "An internal server error occurred"
    default_code = "internal_server_error"


class ServiceUnavailableException(BaseAPIException):
    """Exception for 503 Service Unavailable errors."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = "Service temporarily unavailable"
    default_code = "service_unavailable"


def custom_exception_handler(
    exc: Exception, context: dict[str, Any]
) -> Response | None:
    """
    Custom exception handler for Django REST Framework.

    Formats all exceptions into a standardized response format:
    {
        "success": false,
        "status_code": <int>,
        "message": "<error message>",
        "errors": { ... } or null
    }

    Args:
        exc: The exception that was raised
        context: Context information including request and view

    Returns:
        Response object with formatted error data
    """
    # First, let DRF handle common exceptions
    response = exception_handler(exc, context)

    # Handle Django's Http404
    if isinstance(exc, Http404):
        return _build_error_response(
            message=str(exc) if str(exc) else "Not found",
            status_code=status.HTTP_404_NOT_FOUND,
            errors=None,
        )

    # Handle Django's PermissionDenied
    if isinstance(exc, PermissionDenied):
        return _build_error_response(
            message=str(exc) if str(exc) else "Permission denied",
            status_code=status.HTTP_403_FORBIDDEN,
            errors=None,
        )

    # Handle Django's ValidationError
    if isinstance(exc, ValidationError):
        if hasattr(exc, "message_dict"):
            errors: dict[str, Any] = exc.message_dict  # type: ignore[attr-defined]
            message = "Validation error"
        elif hasattr(exc, "messages"):
            errors = {"non_field_errors": list(exc.messages)}
            message = str(exc.messages[0]) if exc.messages else "Validation error"
        else:
            errors = {"non_field_errors": [str(exc)]}
            message = str(exc)

        return _build_error_response(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            errors=errors,
        )

    # Handle our custom exceptions
    if isinstance(exc, BaseAPIException):
        return _build_error_response(
            message=str(exc.detail),
            status_code=exc.status_code,
            errors=exc.errors,
        )

    # Handle DRF exceptions that were already processed
    if response is not None:
        extracted_errors = _extract_errors(response.data)
        message = _extract_message(response.data, response.status_code)

        return _build_error_response(
            message=message,
            status_code=response.status_code,
            errors=extracted_errors,
        )

    # For unhandled exceptions, return a generic 500 error
    # Note: In production, you may want to log the exception here

    logger = logging.getLogger(__name__)
    logger.exception(f"Unhandled exception: {exc}")
    return _build_error_response(
        message=f"Error: {exc!s}",  # TODO: Change back to generic message in production
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        errors=None,
    )


def _build_error_response(
    message: str,
    status_code: int,
    errors: dict[str, Any] | list[Any] | None,
) -> Response:
    """Build a standardized error response."""
    return Response(
        {
            "success": False,
            "status_code": status_code,
            "message": message,
            "errors": errors,
        },
        status=status_code,
    )


def _extract_message_from_dict(data: dict[str, Any]) -> str | None:
    """Extract error message from a dictionary."""
    if "detail" in data:
        detail = data["detail"]
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list) and detail:
            return str(detail[0])
    if "message" in data:
        return str(data["message"])
    if "error" in data:
        return str(data["error"])
    if "non_field_errors" in data:
        non_field_errors = data["non_field_errors"]
        if isinstance(non_field_errors, list) and non_field_errors:
            return str(non_field_errors[0])
    return None


def _extract_message_from_sequence(data: list[Any]) -> str | None:
    """Extract error message from a list."""
    return str(data[0]) if data else None


def _extract_message(data: Any, status_code: int) -> str:
    """Extract error message from response data."""
    if isinstance(data, dict):
        message = _extract_message_from_dict(data)
        if message:
            return message
    elif isinstance(data, str):
        return data
    elif isinstance(data, list):
        message = _extract_message_from_sequence(data)
        if message:
            return message

    return _get_default_message(status_code)


def _extract_errors(data: Any) -> dict[str, Any] | list[Any] | None:
    """Extract and format errors from response data."""
    if data is None:
        return None
    if isinstance(data, dict):
        # Remove 'detail' since it's used in message
        extracted = {k: v for k, v in data.items() if k != "detail"}
        return extracted if extracted else None
    if isinstance(data, list):
        return {"non_field_errors": data}
    if isinstance(data, str):
        return {"non_field_errors": [data]}
    return None


def _get_default_message(status_code: int) -> str:
    """Get default error message based on status code."""
    messages: dict[int, str] = {
        400: "Bad request",
        401: "Authentication credentials were not provided",
        403: "Permission denied",
        404: "Not found",
        405: "Method not allowed",
        409: "Conflict",
        422: "Unprocessable entity",
        429: "Too many requests",
        500: "Internal server error",
        502: "Bad gateway",
        503: "Service unavailable",
    }
    return messages.get(status_code, "An error occurred")
