"""
Custom API Renderers.
Standardizes all API responses with a consistent format.
"""

from typing import Any
from typing_extensions import override

from rest_framework.renderers import JSONRenderer


class CustomJSONRenderer(JSONRenderer):
    """
    Custom JSON Renderer that wraps all API responses in a standardized format.

    Success Response Format:
    {
        "success": true,
        "status_code": 200,
        "message": "Success",
        "data": { ... }
    }

    Error Response Format:
    {
        "success": false,
        "status_code": 400,
        "message": "Error message",
        "errors": { ... }
    }
    """

    @override
    def render(
        self,
        data: Any,
        accepted_media_type: str | None = None,
        renderer_context: dict[str, Any] | None = None,
    ) -> bytes:
        """Render data into JSON with standardized response wrapper."""
        if renderer_context is None:
            renderer_context = {}

        response = renderer_context.get("response")
        status_code = response.status_code if response else 200

        # Check if response is already formatted (from exception handler)
        if isinstance(data, dict) and "success" in data and "status_code" in data:
            return super().render(data, accepted_media_type, renderer_context)

        # Determine if this is an error response
        is_error = status_code >= 400

        if is_error:
            wrapped_data = {
                "success": False,
                "status_code": status_code,
                "message": self._extract_error_message(data, status_code),
                "errors": self._format_errors(data),
            }
        else:
            custom_message = None
            response_data = data
            if isinstance(data, dict) and "message" in data:
                custom_message = data.pop("message", None)
                response_data = data if data else None

            wrapped_data = {
                "success": True,
                "status_code": status_code,
                "message": custom_message or self._get_success_message(status_code),
                "data": response_data,
            }

        return super().render(wrapped_data, accepted_media_type, renderer_context)

    def _extract_error_from_dict(self, data: dict[str, Any]) -> str | None:
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
            errors = data["non_field_errors"]
            if isinstance(errors, list) and errors:
                return str(errors[0])
        return None

    def _extract_error_message(self, data: Any, status_code: int) -> str:
        """Extract a human-readable error message from the data."""
        if isinstance(data, dict):
            message = self._extract_error_from_dict(data)
            if message:
                return message
        elif isinstance(data, str):
            return data
        elif isinstance(data, list) and data:
            return str(data[0])

        return self._get_default_error_message(status_code)

    def _get_default_error_message(self, status_code: int) -> str:
        """Get default error message based on status code."""
        messages: dict[int, str] = {
            400: "Bad Request",
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

    def _format_errors(self, data: Any) -> dict[str, Any] | list[Any] | None:
        """Format errors for consistent output."""
        if data is None:
            return None
        if isinstance(data, dict):
            # Remove 'detail' if it exists since we've already used it for message
            errors = {k: v for k, v in data.items() if k != "detail"}
            return errors if errors else None
        if isinstance(data, list):
            return {"non_field_errors": data}
        if isinstance(data, str):
            return {"non_field_errors": [data]}
        return None

    def _get_success_message(self, status_code: int) -> str:
        """Get success message based on status code."""
        messages: dict[int, str] = {
            200: "Success",
            201: "Created successfully",
            202: "Accepted",
            204: "Deleted successfully",
        }
        return messages.get(status_code, "Success")
