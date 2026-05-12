from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path, re_path

from rest_framework import permissions

from drf_yasg import openapi
from drf_yasg.views import get_schema_view


class ApiDocsPermission(permissions.BasePermission):
    def has_permission(self, request, view):
        if settings.DEBUG or settings.PUBLIC_API_DOCS:
            return True
        return bool(request.user and request.user.is_staff)


SchemaView = get_schema_view(
    openapi.Info(
        title="MediBot Backend API",
        default_version="v1",
        description=(
            "Generated API schema for the MediBot Django backend.\n\n"
            "Current public REST surface includes authentication, chatbot, "
            "vector-store, and vision endpoints under `/api/v1/`.\n\n"
            "Important runtime notes:\n"
            "- Most REST responses are wrapped by `CustomJSONRenderer`.\n"
            "- Chat streaming is served as Server-Sent Events at "
            "`/api/v1/chatbot/chat/` when `stream=true`.\n"
            "- Offline benchmark workflows live in Django admin / management "
            "commands and are not exposed as public REST endpoints."
        ),
        contact=openapi.Contact(email="backend@medibot.local"),
        license=openapi.License(name="MediBot Project License"),
    ),
    public=True,
    permission_classes=[ApiDocsPermission],
)

# urls
urlpatterns = (
    [
        re_path(
            r"^swagger(?P<format>\.json|\.yaml)$",
            SchemaView.without_ui(cache_timeout=0),
            name="schema-json",
        ),
        re_path(
            r"^swagger/$",
            SchemaView.with_ui("swagger", cache_timeout=0),
            name="schema-swagger-ui",
        ),
        re_path(
            r"^redoc/$",
            SchemaView.with_ui("redoc", cache_timeout=0),
            name="schema-redoc",
        ),
        path("api/v1/auth/", include("authentication.urls")),
        path("api/v1/chatbot/", include("chatbot.urls")),
        path("api/v1/vector-store/", include("vector_store.urls")),
        path("api/v1/vision/", include("vision.urls")),
        path("admin/", admin.site.urls),
    ]
    + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
)

if settings.DEBUG:
    try:
        import debug_toolbar
    except ImportError:
        pass
    else:
        urlpatterns += [
            path("__debug__/", include(debug_toolbar.urls)),
        ]

admin.site.site_header = "MediBot Admin"
admin.site.index_title = "MediBot Operations"
