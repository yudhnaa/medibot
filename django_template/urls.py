from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path, re_path

from rest_framework import permissions

from drf_yasg import openapi
from drf_yasg.views import get_schema_view

SchemaView = get_schema_view(
    openapi.Info(
        title="Snippets API",
        default_version="v1",
        description="Test description",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@snippets.local"),
        license=openapi.License(name="python-django-api-template License"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

# urls
urlpatterns = [
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
        r"^redoc/$", SchemaView.with_ui("redoc", cache_timeout=0), name="schema-redoc"
    ),
    path("api/v1/auth/", include("authentication.urls")),
    path("api/v1/chatbot/", include("chatbot.urls")),
    path("api/v1/vector-store/", include("vector_store.urls")),
    path("api/v1/vision/", include("vision.urls")),
    path("admin/", admin.site.urls),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

if settings.DEBUG:
    try:
        import debug_toolbar
    except ImportError:
        pass
    else:
        urlpatterns += [
            path("__debug__/", include(debug_toolbar.urls)),
        ]

admin.site.site_header = "python-django-api-template"
admin.site.index_title = "python-django-api-template"
