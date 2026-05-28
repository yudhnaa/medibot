from typing import Protocol, cast

from django.conf import settings
from django.test import TestCase, override_settings
from django.urls import reverse

from rest_framework.test import APIClient

from authentication.models import Customer

DOCS_TEST_MIDDLEWARE = [
    middleware
    for middleware in settings.MIDDLEWARE
    if middleware != "debug_toolbar.middleware.DebugToolbarMiddleware"
]


class ResponseWithStatus(Protocol):
    status_code: int


class ApiDocsPermissionTests(TestCase):
    api_client: APIClient

    def setUp(self) -> None:
        self.api_client = APIClient()
        self.user = Customer.objects.create_user(
            username="regular", email="regular@example.com", password="password"
        )
        self.staff = Customer.objects.create_user(
            username="staff",
            email="staff@example.com",
            password="password",
            is_staff=True,
        )
        self.urls = ("/swagger.json", "/swagger/", "/redoc/")

    def assert_docs_available(self) -> None:
        for url in self.urls:
            with self.subTest(url=url):
                response = cast(ResponseWithStatus, self.api_client.get(url))
                self.assertEqual(response.status_code, 200)

    @override_settings(
        DEBUG=True,
        PUBLIC_API_DOCS=False,
        MIDDLEWARE=DOCS_TEST_MIDDLEWARE,
    )
    def test_docs_are_public_in_debug_mode(self) -> None:
        self.assert_docs_available()

    @override_settings(DEBUG=False, PUBLIC_API_DOCS=False)
    def test_docs_require_staff_when_not_public(self) -> None:
        for url in self.urls:
            with self.subTest(url=url):
                response = cast(ResponseWithStatus, self.api_client.get(url))
                self.assertEqual(response.status_code, 401)

    @override_settings(DEBUG=False, PUBLIC_API_DOCS=False)
    def test_docs_deny_non_staff_when_not_public(self) -> None:
        self.api_client.force_authenticate(user=self.user)

        for url in self.urls:
            with self.subTest(url=url):
                response = cast(ResponseWithStatus, self.api_client.get(url))
                self.assertEqual(response.status_code, 403)

    @override_settings(DEBUG=False, PUBLIC_API_DOCS=False)
    def test_docs_allow_staff_when_not_public(self) -> None:
        self.api_client.force_authenticate(user=self.staff)
        self.assert_docs_available()

    @override_settings(DEBUG=False, PUBLIC_API_DOCS=True)
    def test_docs_can_be_public_by_explicit_opt_in(self) -> None:
        self.assert_docs_available()


class AdminDashboardTests(TestCase):
    def setUp(self) -> None:
        self.admin_user = Customer.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="password",
        )

    def test_admin_dashboard_requires_staff_login(self) -> None:
        response = self.client.get(reverse("admin:index"))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", response["Location"])

    def test_admin_dashboard_renders_metrics_and_print_report(self) -> None:
        self.client.force_login(self.admin_user)

        response = self.client.get(reverse("admin:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "MediBot Operations Dashboard")
        self.assertContains(response, "Print report")
        self.assertContains(response, "Knowledge base report")
        self.assertContains(response, "Operations report")
        self.assertContains(response, "Diseases")
        self.assertIn("summary_cards", response.context)
        self.assertIn("knowledge_report", response.context)
        self.assertIn("operations_report", response.context)
        self.assertEqual(response.context["summary_cards"][0]["label"], "Customers")
        self.assertEqual(response.context["summary_cards"][4]["label"], "Diseases")
        self.assertEqual(response.context["summary_cards"][4]["value"], 0)
        self.assertEqual(response.context["knowledge_report"]["embedding_coverage"], 0)
