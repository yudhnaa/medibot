from django.conf import settings
from django.test import TestCase

from faker import Faker

from authentication.models import Customer

fake = Faker()


class TestCalls(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.password = fake.password()
        cls.username = fake.profile()["username"]
        cls.email = fake.email()
        cls.customer = Customer.objects.create_user(
            username=cls.username,
            password=cls.password,
            email=cls.email,
        )

    def test_call_register(self):
        user_name = fake.profile()["username"]
        data = {
            "username": user_name,
            "password": fake.password(),
            "email": fake.email(),
        }

        response = self.client.post(
            "/api/v1/auth/register/", data, content_type="application/json"
        )

        self.assertEqual(response.status_code, 201)
        data = response.json().get("data", {})
        self.assertIn("username", data)
        self.assertIn("email", data)
        self.assertEqual(data.get("username"), user_name)

    def test_call_login(self):
        data = {"username": TestCalls.username, "password": TestCalls.password}

        response = self.client.post(
            "/api/v1/auth/login/", data, content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        response_data = response.json().get("data", {})
        self.assertIn("access", response_data)
        self.assertIn("refresh", response_data)

        access_cookie_name = settings.AUTH_SESSION["ACCESS_COOKIE_NAME"]
        refresh_cookie_name = settings.AUTH_SESSION["REFRESH_COOKIE_NAME"]
        self.assertIn(access_cookie_name, response.cookies)
        self.assertIn(refresh_cookie_name, response.cookies)

        access_cookie = response.cookies[access_cookie_name]
        refresh_cookie = response.cookies[refresh_cookie_name]
        self.assertEqual(
            access_cookie["path"], settings.AUTH_SESSION["ACCESS_COOKIE_PATH"]
        )
        self.assertEqual(
            refresh_cookie["path"], settings.AUTH_SESSION["REFRESH_COOKIE_PATH"]
        )
        self.assertEqual(
            bool(access_cookie["secure"]), settings.AUTH_SESSION["COOKIE_SECURE"]
        )
        self.assertEqual(
            bool(refresh_cookie["secure"]), settings.AUTH_SESSION["COOKIE_SECURE"]
        )

    def test_call_refresh_token(self):
        login_data = {"username": TestCalls.username, "password": TestCalls.password}
        login_response = self.client.post(
            "/api/v1/auth/login/", login_data, content_type="application/json"
        )
        refresh_token = login_response.json().get("data", {}).get("refresh")
        data = {"refresh": refresh_token}

        response = self.client.post(
            "/api/v1/auth/token/refresh/", data, content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        response_data = response.json().get("data", {})
        self.assertIn("access", response_data)

    def test_call_refresh_token_with_cookie_only(self):
        login_data = {"username": TestCalls.username, "password": TestCalls.password}
        self.client.post(
            "/api/v1/auth/login/", login_data, content_type="application/json"
        )

        response = self.client.post(
            "/api/v1/auth/token/refresh/", {}, content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        response_data = response.json().get("data", {})
        self.assertIn("access", response_data)

    def test_call_me_without_access_cookie_returns_401(self):
        response = self.client.get("/api/v1/auth/me/")

        self.assertEqual(response.status_code, 401)
        response_data = response.json()
        self.assertFalse(response_data.get("success"))
        self.assertEqual(response_data.get("status_code"), 401)

    def test_call_me_with_access_cookie(self):
        login_data = {"username": TestCalls.username, "password": TestCalls.password}
        self.client.post(
            "/api/v1/auth/login/", login_data, content_type="application/json"
        )

        response = self.client.get("/api/v1/auth/me/")

        self.assertEqual(response.status_code, 200)
        response_data = response.json().get("data", {})
        self.assertEqual(response_data.get("username"), TestCalls.username)

    def test_call_logout_clears_cookies_and_blocks_next_me_request(self):
        login_data = {"username": TestCalls.username, "password": TestCalls.password}
        self.client.post(
            "/api/v1/auth/login/", login_data, content_type="application/json"
        )

        logout_response = self.client.post(
            "/api/v1/auth/logout/", {}, content_type="application/json"
        )

        self.assertEqual(logout_response.status_code, 200)
        access_cookie_name = settings.AUTH_SESSION["ACCESS_COOKIE_NAME"]
        refresh_cookie_name = settings.AUTH_SESSION["REFRESH_COOKIE_NAME"]
        self.assertEqual(int(logout_response.cookies[access_cookie_name]["max-age"]), 0)
        self.assertEqual(
            int(logout_response.cookies[refresh_cookie_name]["max-age"]), 0
        )

        me_response = self.client.get("/api/v1/auth/me/")
        self.assertEqual(me_response.status_code, 401)
        me_response_data = me_response.json()
        self.assertFalse(me_response_data.get("success"))
        self.assertEqual(me_response_data.get("status_code"), 401)

    def test_call_me_with_bearer_header_without_cookie_returns_401(self):
        login_data = {"username": TestCalls.username, "password": TestCalls.password}
        login_response = self.client.post(
            "/api/v1/auth/login/", login_data, content_type="application/json"
        )
        access_token = login_response.json().get("data", {}).get("access")

        bearer_client = self.client_class()
        bearer_client.defaults["HTTP_AUTHORIZATION"] = f"Bearer {access_token}"
        response = bearer_client.get("/api/v1/auth/me/")

        self.assertEqual(response.status_code, 401)
        response_data = response.json()
        self.assertFalse(response_data.get("success"))
        self.assertEqual(response_data.get("status_code"), 401)

    def test_drf_uses_cookie_authentication_class(self):
        auth_classes = settings.REST_FRAMEWORK["DEFAULT_AUTHENTICATION_CLASSES"]
        self.assertIn(
            "authentication.authentication.CookieJWTAuthentication", auth_classes
        )

    def test_drf_requires_authentication_by_default(self):
        permission_classes = settings.REST_FRAMEWORK["DEFAULT_PERMISSION_CLASSES"]
        self.assertIn("rest_framework.permissions.IsAuthenticated", permission_classes)
