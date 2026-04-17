from __future__ import annotations

import os
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase

from authentication.models import Customer


class SeedAdminUserCommandTests(TestCase):
    def test_creates_superuser_from_env(self):
        output = StringIO()

        with patch.dict(
            os.environ,
            {
                "DJANGO_SUPERUSER_USERNAME": "seed-admin",
                "DJANGO_SUPERUSER_PASSWORD": "seed-password",
                "DJANGO_SUPERUSER_EMAIL": "seed@example.com",
            },
            clear=False,
        ):
            call_command("seed_admin_user", stdout=output)

        user = Customer.objects.get(username="seed-admin")
        self.assertEqual(user.email, "seed@example.com")
        self.assertTrue(user.is_active)
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)
        self.assertTrue(user.check_password("seed-password"))
        self.assertIn("Created admin user 'seed-admin'.", output.getvalue())

    def test_updates_existing_user_to_admin(self):
        user = Customer.objects.create_user(
            username="existing-admin",
            password="old-password",
            email="old@example.com",
        )

        output = StringIO()
        with patch.dict(
            os.environ,
            {
                "DJANGO_SUPERUSER_USERNAME": "existing-admin",
                "DJANGO_SUPERUSER_PASSWORD": "new-password",
                "DJANGO_SUPERUSER_EMAIL": "new@example.com",
            },
            clear=False,
        ):
            call_command("seed_admin_user", stdout=output)

        user.refresh_from_db()
        self.assertEqual(user.email, "new@example.com")
        self.assertTrue(user.is_active)
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)
        self.assertTrue(user.check_password("new-password"))
        self.assertIn("Updated admin user 'existing-admin'", output.getvalue())
