"""
Management command to create or update a Django admin user from env vars.
Usage: python manage.py seed_admin_user
"""

from __future__ import annotations

import os
from typing import override

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Create or update a superuser from DJANGO_SUPERUSER_* env vars"

    @override
    def add_arguments(self, parser):
        parser.add_argument("--username", default=None, help="Admin username override")
        parser.add_argument("--password", default=None, help="Admin password override")
        parser.add_argument("--email", default=None, help="Admin email override")

    @override
    def handle(self, *args, **options):
        username = options["username"] or os.getenv("DJANGO_SUPERUSER_USERNAME", "")
        password = options["password"] or os.getenv("DJANGO_SUPERUSER_PASSWORD", "")
        email = options["email"] or os.getenv("DJANGO_SUPERUSER_EMAIL", "")

        if not username or not password:
            self.stdout.write(
                self.style.WARNING(
                    "Skipping admin seed: DJANGO_SUPERUSER_USERNAME and "
                    "DJANGO_SUPERUSER_PASSWORD are required."
                )
            )
            return

        user_model = get_user_model()
        user, created = user_model.objects.get_or_create(
            username=username,
            defaults={
                "email": email,
                "is_active": True,
                "is_staff": True,
                "is_superuser": True,
            },
        )

        changed_fields: set[str] = set()
        if user.email != email:
            user.email = email
            changed_fields.add("email")
        if not user.is_active:
            user.is_active = True
            changed_fields.add("is_active")
        if not user.is_staff:
            user.is_staff = True
            changed_fields.add("is_staff")
        if not user.is_superuser:
            user.is_superuser = True
            changed_fields.add("is_superuser")
        if created or not user.check_password(password):
            user.set_password(password)
            changed_fields.add("password")

        if changed_fields:
            user.save()

        if created:
            self.stdout.write(
                self.style.SUCCESS(f"Created admin user '{username}'.")
            )
            return

        if changed_fields:
            fields = ", ".join(sorted(changed_fields))
            self.stdout.write(
                self.style.SUCCESS(
                    f"Updated admin user '{username}' ({fields})."
                )
            )
            return

        self.stdout.write(self.style.NOTICE(f"Admin user '{username}' already up to date."))
