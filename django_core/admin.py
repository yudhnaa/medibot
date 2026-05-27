from django.contrib import admin
from django.contrib.auth.models import Group

from django_celery_beat.models import (
    ClockedSchedule,
    CrontabSchedule,
    IntervalSchedule,
    PeriodicTask,
    SolarSchedule,
)
from django_celery_results.models import GroupResult, TaskResult
from rest_framework_simplejwt.token_blacklist.models import (
    BlacklistedToken,
    OutstandingToken,
)


def safe_unregister(*models):
    for model in models:
        if model in admin.site._registry:
            admin.site.unregister(model)


safe_unregister(
    # Django auth
    Group,
    # Celery Results
    TaskResult,
    GroupResult,
    # Celery Beat / Periodic Tasks
    ClockedSchedule,
    CrontabSchedule,
    IntervalSchedule,
    PeriodicTask,
    SolarSchedule,
    # Simple JWT Token Blacklist
    BlacklistedToken,
    OutstandingToken,
)
