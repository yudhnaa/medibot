"""
Django signals for chatbot app.
Handles automatic creation of related models.
"""

from django.db.models.signals import post_save
from django.dispatch import receiver

from authentication.models import Customer
from chatbot.models import UserIntake


@receiver(post_save, sender=Customer)
def create_user_intake(sender, instance, created, **kwargs):
    """
    Auto-create UserIntake when Customer is created.

    Args:
        sender: Model class (Customer)
        instance: Customer instance
        created: Boolean indicating if this is a new record
        **kwargs: Additional signal arguments
    """
    if created:
        UserIntake.objects.create(customer=instance)
