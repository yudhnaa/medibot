from django.urls import path

from authentication.views import (
    LoginView,
    LogoutView,
    RefreshView,
    RegisterView,
    UserProfileView,
)

urlpatterns = [
    path("login/", LoginView.as_view(), name="token_obtain_pair"),
    path("logout/", LogoutView.as_view(), name="token_blacklist"),
    path("token/refresh/", RefreshView.as_view(), name="token_refresh"),
    path("register/", RegisterView.as_view(), name="auth_register"),
    path("me/", UserProfileView.as_view(), name="user_profile"),
]
