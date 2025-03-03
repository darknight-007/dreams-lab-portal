from django.contrib.auth.backends import BaseBackend
from django.contrib.auth import get_user_model

User = get_user_model()

class PhoneNumberBackend(BaseBackend):
    """
    Custom authentication backend to authenticate users via phone number.
    """
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            # Look up the user by phone number
            user = User.objects.get(phone_number=username)
            if user.check_password(password):  # Verify the password
                return user
        except User.DoesNotExist:
            return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
