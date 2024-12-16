from twilio.rest import Client
import os

class TwilioVerify:
    def __init__(self):
        self.client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        self.service_sid = os.getenv("TWILIO_VERIFY_SERVICE_SID")

    def send_verification(self, to, channel="sms"):
        """Send a verification code."""
        try:
            verification = self.client.verify.services(self.service_sid).verifications.create(
                to=to,
                channel=channel
            )
            return verification.status
        except Exception as e:
            return str(e)

    def check_verification(self, to, code):
        """Verify the code."""
        try:
            verification_check = self.client.verify.services(self.service_sid).verification_checks.create(
                to=to,
                code=code
            )
            return verification_check.status
        except Exception as e:
            return str(e)
