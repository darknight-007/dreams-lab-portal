import uuid
from django.utils.deprecation import MiddlewareMixin

class SessionIDMiddleware(MiddlewareMixin):
    def process_request(self, request):
        if not request.session.get('openuav_session_id'):
            request.session['openuav_session_id'] = str(uuid.uuid4())
        return None
    
    def process_response(self, request, response):
        # If this is a new session, set the cookie
        if getattr(request, 'new_session', False):
            # Set cookie to expire in 24 hours
            response.set_cookie('openuav_session_id', 
                              request.session_id, 
                              max_age=86400,  # 24 hours in seconds
                              httponly=True,   # Cookie not accessible via JavaScript
                              samesite='Lax') # Prevents CSRF
        return response 