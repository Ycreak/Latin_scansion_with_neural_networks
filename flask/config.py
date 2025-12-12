import os

# Trusted origins for CORS purposes
LOCAL_HOST_RESOLVED = "http://localhost:4200"
LOCAL_HOST = "http://127.0.0.1:4200"
# REMOTE_HOST: str = os.getenv("REMOTE_HOST")
# STAGING_HOST: str = os.getenv("STAGING_HOST")

TRUSTED_ORIGINS = [
    LOCAL_HOST,
    LOCAL_HOST_RESOLVED,
    # REMOTE_HOST,
    # STAGING_HOST,
]

# Name of our log file
LOG_FILE = "server.log"
