from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import os, json, base64

# Usa drive.file por defecto (compatible con tu refresh token actual).
# Si algún workflow requiere otro scope, se puede sobreescribir con la env GOOGLE_DRIVE_SCOPE.
DEFAULT_SCOPE = "https://www.googleapis.com/auth/drive.file"
SCOPES = [os.environ.get("GOOGLE_DRIVE_SCOPE", DEFAULT_SCOPE)]

def authenticate_drive():
    b64 = os.environ.get("GOOGLE_OAUTH_B64")
    if not b64:
        raise RuntimeError("Falta GOOGLE_OAUTH_B64 en el entorno (secret Base64)")
    info = json.loads(base64.b64decode(b64).decode("utf-8"))
    creds = Credentials(
        token=None,
        refresh_token=info["refresh_token"],
        client_id=info["client_id"],
        client_secret=info["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=creds)

