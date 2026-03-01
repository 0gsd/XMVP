import os
import logging
import subprocess
import base64
import requests
from mvp_shared import get_project_id

class CloudTTSBridge:
    def __init__(self):
        self.project_id = get_project_id()
        self.default_voice = "en-US-Journey-D"
        
    def get_access_token(self):
        try:
            cmd = ["gcloud", "auth", "print-access-token"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except Exception as e:
            logging.error(f"   ❌ Failed to get gcloud access token: {e}")
            return None

    def generate(self, text, output_path, voice_name=None, speed=1.0):
        """
        Synthesize text using Google Cloud TTS.
        """
        voice = voice_name if voice_name else self.default_voice
        token = self.get_access_token()
        
        if not token: 
            logging.error("   ❌ No GCP Token available. Ensure you are logged in via 'gcloud auth login'.")
            return False
            
        url = "https://texttospeech.googleapis.com/v1/text:synthesize"
        
        # Journey voices only support specific language codes usually matching their prefix
        # e.g. en-US-Journey-D -> en-US
        parts = voice.split("-")
        lang_code = "-".join(parts[:2]) if len(parts) >= 2 else "en-US"
        
        payload = {
            "input": {"text": text},
            "voice": {"languageCode": lang_code, "name": voice},
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "speakingRate": speed
            }
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        if self.project_id:
            headers["x-goog-user-project"] = self.project_id
            
        try:
            logging.info(f"   ☁️  Google Cloud TTS Speaking: '{text[:30]}...' ({voice})")
            res = requests.post(url, json=payload, headers=headers)
            
            if res.status_code == 200:
                audio_content = res.json().get('audioContent')
                if not audio_content:
                    logging.error("   ❌ Cloud TTS Error: No audioContent in response.")
                    return False
                    
                content = base64.b64decode(audio_content)
                with open(output_path, 'wb') as f:
                    f.write(content)
                return True
            else:
                logging.error(f"   ❌ Cloud TTS API Error {res.status_code}: {res.text}")
                return False
                
        except Exception as e:
            logging.error(f"   ❌ Cloud TTS Exception: {e}")
            return False

# Singleton pattern
_BRIDGE = None
def get_cloud_tts_bridge():
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = CloudTTSBridge()
    return _BRIDGE
