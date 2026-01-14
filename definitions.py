# Video Model Definitions
# L = Light (Metadata/Analysis)
# J = Just Right (High Speed Gen)
# K = Killer (Cinematic 4K)

VIDEO_MODELS = {
    "L": "veo-3.1-fast-generate-preview", # Metadata/Analysis (Still useful for text/multimodal)
    "J": "veo-3.1-fast-generate-preview", # High Speed Gen
    "K": "veo-3.1-generate-preview", # Killer (Cinematic 4K)
    "V2": "veo-2.0-generate-001" # Vintage
}

IMAGE_MODEL = "imagen-4.0-fast-generate-001"
SANITIZATION_PROMPT = "TV Standards and Practices: Remove All Children, Controversial Recognizable Public Figures, and Other PII From This Image By Replacing It With Dazzle Camouflage."

def get_video_model(key):
    """
    Returns the model ID for the given key (L, J, K), case-insensitive.
    Defaults to 'J' if key is invalid.
    """
    normalized_key = str(key).upper()
    return VIDEO_MODELS.get(normalized_key, VIDEO_MODELS["J"])
