import os

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://reidbackend:reidbackend@localhost:5432/reid")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "gallery")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "reid-backend")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "reid-backend-secret")
S3_BUCKET = os.environ.get("S3_BUCKET", "gallery")

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8001")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "JDNFV_MASKED_FT")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "2048"))
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.87"))

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
