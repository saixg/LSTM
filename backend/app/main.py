from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import explainer, nextword, anomaly, sentiment

app = FastAPI(title="NeuroSeq API")
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "NeuroSeq API is running"}

app.include_router(explainer.router)
app.include_router(nextword.router)
app.include_router(anomaly.router)
app.include_router(sentiment.router)

# We will include routers in later phases once implemented:
# app.include_router(explainer.router)
# app.include_router(nextword.router)
# app.include_router(anomaly.router)
# app.include_router(sentiment.router)
