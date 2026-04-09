from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import router

app = FastAPI(
    title="SLM Assistant",
    description="A RAG-powered assistant backed by Groq.",
    version="2.0.0",
)

app.include_router(router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("app/static/index.html")