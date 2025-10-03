from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

def create_app():
    """Create and configure FastAPI app"""
    app = FastAPI()
    templates = Jinja2Templates(directory="app/templates")
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app, templates
