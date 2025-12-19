from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends

# Import routers
from loggin import router as auth_router
from comment import router as comment_router

class Server():
    """
    singleton class for accessing the FastAPI app instance
    """
    app = FastAPI(title="JourNet API")
    def __init__(self):
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    def start(self, host: str = "127.0.0.1", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)
    def get_app(self):
        return self.app

# Get the app instance
server_instance = Server()
app = server_instance.get_app()

# Register routers
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(comment_router, prefix="/comments", tags=["comments"])

# Root route
@app.get("/")
def root():
    return {
        "message": "Welcome to JourNet API",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "auth": "/auth",
            "comments": "/comments",
            "items": "/items"
        }
    }

# Example route (can be removed)
class Item(BaseModel):
    name: str
    description: str

@app.post("/items/")
def make_response(item: Item):
    return {"item_name": item.name, "item_description": item.description}