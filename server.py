from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends

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
    def get_current_user(self, token: str = Depends(oauth2_scheme)):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                raise HTTPException(status_code=401)
        except JWTError:
            raise HTTPException(status_code=401)

        user = get_user_by_id(int(user_id))
        if not user:
            raise HTTPException(status_code=401)

        return user

# Get the app instance
server_instance = Server()
app = server_instance.get_app()

# Import auth router from loggin
from loggin import router as auth_router
app.include_router(auth_router, prefix="/auth", tags=["auth"])

# Example route (can be removed)
class Item(BaseModel):
    name: str
    description: str

@app.post("/items/")
def make_response(item: Item):
    return {"item_name": item.name, "item_description": item.description}