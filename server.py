from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi import Depends

class Server():
    """
    singleton class for accessing the FastAPI app instance
    """
    app = FastAPI()
    def __init__(self):
        pass
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

#example
server_instance = Server()
app = server_instance.get_app()
class Item(BaseModel):
    name: str
    description: str

@app.post("/items/")
def make_response(item: Item):
    return {"item_name": item.name, "item_description": item.description}