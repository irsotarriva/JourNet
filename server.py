from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

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

#example
server_instance = Server()
app = server_instance.get_app()
class Item(BaseModel):
    name: str
    description: str

@app.post("/items/")
def make_response(item: Item):
    return {"item_name": item.name, "item_description": item.description}