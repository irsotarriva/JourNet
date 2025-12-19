# Launch the server using FastAPI
import server
if __name__ == "__main__":
    server_instance = Server()
    server_instance.start(host="0.0.0.0", port=8000)