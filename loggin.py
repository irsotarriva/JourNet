from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError
from datetime import datetime, timedelta
import psycopg
import bcrypt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------
# Configuration
# -------------------

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Validate required environment variables
required_vars = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_NAME", "SECRET_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# -------------------
# Utility functions
# -------------------

def get_db_connection():
    return psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# -------------------
# Schemas
# -------------------

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    confirm_password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


# -------------------
# Auth helpers
# -------------------

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int | None = payload.get("sub")
        email: str | None = payload.get("email")

        if user_id is None or email is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, email FROM users WHERE id = %s;",
                (user_id,),
            )
            user = cur.fetchone()

            if not user:
                raise credentials_exception

            return {
                "id": user[0],
                "name": user[1],
                "email": user[2],
            }
    finally:
        conn.close()


# -------------------
# Routes
# -------------------

@router.post("/register")
def register_user(data: RegisterRequest):
    if data.password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    conn = get_db_connection()
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM users WHERE email = %s OR name = %s;",
                (data.email, data.name),
            )
            if cur.fetchone():
                raise HTTPException(
                    status_code=400,
                    detail="User with this email or name already exists",
                )

            password_hash = hash_password(data.password)

            cur.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (%s, %s, %s);
                """,
                (data.name, data.email, password_hash),
            )

        return {"message": "User registered successfully"}

    finally:
        conn.close()


@router.post("/login", response_model=Token)
def login_user(data: LoginRequest):
    conn = get_db_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, email, password_hash FROM users WHERE email = %s;",
                (data.email,),
            )
            user = cur.fetchone()

            if not user:
                raise HTTPException(status_code=401, detail="Invalid credentials")

            user_id, name, email, password_hash = user

            if not verify_password(data.password, password_hash):
                raise HTTPException(status_code=401, detail="Invalid credentials")

            access_token = create_access_token(
                data={"sub": str(user_id), "email": email},
                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            )

            return {
                "access_token": access_token,
                "token_type": "bearer",
            }

    finally:
        conn.close()


@router.get("/me")
def read_current_user(current_user=Depends(get_current_user)):
    return current_user
