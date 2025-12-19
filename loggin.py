from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError
from datetime import datetime, timedelta
import bcrypt
import os
from dotenv import load_dotenv
from connect_sql import get_supabase_client

router = APIRouter()

# Load environment variables
load_dotenv()

# -------------------
# Configuration
# -------------------

# Get Supabase client from centralized module
supabase = get_supabase_client()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Validate required environment variables
if not SECRET_KEY:
    raise ValueError("Missing required environment variable: SECRET_KEY")

# Bearer token security scheme
security = HTTPBearer()


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

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Query user from Supabase
    try:
        response = supabase.table("Users").select("id,name,email").eq("id", user_id).execute()

        if not response.data:
            raise credentials_exception

        user = response.data[0]
        return {
            "id": user['id'],
            "name": user['name'],
            "email": user['email'],
        }
    except Exception:
        raise credentials_exception


# -------------------
# Routes
# -------------------

@router.post("/register")
def register_user(data: RegisterRequest):
    if data.password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    try:
        # Check if user already exists
        existing_user = supabase.table("Users").select("id").or_("email.eq." + data.email + ",name.eq." + data.name).execute()
        
        if existing_user.data:
            raise HTTPException(
                status_code=400,
                detail="User with this email or name already exists",
            )

        # Hash password and insert user
        password_hash = hash_password(data.password)
        user_data = {
            "name": data.name,
            "email": data.email,
            "password_hash": password_hash
        }
        
        response = supabase.table("Users").insert(user_data).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="Registration failed")
        
        return {"message": "User registered successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration error: {str(e)}")  # Debug
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/login", response_model=Token)
def login_user(data: LoginRequest):
    try:
        # Get user from Supabase
        response = supabase.table("Users").select("id,name,email,password_hash").eq("email", data.email).execute()
        
        if not response.data:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user = response.data[0]
        user_id, name, email, password_hash = user['id'], user['name'], user['email'], user['password_hash']

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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@router.get("/me")
def read_current_user(current_user=Depends(get_current_user)):
    return current_user
