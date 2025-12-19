import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables once
load_dotenv()

# Singleton instance
_supabase_client: Client | None = None


def get_supabase_client() -> Client:
    """
    Returns a singleton Supabase client instance.
    This ensures the same client is reused across the application.
    """
    global _supabase_client

    if _supabase_client is None:
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")

        _supabase_client = create_client(url, key)

    return _supabase_client


# For backward compatibility and convenience
supabase: Client = get_supabase_client()