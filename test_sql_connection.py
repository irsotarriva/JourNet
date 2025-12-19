#conda install -c conda-forge psycopg -y
import psycopg

DB_USER = "remote"
DB_PASSWORD = "journet"
DB_HOST = "172.31.2.247"
DB_PORT = 5432

# Connect to the admin database
conn = psycopg.connect(
    dbname="postgres",
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)
print("✅ Connected to PostgreSQL successfully")
with conn.cursor() as cur:
    # Get PostgreSQL version
    cur.execute("SELECT version();")
    version = cur.fetchone()[0]
    print("\nPostgreSQL version:")
    print(version)
    # List databases
    cur.execute("""
        SELECT datname
        FROM pg_database
        WHERE datistemplate = false
        ORDER BY datname;
    """)
    databases = cur.fetchall()
    print("\nDatabases on this server:")
    for db in databases:
        print(f" - {db[0]}")
conn.close()
print("\n🔌 Connection closed")
