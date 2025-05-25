import psycopg2

DB_SETTINGS = {
    "host": "localhost",
    "port": 5432,
    "database": "",
    "user": "",
    "password": ""
}

try:
    conn = psycopg2.connect(**DB_SETTINGS)
    print("✅ Database connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Database connection failed: {e}")
