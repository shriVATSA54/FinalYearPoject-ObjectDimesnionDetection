
import psycopg2
import psycopg2.extras
import json
from io import BytesIO

DB_SETTINGS = {
    "host": "localhost",
    "port": 5432,
    "database": "",
    "user": "",
    "password": ""
}

def connect_db():
    return psycopg2.connect(**DB_SETTINGS)

def initialize_table():
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS saved_results (
                id SERIAL PRIMARY KEY,
                image BYTEA,
                detections TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"Error initializing table: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()

def save_result_to_db(image_pil, detections):
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()

        img_buffer = BytesIO()
        image_pil.save(img_buffer, format="JPEG")
        img_bytes = img_buffer.getvalue()

        detections_json = json.dumps(detections)
        cur.execute(
            "INSERT INTO saved_results (image, detections) VALUES (%s, %s)",
            (psycopg2.Binary(img_bytes), detections_json)
        )
        conn.commit()
    except Exception as e:
        print(f"Error saving result to database: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()

def load_saved_results():
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM saved_results ORDER BY timestamp DESC")
        rows = cur.fetchall()
        return rows
    except Exception as e:
        print(f"Error loading saved results: {e}")
        return []
    finally:
        if cur: cur.close()
        if conn: conn.close()

def delete_result(result_id):
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM saved_results WHERE id = %s", (result_id,))
        conn.commit()
    except Exception as e:
        print(f"Error deleting result with ID {result_id}: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()
