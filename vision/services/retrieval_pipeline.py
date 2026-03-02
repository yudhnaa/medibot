import json
import logging

import psycopg2
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)


def get_conn(cfg):
    conn = psycopg2.connect(
        host=cfg["pgvector"]["host"],
        port=cfg["pgvector"]["port"],
        dbname=cfg["pgvector"]["name"],
        user=cfg["pgvector"]["user"],
        password=cfg["pgvector"]["password"],
    )
    register_vector(conn)
    return conn


def ensure_table(conn, table: str, dim: int):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id SERIAL PRIMARY KEY,
                image_id TEXT,
                label TEXT,
                embedding vector({dim}),
                metadata JSONB
            )
            """
        )
    conn.commit()


def insert_embedding(
    conn, table: str, image_id: str, label: str, embedding, metadata: dict
):
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} (image_id, label, embedding, metadata) VALUES (%s, %s, %s, %s)",
            (image_id, label, embedding, json.dumps(metadata)),
        )
    conn.commit()


def query_similar(conn, table: str, embedding, k: int = 5):
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT image_id, label, metadata, embedding <-> %s AS distance FROM {table} ORDER BY embedding <-> %s LIMIT %s",
            (embedding, embedding, k),
        )
        rows = cur.fetchall()
    return rows
