"""Extract page-aware passages and atomically replace the project's Postgres index."""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
SOURCES = {
    "moon": "https://science.nasa.gov/wp-content/uploads/2024/01/62217main-moon-lithograph.pdf",
    "guide": "https://science.nasa.gov/wp-content/uploads/2024/01/exploring-the-moon-teachers-guide.pdf",
}
ENCODER = "BAAI/bge-small-en-v1.5"


def connect():
    # libpq reads PGHOST, PGDATABASE, PGUSER and PGPASSWORD.
    return psycopg.connect(connect_timeout=5)


def passages():
    splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=120)
    for source in SOURCES:
        reader = PdfReader(ROOT / "data" / f"{source}.pdf")
        count = 0
        for page, item in enumerate(reader.pages, 1):
            for chunk in splitter.split_text(item.extract_text() or ""):
                count += 1
                yield source, page, chunk.replace("\x00", "")
        if not count:
            raise ValueError(f"No readable text in {source}.pdf")


def ingest():
    from sentence_transformers import SentenceTransformer

    with connect() as db:
        rows = list(passages())
        vectors = SentenceTransformer(ENCODER).encode(
            [row[2] for row in rows], normalize_embeddings=True
        )
        db.execute("""CREATE TABLE IF NOT EXISTS nasa_rag_chunks (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            source text NOT NULL, page integer NOT NULL, body text NOT NULL,
            embedding real[] NOT NULL CHECK (cardinality(embedding)=384),
            words tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
        )""")
        db.execute("CREATE INDEX IF NOT EXISTS nasa_rag_words ON nasa_rag_chunks USING gin(words)")
        # Readers see either the old corpus or the complete new corpus.
        db.execute("LOCK TABLE nasa_rag_chunks IN SHARE ROW EXCLUSIVE MODE")
        db.execute("DELETE FROM nasa_rag_chunks")
        with db.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO nasa_rag_chunks(source,page,body,embedding) VALUES (%s,%s,%s,%s)",
                [(*row, vec.tolist()) for row, vec in zip(rows, vectors, strict=True)],
            )
    print(f"Indexed {len(rows)} passages.")
