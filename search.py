"""Hybrid retrieval with reciprocal rank fusion and a joint question/passage reranker."""

from collections import Counter
from functools import lru_cache

import numpy as np
from psycopg.rows import dict_row

from corpus import ENCODER, connect


@lru_cache(maxsize=1)
def models():
    from sentence_transformers import CrossEncoder, SentenceTransformer

    return SentenceTransformer(ENCODER), CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )


def retrieve(question):
    if not 1 <= len(question.strip()) <= 1500:
        raise ValueError("Enter a question between 1 and 1500 characters.")
    with connect() as db, db.cursor(row_factory=dict_row) as cur:
        # Keep the two retrieval branches on the same corpus snapshot.
        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        cur.execute(
            "SELECT id,source,page,body,embedding FROM nasa_rag_chunks ORDER BY id"
        )
        rows = cur.fetchall()
        if not rows:
            raise ValueError("Run python app.py ingest first.")
        cur.execute(
            """SELECT id FROM nasa_rag_chunks, plainto_tsquery('english', %s) q
            WHERE words @@ q ORDER BY ts_rank_cd(words,q) DESC,id LIMIT 20""",
            (question,),
        )
        lexical = [row["id"] for row in cur]
    encoder, reranker = models()
    vector = encoder.encode(
        "Represent this sentence for searching relevant passages: " + question,
        normalize_embeddings=True,
    )
    # Exact cosine search is inexpensive for these two PDFs. Both sides are unit vectors.
    similarities = (
        np.asarray([row["embedding"] for row in rows], dtype=np.float32) @ vector
    )
    dense = [rows[i]["id"] for i in np.argsort(-similarities, kind="stable")[:20]]
    scores = Counter()
    for ranking in (lexical, dense):
        for rank, key in enumerate(ranking, 1):
            scores[key] += 1 / (60 + rank)
    lookup = {row["id"]: row for row in rows}
    candidates = [lookup[key] for key, _ in scores.most_common(16)]
    relevance = reranker.predict([(question, row["body"]) for row in candidates])
    return [candidates[i] for i in np.argsort(-relevance)[:4]]
