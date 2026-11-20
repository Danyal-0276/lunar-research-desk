ALTER TABLE nasa_rag_chunks
ADD COLUMN character_count integer
GENERATED ALWAYS AS (char_length(body)) STORED;