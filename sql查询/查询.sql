SELECT * FROM embeddings_documents ORDER BY embedding <-> (select embedding from embeddings_questions) LIMIT 5;

