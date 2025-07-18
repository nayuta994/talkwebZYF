import psycopg2
from psycopg2 import sql
from embedding import embedding, save_2_pgvector


def query(question: str):
    question_after_embedding = list(embedding([question]))[0][1]
    conn_params = {
        "host": "192.168.143.117",
        "port": "9002",
        "database": "pgvector",
        "user": "postgres",
        "password": "123456"
    }

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            # 查询
            sql_query = sql.SQL("""
                SELECT content, embedding <-> %s::real[] as similarity
                FROM embeddings_documents
                ORDER BY embedding <-> %s::real[]
                LIMIT 5;
            """)
            cur.execute(sql_query, (question_after_embedding, question_after_embedding))
            results = cur.fetchall()
            # for row in results:
            #     print(row)
            return results


if __name__ == '__main__':
    question = "What is Task Decomposition?"
    print(query(question))
