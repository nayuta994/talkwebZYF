import psycopg2


def connect_to_pgvector(host, port, database, user, password):
    # 建立连接
    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        return connection

    except Exception as e:
        print("连接数据库时出错:", e)


# 数据库连接信息
data = {
    'host': "192.168.143.117",
    'port': "9002",
    'database': "knowledge_database",
    'user': "postgres",
    'password': "123456"
}

# 建立连接
connection = connect_to_pgvector(**data)

# 创建一个游标对象
cursor = connection.cursor()

# 执行一个简单的查询
cursor.execute("SELECT version();")
db_version = cursor.fetchone()
print("数据库版本:", db_version)

# 关闭游标和连接
cursor.close()
connection.close()
