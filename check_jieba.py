import asyncio
from raven_agent.memory2 import MemoryStore2
import logging
import os

logging.basicConfig(level=logging.INFO)

async def check():
    db_path = '.raven/memory2/check-jieba.db'
    
    # 每次测试前删掉旧库，确保环境干净
    if os.path.exists(db_path):
        os.remove(db_path)
        
    # 1. 连上数据库，触发加载 jieba
    store = MemoryStore2(db_path, vec_dim=2)
    
    if not store._has_fts:
        print("❌ 失败：FTS 虚拟表创建失败，已降级为 LIKE！")
        return

    if store._simple_loaded:
        print("✅ 成功：已加载 libsimple 扩展，使用 jieba 分词器！")
    else:
        print("⚠️ 未加载 libsimple 扩展，使用内置 trigram 分词器（中文双字词可能查不到）")
    
    # 2. 【正确的插入方式】：写进主表，让 Trigger 自动同步给 FTS
    store.upsert_item(
        memory_type="event",
        summary="我今天路过了上海市长江大桥",
        embedding=None  # 我们只测分词，不需要向量
    )
    
    # 3. 跨表联合查询：用 MATCH 查 FTS 索引，把主表的原句带出来
    query = """
        SELECT m.summary 
        FROM memory_items_fts fts
        JOIN memory_items m ON fts.rowid = m.rowid
        WHERE memory_items_fts MATCH '长江'
    """
    cursor = store._db.execute(query)
    result = cursor.fetchall()
    
    if len(result) > 0:
        print(f"✅ 成功：MATCH 语法精准命中！找到了原句 -> {result[0][0]}")
    else:
        print("❌ 失败：MATCH 查不到词。")

asyncio.run(check())