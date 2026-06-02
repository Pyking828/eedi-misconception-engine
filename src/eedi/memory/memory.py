"""
感知记忆模块（上游感知记忆 + 下游缓存/反馈，复刻 JD "上游感知记忆数据"）。

功能：
1. 查询缓存（qa_key → 已计算结果，TTL=24h）
2. 会话历史（用户多轮交互记录）
3. 用户反馈（教师/学生对诊断结果的好/差反馈）
4. 难负例缓冲（缓存召回结果供重训时挖掘难负例）

存储：aiosqlite（异步 SQLite），轻量、无需额外服务。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import aiosqlite


class MemoryModule:
    """
    异步 SQLite 记忆模块。

    示例：
        memory = MemoryModule("/root/autodl-tmp/eedi-data/memory.db")
        await memory.init()
        await memory.set_result(qa_key, misconception_ids)
        result = await memory.get_result(qa_key)
        await memory.add_feedback(qa_key, user_id="teacher_1", rating=5, comment="Correct!")
    """

    def __init__(
        self,
        db_path: str,
        cache_ttl_hours: float = 24.0,
    ) -> None:
        self.db_path = str(db_path)
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS result_cache (
                qa_key      TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                qa_key      TEXT NOT NULL,
                query_text  TEXT,
                result_json TEXT,
                created_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                qa_key      TEXT NOT NULL,
                user_id     TEXT,
                rating      INTEGER,
                comment     TEXT,
                created_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hard_neg_pool (
                qa_key          TEXT NOT NULL,
                misconception_id INTEGER NOT NULL,
                score           REAL,
                PRIMARY KEY (qa_key, misconception_id)
            );
            """
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ── 查询缓存 ────────────────────────────────

    async def get_result(self, qa_key: str) -> Optional[list[int]]:
        """返回缓存结果（若过期则返回 None）。"""
        async with self._db.execute(
            "SELECT result_json, created_at FROM result_cache WHERE qa_key=?",
            (qa_key,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        result_json, created_at = row
        if time.time() - created_at > self.cache_ttl_seconds:
            await self._db.execute("DELETE FROM result_cache WHERE qa_key=?", (qa_key,))
            await self._db.commit()
            return None
        return json.loads(result_json)

    async def set_result(self, qa_key: str, misconception_ids: list[int]) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO result_cache (qa_key, result_json, created_at) VALUES (?, ?, ?)",
            (qa_key, json.dumps(misconception_ids), time.time()),
        )
        await self._db.commit()

    # ── 会话历史 ────────────────────────────────

    async def log_session(
        self,
        session_id: str,
        qa_key: str,
        query_text: str,
        result: list[int],
    ) -> None:
        await self._db.execute(
            "INSERT INTO session_history (session_id, qa_key, query_text, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, qa_key, query_text, json.dumps(result), time.time()),
        )
        await self._db.commit()

    async def get_session_history(self, session_id: str, limit: int = 10) -> list[dict]:
        async with self._db.execute(
            "SELECT qa_key, query_text, result_json, created_at "
            "FROM session_history WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "qa_key": r[0],
                "query": r[1],
                "result": json.loads(r[2]),
                "ts": r[3],
            }
            for r in rows
        ]

    # ── 用户反馈 ────────────────────────────────

    async def add_feedback(
        self,
        qa_key: str,
        user_id: str = "anonymous",
        rating: int = 3,
        comment: str = "",
    ) -> None:
        await self._db.execute(
            "INSERT INTO user_feedback (qa_key, user_id, rating, comment, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (qa_key, user_id, rating, comment, time.time()),
        )
        await self._db.commit()

    async def get_feedback_stats(self) -> dict:
        async with self._db.execute(
            "SELECT AVG(rating), COUNT(*) FROM user_feedback"
        ) as cursor:
            row = await cursor.fetchone()
        return {"avg_rating": row[0], "total_feedback": row[1]}

    # ── 难负例缓存 ────────────────────────────────

    async def update_hard_negs(self, qa_key: str, misc_id: int, score: float) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO hard_neg_pool (qa_key, misconception_id, score) VALUES (?, ?, ?)",
            (qa_key, misc_id, score),
        )
        await self._db.commit()

    async def get_hard_negs(self, qa_key: str, top_k: int = 20) -> list[int]:
        async with self._db.execute(
            "SELECT misconception_id FROM hard_neg_pool "
            "WHERE qa_key=? ORDER BY score DESC LIMIT ?",
            (qa_key, top_k),
        ) as cursor:
            rows = await cursor.fetchall()
        return [r[0] for r in rows]
