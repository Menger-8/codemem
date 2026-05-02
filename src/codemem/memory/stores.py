"""SQLite-backed storage for all memory layers.

Handles persistence, vector similarity search, and graph traversal
for linked memories.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from codemem.memory.models import (
    EpisodicFact,
    EpisodicRecord,
    MemoryLink,
    MemoryNote,
    MemorySkill,
    Session,
    WorkingMemory,
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class MemoryStore:
    """SQLite-backed storage for the 4-layer memory system."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS memory_notes (
                id TEXT PRIMARY KEY,
                layer TEXT NOT NULL,
                content TEXT NOT NULL,
                keywords TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',
                context TEXT DEFAULT '',
                embedding BLOB,
                links TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS episodic_records (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                gist TEXT NOT NULL,
                facts TEXT DEFAULT '[]',
                files_involved TEXT DEFAULT '[]',
                action_type TEXT DEFAULT 'other',
                outcome TEXT DEFAULT 'unknown',
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS memory_skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                purpose TEXT NOT NULL,
                when_to_use TEXT NOT NULL,
                how_to_apply TEXT NOT NULL,
                constraints TEXT DEFAULT '',
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                created_from TEXT DEFAULT 'manual',
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_path TEXT DEFAULT '',
                started_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                turn_count INTEGER DEFAULT 0,
                summary TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS working_memory (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                task_summary TEXT DEFAULT '',
                active_files TEXT DEFAULT '[]',
                recent_outputs TEXT DEFAULT '[]',
                user_preferences TEXT DEFAULT '{}',
                internal_state TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_notes_layer ON memory_notes(layer);
            CREATE INDEX IF NOT EXISTS idx_episodic_session ON episodic_records(session_id);
            CREATE INDEX IF NOT EXISTS idx_episodic_time ON episodic_records(timestamp);
        """)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- Memory Notes (Semantic + Procedural) ---

    def store_note(self, note: MemoryNote) -> None:
        """Store or update a memory note."""
        cur = self.conn.cursor()
        embedding_blob = self._vec_to_blob(note.embedding)
        cur.execute(
            """INSERT OR REPLACE INTO memory_notes
               (id, layer, content, keywords, tags, context, embedding, links, created_at, updated_at, access_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                note.id,
                note.layer,
                note.content,
                json.dumps(note.keywords),
                json.dumps(note.tags),
                note.context,
                embedding_blob,
                json.dumps([l.model_dump() for l in note.links]),
                note.created_at.isoformat(),
                note.updated_at.isoformat(),
                note.access_count,
            ),
        )
        self.conn.commit()

    def get_note(self, note_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM memory_notes WHERE id = ?", (note_id,))
        row = cur.fetchone()
        if row:
            return self._row_to_note(row)
        return None

    def delete_note(self, note_id: str) -> bool:
        """Delete a memory note by ID."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM memory_notes WHERE id = ?", (note_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def search_notes(
        self,
        query_embedding: list[float],
        layer: Optional[str] = None,
        top_k: int = 5,
    ) -> list[tuple[MemoryNote, float]]:
        """Search memory notes by embedding similarity."""
        cur = self.conn.cursor()
        if layer:
            cur.execute("SELECT * FROM memory_notes WHERE layer = ?", (layer,))
        else:
            cur.execute("SELECT * FROM memory_notes")

        results = []
        query_vec = np.array(query_embedding, dtype=np.float32)
        for row in cur.fetchall():
            note = self._row_to_note(row)
            if note.embedding:
                sim = _cosine_similarity(query_vec, np.array(note.embedding, dtype=np.float32))
                results.append((note, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_notes_by_layer(self, layer: str) -> list[MemoryNote]:
        """Get all notes in a specific layer."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM memory_notes WHERE layer = ?", (layer,))
        return [self._row_to_note(row) for row in cur.fetchall()]

    def get_linked_notes(self, note_id: str) -> list[MemoryNote]:
        """Get all notes linked to a given note."""
        note = self.get_note(note_id)
        if not note:
            return []
        linked = []
        for link in note.links:
            linked_note = self.get_note(link.target_id)
            if linked_note:
                linked.append(linked_note)
        return linked

    # --- Episodic Records ---

    def store_episode(self, episode: EpisodicRecord) -> None:
        """Store an episodic record."""
        cur = self.conn.cursor()
        embedding_blob = self._vec_to_blob(episode.embedding)
        cur.execute(
            """INSERT OR REPLACE INTO episodic_records
               (id, session_id, timestamp, gist, facts, files_involved, action_type, outcome, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.id,
                episode.session_id,
                episode.timestamp.isoformat(),
                episode.gist,
                json.dumps([f.model_dump() for f in episode.facts]),
                json.dumps(episode.files_involved),
                episode.action_type,
                episode.outcome,
                embedding_blob,
            ),
        )
        self.conn.commit()

    def search_episodes(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        session_id: Optional[str] = None,
    ) -> list[tuple[EpisodicRecord, float]]:
        """Search episodic records by embedding similarity."""
        cur = self.conn.cursor()
        if session_id:
            cur.execute(
                "SELECT * FROM episodic_records WHERE session_id = ? ORDER BY timestamp DESC",
                (session_id,),
            )
        else:
            cur.execute("SELECT * FROM episodic_records ORDER BY timestamp DESC")

        results = []
        query_vec = np.array(query_embedding, dtype=np.float32)
        for row in cur.fetchall():
            episode = self._row_to_episode(row)
            if episode.embedding:
                sim = _cosine_similarity(query_vec, np.array(episode.embedding, dtype=np.float32))
                results.append((episode, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_recent_episodes(self, limit: int = 10) -> list[EpisodicRecord]:
        """Get most recent episodic records."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM episodic_records ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [self._row_to_episode(row) for row in cur.fetchall()]

    # --- Skills ---

    def store_skill(self, skill: MemorySkill) -> None:
        """Store or update a memory skill."""
        cur = self.conn.cursor()
        embedding_blob = self._vec_to_blob(skill.embedding)
        cur.execute(
            """INSERT OR REPLACE INTO memory_skills
               (id, name, purpose, when_to_use, how_to_apply, constraints, success_rate, usage_count, created_from, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                skill.id,
                skill.name,
                skill.purpose,
                skill.when_to_use,
                skill.how_to_apply,
                skill.constraints,
                skill.success_rate,
                skill.usage_count,
                skill.created_from,
                embedding_blob,
            ),
        )
        self.conn.commit()

    def search_skills(
        self,
        query_embedding: list[float],
        top_k: int = 2,
    ) -> list[tuple[MemorySkill, float]]:
        """Search skills by embedding similarity."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM memory_skills")

        results = []
        query_vec = np.array(query_embedding, dtype=np.float32)
        for row in cur.fetchall():
            skill = self._row_to_skill(row)
            if skill.embedding:
                sim = _cosine_similarity(query_vec, np.array(skill.embedding, dtype=np.float32))
                results.append((skill, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_all_skills(self) -> list[MemorySkill]:
        """Get all stored skills."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM memory_skills")
        return [self._row_to_skill(row) for row in cur.fetchall()]

    # --- Sessions ---

    def store_session(self, session: Session) -> None:
        """Store or update a session."""
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO sessions
               (id, project_path, started_at, last_active, turn_count, summary)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session.id,
                session.project_path,
                session.started_at.isoformat(),
                session.last_active.isoformat(),
                session.turn_count,
                session.summary,
            ),
        )
        self.conn.commit()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        if row:
            return Session(
                id=row["id"],
                project_path=row["project_path"],
                started_at=datetime.fromisoformat(row["started_at"]),
                last_active=datetime.fromisoformat(row["last_active"]),
                turn_count=row["turn_count"],
                summary=row["summary"],
            )
        return None

    def get_recent_sessions(self, limit: int = 10) -> list[Session]:
        """Get recent sessions."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM sessions ORDER BY last_active DESC LIMIT ?", (limit,))
        return [
            Session(
                id=row["id"],
                project_path=row["project_path"],
                started_at=datetime.fromisoformat(row["started_at"]),
                last_active=datetime.fromisoformat(row["last_active"]),
                turn_count=row["turn_count"],
                summary=row["summary"],
            )
            for row in cur.fetchall()
        ]

    # --- Working Memory ---

    def save_working_memory(self, wm: WorkingMemory) -> None:
        """Save working memory state (singleton)."""
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO working_memory
               (id, task_summary, active_files, recent_outputs, user_preferences, internal_state)
               VALUES (1, ?, ?, ?, ?, ?)""",
            (
                wm.task_summary,
                json.dumps(wm.active_files),
                json.dumps(wm.recent_outputs),
                json.dumps(wm.user_preferences),
                wm.internal_state,
            ),
        )
        self.conn.commit()

    def load_working_memory(self) -> WorkingMemory:
        """Load working memory state."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM working_memory WHERE id = 1")
        row = cur.fetchone()
        if row:
            return WorkingMemory(
                task_summary=row["task_summary"],
                active_files=json.loads(row["active_files"]),
                recent_outputs=json.loads(row["recent_outputs"]),
                user_preferences=json.loads(row["user_preferences"]),
                internal_state=row["internal_state"],
            )
        return WorkingMemory()

    # --- Statistics ---

    def get_stats(self) -> dict:
        """Get memory statistics."""
        cur = self.conn.cursor()
        stats = {}
        for layer in ("semantic", "episodic", "procedural"):
            cur.execute("SELECT COUNT(*) FROM memory_notes WHERE layer = ?", (layer,))
            stats[f"{layer}_notes"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM episodic_records")
        stats["episodes"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM memory_skills")
        stats["skills"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM sessions")
        stats["sessions"] = cur.fetchone()[0]
        return stats

    # --- Helpers ---

    @staticmethod
    def _vec_to_blob(vec: list[float]) -> Optional[bytes]:
        if not vec:
            return None
        return np.array(vec, dtype=np.float32).tobytes()

    @staticmethod
    def _blob_to_vec(blob: Optional[bytes]) -> list[float]:
        if not blob:
            return []
        return np.frombuffer(blob, dtype=np.float32).tolist()

    def _row_to_note(self, row: sqlite3.Row) -> MemoryNote:
        return MemoryNote(
            id=row["id"],
            layer=row["layer"],
            content=row["content"],
            keywords=json.loads(row["keywords"]),
            tags=json.loads(row["tags"]),
            context=row["context"],
            embedding=self._blob_to_vec(row["embedding"]),
            links=[MemoryLink(**l) for l in json.loads(row["links"])],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            access_count=row["access_count"],
        )

    def _row_to_episode(self, row: sqlite3.Row) -> EpisodicRecord:
        return EpisodicRecord(
            id=row["id"],
            session_id=row["session_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            gist=row["gist"],
            facts=[EpisodicFact(**f) for f in json.loads(row["facts"])],
            files_involved=json.loads(row["files_involved"]),
            action_type=row["action_type"],
            outcome=row["outcome"],
            embedding=self._blob_to_vec(row["embedding"]),
        )

    def _row_to_skill(self, row: sqlite3.Row) -> MemorySkill:
        return MemorySkill(
            id=row["id"],
            name=row["name"],
            purpose=row["purpose"],
            when_to_use=row["when_to_use"],
            how_to_apply=row["how_to_apply"],
            constraints=row["constraints"],
            success_rate=row["success_rate"],
            usage_count=row["usage_count"],
            created_from=row["created_from"],
            embedding=self._blob_to_vec(row["embedding"]),
        )
