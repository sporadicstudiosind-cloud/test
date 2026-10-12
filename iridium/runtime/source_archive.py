"""Exact source retention complements lossy neural memory; ordinary code only."""
import hashlib
import sqlite3


class SourceArchive:
    """Explicit per-project disk archive. Never silently copies user documents.

    Call add_text yourself to retain a source. Retrieval is lexical FTS, not a
    second neural model. Read source chunks by id to recover exact text.
    """
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS sources USING fts5(id UNINDEXED, source UNINDEXED, text)')

    def add_text(self, source, text, chunk_chars=4000):
        if chunk_chars < 1:
            raise ValueError('chunk_chars must be positive')
        ids = []
        with self.db:
            for start in range(0, len(text), chunk_chars):
                chunk = text[start:start+chunk_chars]
                key = hashlib.sha256((str(source) + '\0' + str(start) + '\0' + chunk).encode()).hexdigest()
                if self.db.execute('SELECT 1 FROM sources WHERE id=?', (key,)).fetchone() is None:
                    self.db.execute('INSERT INTO sources VALUES (?,?,?)', (key, str(source), chunk))
                ids.append(key)
        return ids

    def search(self, query, limit=4):
        if not 1 <= limit <= 32:
            raise ValueError('limit must be 1..32')
        # Quote user words: do not accept raw FTS query syntax.
        words = query.split()[:32]
        if not words:
            return []
        expression = ' OR '.join('"' + word.replace('"', '""') + '"' for word in words)
        return [dict(zip(('id', 'source', 'text'), row)) for row in self.db.execute(
            'SELECT id,source,text FROM sources WHERE sources MATCH ? ORDER BY rank LIMIT ?', (expression, limit))]

    def read(self, identity):
        row = self.db.execute('SELECT source,text FROM sources WHERE id=?', (identity,)).fetchone()
        if row is None:
            raise KeyError(identity)
        return {'id': identity, 'source': row[0], 'text': row[1]}

    def close(self):
        self.db.close()
