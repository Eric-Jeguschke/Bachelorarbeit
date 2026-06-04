import lancedb
import pyarrow as pa

DEFAULT_DB_PATH = "/db/storage"

# https://docs.lancedb.com/tables
profiles_schema = pa.schema([
    pa.field("guest_id", pa.string()),
    pa.field("name", pa.string()),
    pa.field("voice_emb", pa.list_(pa.float32(), 256)),
    pa.field("last_seen", pa.int64()),
    pa.field("retention_until", pa.int64()),
])

episodes_schema = pa.schema([
    pa.field("episode_id", pa.string()),
    pa.field("guest_id", pa.string()),
    pa.field("timestamp", pa.int64()),
    pa.field("episode_json", pa.string()),
    pa.field("episode_emb", pa.list_(pa.float32(), 384)),
])

memories_schema = pa.schema([
    pa.field("memory_id", pa.string()),
    pa.field("guest_id", pa.string()),
    pa.field("category", pa.string()),
    pa.field("fact", pa.string()),
    pa.field("fact_emb", pa.list_(pa.float32(), 384)),
    pa.field("updated_at", pa.int64()),
])


def connect_db():
    return lancedb.connect(DEFAULT_DB_PATH)

def get_profiles_table():
    return connect_db().create_table("profiles", schema=profiles_schema, exist_ok=True) # Erstellt oder öffnet DB_Table

def get_episodes_table():
    return connect_db().create_table("episodes", schema=episodes_schema, exist_ok=True)

def get_memories_table():
    return connect_db().create_table("memories", schema=memories_schema, exist_ok=True)
