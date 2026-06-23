import json
import logging
import time
import uuid
from typing import Callable
import numpy as np

logger = logging.getLogger(__name__)

from db.db import get_profiles_table, get_episodes_table, get_episode_keywords_table, get_memories_table

# LanceDB Vektor-Suche: https://docs.lancedb.com/search/vector-search
def find_by_voice(voice_vector: list, threshold: float = 0.5) -> dict | None:
    results = (
        get_profiles_table()
        .search(voice_vector, vector_column_name="voice_emb")
        .distance_type("cosine")
        .limit(1)
        .to_list()
    )
    if not results:
        return None
    distance = float(results[0]["_distance"])
    recognized = 'erkannt' if distance <= threshold else 'unbekannt'
    logger.info(f"Beste Voice-Match-Distance: {distance:.3f}")
    logger.info(f"Threshold: {threshold} → {recognized}")
    if distance > threshold:
        return None
    else:
        return results[0]

def create_guest(name: str, voice_emb: list, episode_json: str, memory_json: str, text_embedder: Callable[[str], list[float]]) -> str:
    if len(voice_emb) != 256:
        return ""
    now = int(time.time())
    guest_id = str(uuid.uuid4())
    get_profiles_table().add([{
        "guest_id": guest_id,
        "name": name or "anonym",
        "voice_emb": voice_emb,
        "last_seen": now,
        "retention_until": now + 60 * 60 * 24 * 30, # 30 Tage aufbewahren
    }])
    if not add_episode(guest_id, episode_json, text_embedder):
        logger.warning(f"Keine Episode gespeichert für guest {guest_id}")
    if not update_memories(guest_id, memory_json, text_embedder):
        logger.warning(f"Keine Memories gespeichert für guest {guest_id}")
    return guest_id


def update_guest(guest_id: str, name: str, voice_emb: list, episode_json: str, memory_json: str, text_embedder: Callable[[str], list[float]]) -> str:
    if len(voice_emb) != 256:
        return ""
    now = int(time.time())
    values = {"last_seen": now}
    if name:
        values["name"] = name
    profile = get_profiles_table().search().where(f"guest_id = '{guest_id}'").limit(1).to_list()
    if profile:
        old_voice_emb = np.array(profile[0]["voice_emb"], dtype=np.float32)
        new_voice_emb = np.array(voice_emb, dtype=np.float32)
        updated_voice_emb = 0.4 * new_voice_emb + 0.6 * old_voice_emb
        values["voice_emb"] = updated_voice_emb.tolist()
    get_profiles_table().update(where=f"guest_id = '{guest_id}'", values=values)
    if not add_episode(guest_id, episode_json, text_embedder):
        logger.warning(f"Keine Episode gespeichert für guest {guest_id}")
    if not update_memories(guest_id, memory_json, text_embedder):
        logger.warning(f"Keine Memories gespeichert für guest {guest_id}")
    return guest_id


def add_episode(guest_id: str, episode_json: str, text_embedder: Callable[[str], list[float]]) -> bool:
    """episode_json: {"text": "Gast bestellte Kaffee", "keywords": ["Kaffee", "Zucker"]}"""
    if episode_json == "{}":
        return False
    data = json.loads(episode_json)
    episode_text = data.get("text", "").strip()
    keywords = []
    for k in data.get("keywords", []):
        keywords.append(k)
    if not episode_text or not keywords:
        return False

    episode_id = str(uuid.uuid4())
    get_episodes_table().add([{
        "episode_id": episode_id,
        "guest_id": guest_id,
        "episode_text": episode_text,
    }])

    keyword_table = get_episode_keywords_table()
    for keyword in keywords:
        keyword_table.add([{
            "keyword_id": str(uuid.uuid4()),
            "episode_id": episode_id,
            "guest_id": guest_id,
            "keyword": keyword,
            "keyword_emb": text_embedder(keyword),
        }])
    return True


def update_memories(guest_id: str, memory_json: str, text_embedder: Callable[[str], list[float]]) -> bool:
    """memory_json: '{"Vorlieben": ["mag Kaffee", "kein Zucker"], "Allergien": ["Nüsse"]}'"""
    if memory_json == "{}":
        return False
    data = json.loads(memory_json)

    memory_table = get_memories_table()
    for category, facts in data.items():
        for fact in facts:
            if not (memory_table.search()
                    .where(f"guest_id = '{guest_id}' AND category = '{category}' AND fact = '{fact}'").to_list()):
                memory_table.add([{
                    "memory_id": str(uuid.uuid4()),
                    "guest_id": guest_id,
                    "category": category,
                    "fact": fact,
                    "fact_emb": text_embedder(fact),
                }])
    return True



def retrieve_context(guest_id: str, search_emb: list) -> list:
    contexts = []
    try:
        seen_episodes = set()
        episode_keyword_rows = (((get_episode_keywords_table()
                                  .search(search_emb, vector_column_name="keyword_emb"))
                                 .where(f"guest_id = '{guest_id}'"))
                                .limit(5).to_list())
        for row in episode_keyword_rows:
            episode_id = row["episode_id"]
            if episode_id in seen_episodes:
                continue
            seen_episodes.add(episode_id)
            episode_row = (
                get_episodes_table()
                .search().where(f"episode_id = '{episode_id}'")
                .limit(1).to_list()
            )
            if episode_row:
                contexts.append(f"[Früherer Besuch] {episode_row[0]['episode_text']}") # Tags im Kontext für bessere RAG-Antworten (hier [Früherer Besuch])

        for row in (
                get_memories_table()
                        .search(search_emb, vector_column_name="fact_emb")
                        .where(f"guest_id = '{guest_id}'")
                        .limit(15).to_list()
        ):
            contexts.append(f"[{row['category']}] {row['fact']}")
    except Exception:
        return []
    return contexts


def clear_db():
    try:
        get_profiles_table().delete("guest_id IS NOT NULL") # LanceDB fügt automatisch WHERE hinzu im SQL-Befehl
        get_episodes_table().delete("guest_id IS NOT NULL")
        get_episode_keywords_table().delete("guest_id IS NOT NULL")
        get_memories_table().delete("guest_id IS NOT NULL")
    except Exception:
        pass


def clean_old_entries():
    now = int(time.time())
    try:
        expired = get_profiles_table().search().where(f"retention_until < {now}").to_list()
        for row in expired:
            guest_id = row["guest_id"]
            get_profiles_table().delete(f"guest_id = '{guest_id}'")
            get_episodes_table().delete(f"guest_id = '{guest_id}'")
            get_episode_keywords_table().delete(f"guest_id = '{guest_id}'")
            get_memories_table().delete(f"guest_id = '{guest_id}'")
    except Exception:
        pass
