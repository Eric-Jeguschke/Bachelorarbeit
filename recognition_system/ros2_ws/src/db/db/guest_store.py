import json
import logging
import time
import uuid
import numpy as np
from typing import Callable
from db.db import get_profiles_table, get_episodes_table, get_episode_keywords_table, get_memories_table

logger = logging.getLogger(__name__)


# LanceDB Vektor-Suche: https://docs.lancedb.com/search/vector-search
def find_by_voice(voice_vector: list, threshold: float = 0.549) -> dict | None:
    """Sucht das ähnlichste Stimm-Embedding in der DB und gibt ein Profil zurück, wenn die Distanz unter dem Threshold liegt."""
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
    logger.info(f"Threshold: {threshold} -> {recognized}")
    if distance > threshold:
        return None
    else:
        return results[0]

def create_guest(name: str, voice_emb: list, episode_json: str, memory_json: str, text_embedder: Callable[[str], list[float]]) -> str:
    """
    Legt ein neues Gast-Profil mit Stimm-Embedding an und speichert optional die zugehörige episode_json und memory_json. Beispiele:
    episode_json: {"text": "Gast bestellte Kaffee mit Zucker", "keywords": ["Kaffee", "Zucker"]}
    memory_json: {"persoenliche_infos": ["lebt in Lübeck"], "interessen": ["Fußball"], "praeferenzen": ["mag Kaffee"], "sonstiges": []}

    voice_emb wird nur mit einer Länge von 256 akzeptiert. Der Gast wird nach 30 Tagen gelöscht, wenn er innerhalb dieses Zeitraums nicht wieder mit dem Roboter redet.
    text_embedder: Funktion, die einen Text-String in ein Embedding (Vektor) umwandelt, damit Keywords/Fakten später per Vektorsuche gefunden werden können.
    """
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
    """
    Aktualisiert ein bestehendes Gast-Profil: Name wird überschrieben, falls angegeben. Stimm-Embedding per EMA geglättet
    (0.4 * neues Embedding + 0.6 * altes Embedding), retention_until wird um weitere 30 Tage ab jetzt verlängert.
    Episode und Memories werden nicht überschrieben, sondern zusätzlich angehängt. add_episode legt immer einen
    neuen Eintrag an, update_memories fügt neue Fakten hinzu und überspringt bereits vorhandene identische Fakten.
    voice_emb muss hier auch eine Länge von 256 haben.

    Parameter Beispiele:
    episode_json: {"text": "Gast bestellte Kaffee mit Zucker", "keywords": ["Kaffee", "Zucker"]}
    memory_json: {"persoenliche_infos": ["lebt in Lübeck"], "interessen": ["Fußball"], "praeferenzen": ["mag Kaffee"], "sonstiges": []}
    text_embedder: Funktion, die einen Text-String in ein Embedding (Vektor) umwandelt, damit Keywords/Fakten später per Vektorsuche gefunden werden können.
    """
    if len(voice_emb) != 256:
        return ""
    now = int(time.time())
    values = {"last_seen": now, "retention_until": now + 60 * 60 * 24 * 30} # Daten 30 Tage ab diesem Besuch weiter aufbewahren, sonst würde retention_until vom ersten Besuch nie verlängert werden
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
    """
    Speichert eine Episode (Zusammenfassung + Keywords) zu einem Gast.
    Beispiel:
    episode_json: {"text": "Gast bestellte Kaffee mit Zucker", "keywords": ["Kaffee", "Zucker"]}
    text_embedder: Funktion, die jedes Keyword in ein Embedding (Vektor) umwandelt, das zusammen mit dem Keyword gespeichert wird (keyword_emb).
    """
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
    """
    Speichert neue Fakten zu einem Gast und überspringt bereits vorhandene identische Fakten.
    Beispiel:
    memory_json: {"persoenliche_infos": ["lebt in Lübeck"], "interessen": ["Fußball"], "praeferenzen": ["mag Kaffee"], "sonstiges": []}
    text_embedder: Funktion, die jeden neuen Fakt in ein Embedding (Vektor) umwandelt, das zusammen mit dem Fakt gespeichert wird (fact_emb).
    """
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
    """
    Holt per Vektorsuche relevante frühere Episoden und Fakten zu einem Gast als RAG-Kontext.
    search_emb ist das Embedding der aktuellen Anfrage/Aussage und wird gegen die gespeicherten
    keyword_emb (Keywords vergangener Episoden) und fact_emb (Fakten) verglichen. Bei semantischen Übereinstimmungen
    werden die Episoden/Fakten zurückgegeben, die dann als Kontext für RAG genutzt werden können.
    """
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

        for row in (get_memories_table()
                .search(search_emb, vector_column_name="fact_emb")
                .where(f"guest_id = '{guest_id}'")
                .limit(15).to_list()
        ):
            contexts.append(f"[{row['category']}] {row['fact']}")
    except Exception:
        return []
    return contexts


def clear_db():
    """Löscht alle Einträge aus allen Tabellen."""
    try:
        get_profiles_table().delete("guest_id IS NOT NULL") # LanceDB fügt automatisch WHERE hinzu im SQL-Befehl
        get_episodes_table().delete("guest_id IS NOT NULL")
        get_episode_keywords_table().delete("guest_id IS NOT NULL")
        get_memories_table().delete("guest_id IS NOT NULL")
    except Exception:
        pass


def clean_old_entries():
    """Löscht Gast-Profile (und alle anderen zugehörigen Tabellen episodes/memories/episode_keywords), deren Aufbewahrungsfrist abgelaufen ist."""
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
