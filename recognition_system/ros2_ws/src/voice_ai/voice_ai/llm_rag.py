import json

from json_repair import repair_json
from openai import OpenAI

MEMORY_PROMPT = """Du analysierst ein Gespräch zwischen einem Roboter und einer Person.
Extrahiere nur Informationen, die explizit im Gespräch erwähnt wurden.
Erfinde nichts und ziehe keine tieferen Schlüsse.

Teile die Informationen in diese vier Kategorien ein:
- Persönliche Infos: Name, Alter, Beruf, Familie, Herkunft, Wohnort etc.
- Interessen: Hobbys, Lieblingsthemen, Aktivitäten, Musik, Sport etc.
- Präferenzen: Vorlieben und Abneigungen (Essen, Musik, etc.)
- Sonstiges: Alles was in keine andere Kategorie passt, aber du trotzdem als sinnvoll betrachtest. Wird meistens leer bleiben.

Antworte ausschließlich mit einem JSON-Objekt in diesem Format:
{
  "persoenliche_infos": ["Stichpunkt 1", ...],
  "interessen": ["Stichpunkt 1", ...],
  "praeferenzen": ["Stichpunkt 1", ...],
  "sonstiges": ["Stichpunkt 1", ...]
}

Wenn eine Kategorie keine relevanten Infos enthält, gib eine leere Liste zurück."""

EPISODE_PROMPT = """Du fasst ein Gespräch zwischen einem Roboter und einer Person zusammen.
Fokussiere dich auf Ergebnisse, nicht auf den genauen Gesprächsverlauf.
Erfinde nichts und ziehe keine tieferen Schlüsse.

Antworte ausschließlich mit einem JSON-Objekt in diesem Format:
{
  "text": "2-5 Sätze: Worum ging es, was wurde entschieden, was ist offen?",
  "keywords": ["Stichpunkt 1", ...]
}

- text: Knapper Fließtext, der den Besuch zusammenfasst.
- keywords: 1-3 prägnante Begriffe (keine ganzen Sätze), die diesen Besuch charakterisieren.
  Beispiele: "Laptop-Einrichtung", "Termin vereinbart", "technische Hilfe"

Wenn du keine relevanten Informationen extrahieren kannst, gib {"text": "", "keywords": []} zurück."""


class RAGModule:
    def __init__(self, logger):
        self.client = OpenAI(base_url="https://models.mylab.th-luebeck.dev/v1", api_key="-")
        self.logger = logger

    def call_llm(self, system_prompt: str, conversation: str) -> str:
        if not conversation.strip():
            return "{}"
        try:
            response = self.client.chat.completions.create(
                model="chat-default",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation},
                ],
                temperature=0.1, # Eher deterministische Antworten, da konkrete Informationen extrahiert werden sollen
            )
            content = response.choices[0].message.content.strip()
            if not content:
                return "{}"
            repaired = repair_json(content)
            json.loads(repaired)  # sicherstellen dass es valides JSON ist
            return repaired
        except Exception as e:
            self.logger.error(f"RAG-Extraktion fehlgeschlagen: {e}")
            return "{}"

    def extract_episode(self, conversation: str) -> str:
        return self.call_llm(EPISODE_PROMPT, conversation)

    def extract_memory(self, conversation: str) -> str:
        return self.call_llm(MEMORY_PROMPT, conversation)
