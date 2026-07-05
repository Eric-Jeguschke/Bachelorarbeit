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
    """Extrahiert per LLM strukturierte Infos (Memories, Episode-Zusammenfassung) aus einem Gesprächstext."""

    def __init__(self, logger):
        self.client = OpenAI(base_url="https://models.mylab.th-luebeck.dev/v1", api_key="-")
        self.logger = logger

    def call_llm(self, system_prompt: str, conversation: str) -> str:
        """
        Schickt Prompt und Gespräch ans LLM und repariert/validiert die JSON-Antwort.
        Das LLM hält sich trotz Anweisung im Prompt nicht immer an valides JSON (z.B. Markdown-Codeblock drumherum,
        fehlende/überzählige Kommas). repair_json behebt solche typischen Fehler automatisch. json.loads() prüft
        danach zusätzlich, ob das reparierte Ergebnis wirklich ein valides JSON ist, sonst greift der Except-Block und
        es wird sicherheitshalber "{}" zurückgegeben, statt kaputtes JSON zurückzugeben.
        """
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
        """Fasst das Gespräch als Episode (Text + Keywords) zusammen."""
        return self.call_llm(EPISODE_PROMPT, conversation)

    def extract_memory(self, conversation: str) -> str:
        """Extrahiert Fakten über die Person aus dem Gespräch, kategorisiert nach den vier Kategorien im Prompt."""
        return self.call_llm(MEMORY_PROMPT, conversation)
