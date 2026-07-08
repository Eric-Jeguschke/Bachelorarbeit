import time

from openai import OpenAI

class LLMChatModule:
    """Führt das Gespräch mit dem LLM: System-Prompt aufbauen, Antworten generieren, Gesprächsende erkennen."""

    def __init__(self, logger):
        self.client = OpenAI(base_url="https://models.mylab.th-luebeck.dev/v1", api_key="-", timeout=20.0)
        self.logger = logger
        self.system_prompt = ""

    def warmup(self) -> None:
        """
        Erste Anfrage ans LLM ist oft deutlich langsamer.
        Deshalb einmal Dummy-Anfrage senden, damit im  ersten echten Gespräch schnell antwortet.
        """
        try:
            t0 = time.time()
            self.client.chat.completions.create(
                model="chat-default",
                messages=[{"role": "user", "content": "Hallo"}],
                max_tokens=1,
            )
            self.logger.info(f"LLM-Warmup abgeschlossen ({time.time() - t0:.2f}s)")
        except Exception as error:
            self.logger.error(f"LLM-Warmup fehlgeschlagen: {error}")

    def init_with_context(self, name: str, last_seen: int) -> None:
        """Baut den System-Prompt inkl. Name und letztem Besuch der Person auf."""
        name_text = f"Die Person heißt {name}. Begrüße mit dem Namen. Wenn der Name anonym ist, dann erwähne ihn nicht." if name else ""
        if last_seen:
            days = int((time.time() - last_seen) / 86400)
            seen_info = f"Zuletzt gesehen vor {days} Tagen." if days != 0 else "Die Person war heute schon da."
        else:
            seen_info = ""
        self.system_prompt = (
            "Dein Name ist Nova, ein freundlicher Roboter von pib. " +
            "Führe ein natürliches, kurzes Gespräch auf Deutsch mit einem Gesprächspartner. " +
            "Antworte kurz, grundsätzlich 1-4 Sätze, entscheide selbst was sinnig erscheint. " +
            "Bei Bedarf darfst du auch gerne ausführlicher werden. " +
            f"{name_text} {seen_info} " +
            "Deine Antwort wird per Sprachausgabe vorgelesen: Schreibe nur Fließtext, " +
            "keine Listen, Emojis, Sonderzeichen oder Markdown-Formatierung. " +
            "Falls die Eingabe unklar ist oder du nicht verstanden hast was gesagt wurde, " +
            "frag kurz und locker nach, statt zu raten. " +
            "Wenn das Gespräch natürlich zu Ende ist (Verabschiedung, Person möchte gehen), " +
            "beende deine Antwort mit [ENDE] in eckigen Klammern. Sonst lasse es weg. " +
            "Wichtig: Wenn deine Antwort eine Frage an die Person enthält, setze niemals [ENDE]. " +
            "Das Gespräch geht dann eindeutig weiter."
        )

    def chat(self, user_text: str, context: str, conversation_history: str = "") -> tuple[str, bool]:
        """
        Schickt die Nutzernachricht ans LLM und gibt die bereinigte Antwort und das Gesprächsende-Flag zurück.
        Flag ist True, wenn das Gespräch beendet werden soll (LLM hat [ENDE] gesendet und die Antwort enthält keine
        Frage), sonst False, wenn das Gespräch weitergeht.
        conversation_history (falls vorhanden) und der Hinweis, worauf zu antworten ist, werden als eigene
        system-Nachrichten mitgeschickt statt in user_text eingebettet, damit user_text ausschließlich das enthält,
        was die Person gerade tatsächlich gesagt hat.
        """
        system_prompt_context = self.system_prompt + (f"\nRelevante Erinnerungen:\n{context}" if context else "")
        message = [{"role": "system", "content": system_prompt_context}]
        if conversation_history:
            message.append({"role": "system", "content": f"Bisheriger Gesprächsverlauf:\n{conversation_history}"})
        message.append({"role": "user", "content": user_text})
        try:
            t0 = time.time()
            response = self.client.chat.completions.create(
                model="chat-default",
                messages=message,
                temperature=0.7,
            )
            self.logger.info(f"LLM-Antwort erhalten ({time.time() - t0:.1f}s)")
            raw = response.choices[0].message.content.strip()
            reply = raw.replace("[ENDE]", "").strip() # [ENDE] entfernen, damit es nicht vorgelesen wird
            # Systemprompt verbietet [ENDE] bei Fragen, aber das LLM hält sich nicht zuverlässig daran.
            # Deshalb zusätzlich Gespräch nie auf einer Frage beenden, damit eine Frage nie das Gespräch beendet.
            done = "[ENDE]" in raw and "?" not in reply
            return reply, done
        except Exception as error:
            self.logger.error(f"LLM-Fehler: {error}")
            return "Entschuldigung, ich hatte gerade ein technisches Problem. Kannst du das von eben bitte wiederholen?", False
