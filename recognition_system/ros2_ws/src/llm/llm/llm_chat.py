import time

from openai import OpenAI

class LLMChatModule:
    def __init__(self, logger):
        self.client = OpenAI(base_url="https://models.mylab.th-luebeck.dev/v1", api_key="-")
        self.logger = logger
        self.system_prompt = ""

    def init_with_context(self, name: str, last_seen: int) -> None:
        name_text = f"Die Person heißt {name}. Begrüße mit dem Namen. Wenn der Name anonym ist, dann erwähne ihn nicht." if name else ""
        if last_seen:
            days = int((time.time() - last_seen) / 86400)
            seen_info = f"Zuletzt gesehen vor {days} Tagen." if days != 0 else "Die Person war heute schon da."
        else:
            seen_info = ""
        self.system_prompt = (
            "Dein Name ist Lisa, ein freundlicher Roboter." +
            "Führe ein natürliches, kurzes Gespräch auf Deutsch mit einem Gesprächspartner." +
            "Antworte in maximal 2-3 Sätzen. Außer der Nutzer möchte explizit mehr Details," +
            "dann kannst du auch etwas ausführlicher antworten." +
            f"{name_text} {seen_info} " +
            "Wenn das Gespräch natürlich zu Ende ist (Verabschiedung, Person möchte gehen)," +
            "beende deine Antwort mit [ENDE] in eckigen Klammern. Sonst lasse es weg."
        )

    def chat(self, user_text: str, context: str) -> tuple[str, bool]:
        '''

        :param user_text:
        :param context:
        :return: text vom LLM und bool, ob [ENDE] im Text war (also Gespräch zu Ende)
        '''
        system_prompt_context = self.system_prompt + (f"\nRelevante Erinnerungen:\n{context}" if context else "")
        try:
            response = self.client.chat.completions.create(
                model="chat-large",
                messages=[
                    {"role": "system", "content": system_prompt_context},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.7,
            )
            raw = response.choices[0].message.content.strip()
            done = "[ENDE]" in raw
            reply = raw.replace("[ENDE]", "").strip()
            return reply, done
        except Exception as error:
            self.logger.error(f"LLM-Fehler: {error}")
            return "Entschuldigung, ich hatte gerade ein technisches Problem. Kannst du das von eben bitte wiederholen?", False
