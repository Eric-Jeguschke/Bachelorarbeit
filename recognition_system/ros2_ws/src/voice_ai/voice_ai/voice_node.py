import threading
import time
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from interfaces.msg import PersonDetected
from interfaces.srv import CreateGuest, UpdateGuest, Listen, RetrieveContext, TTS, FindByVoice
from voice_ai.llm_chat import LLMChatModule
from voice_ai.llm_rag import RAGModule


class VoiceNode(Node):
    """ROS2-Node: Führt nach der recognition-Node das Gespräch (LLM, TTS, Wiedererkennung im Gespräch (wenn noch nicht erkannt)) und speichert das Ergebnis."""

    def __init__(self):
        super().__init__("voice_ai")
        self.llm = LLMChatModule(self.get_logger())
        self.llm.warmup()
        self.rag = RAGModule(self.get_logger())

        self.mic_client = self.create_client(Listen, "/microphone/stt")
        self.speaker_client = self.create_client(TTS, "/speaker/tts")
        self.db_create_guest_client = self.create_client(CreateGuest, "/db/create_guest")
        self.db_update_guest_client = self.create_client(UpdateGuest, "/db/update_guest")
        self.db_find_guest_client = self.create_client(FindByVoice, "/db/find_by_voice")
        self.db_context_client = self.create_client(RetrieveContext, "/db/retrieve_context")
        self.conv_done_pub = self.create_publisher(String, "/voice_ai/conversation_done", 1)  # 1 = Queue-Größe
        self.create_subscription(PersonDetected, "/recognition/person_detected", self.person_detected, 1)
        self.conversation_running = False
        self.get_logger().info("Voice AI node gestartet")

    def person_detected(self, person: PersonDetected):
        """Callback: Startet ein neues Gespräch in einem eigenen Thread, sobald eine Person erkannt wurde."""
        if self.conversation_running:
            return
        self.conversation_running = True
        # Eigener Thread nötig: run_conversation blockiert mit time.sleep() in call_service.
        # Läuft run_conversation direkt im Callback, kann der Executor die Service-Response nicht verarbeiten -> Deadlock.
        threading.Thread(target=self.run_conversation, args=(person,), daemon=True).start()

    def run_conversation(self, person: PersonDetected):
        """
        Führt die Gesprächsschleife (aufnehmen, transkribieren, LLM-Antwort, TTS) bis zum Gesprächsende durch und
        speichert danach das Gespräch (Memories, Episode, Voice-Embedding) in der Datenbank, sofern überhaupt etwas
        gesprochen wurde. Als Voice-Embedding wird bevorzugt das vorletzte statt das letzte genommen, da die letzte
        Aufnahme oft nur eine kurze Verabschiedung ("Tschüss") ist und dafür ein schlechteres Embedding liefert.
        """
        try:
            self.llm.init_with_context(person.name, person.last_seen)
            conversation_text = ""
            last_voice_emb = list(person.voice_emb) if person.voice_emb else []
            prev_voice_emb: list = []  # Vorletztes Embedding. Besser als letztes, das oft nur "Tschüss" ist
            reidentification = person.found  # True = schon erkannt, kein weiterer Versuch nötig

            # Erstes Transkript kommt direkt aus Recognition. Kein erneutes Aufnehmen nötig
            first_transcript = person.transcript.strip() if person.transcript else ""

            while True:
                if first_transcript:
                    spoken_text = first_transcript
                    first_transcript = ""  # Hiernach soll nurnoch record() verwendet werden, damit nicht immer wieder das gleiche Transkript aus Recognition verwendet wird
                else:
                    audio = self.record()
                    if not audio:
                        self.get_logger().error("Keine Audio-Daten erhalten")
                        break

                    if audio["voice_emb"]:
                        prev_voice_emb = last_voice_emb # Vorheriges Embedding speichern, um es später als "bestes" zu verwenden, da das letzte häufig zu kurz für ein gutes Voice-Embedding ist.
                        last_voice_emb = audio["voice_emb"]

                        # Einmal im Gespräch nochmal versuchen Person zu erkennen. Embedding aus Begrüßung schlägt häufig fehl, da zu kurz. Deswegen längere aus richtigem Gespräch.
                        if not reidentification:
                            reidentification = True
                            req = FindByVoice.Request()
                            req.voice_emb = last_voice_emb
                            result = self.call_service(self.db_find_guest_client, req, timeout=3.0)
                            if result and result.found:
                                person.found = True
                                person.guest_id = result.guest_id
                                person.name = result.name
                                self.llm.init_with_context(person.name, person.last_seen)
                                self.get_logger().info(f"Person im Gespräch erkannt: {result.name}")

                    if not audio["transcript"]:
                        if not self.speak("Entschuldigung, ich habe dich leider nicht verstanden. Kannst du das wiederholen?"):
                            self.get_logger().warning("Abbruch: TTS für Nachfrage fehlgeschlagen.")
                            break
                        audio = self.record()
                        if not audio or not audio["transcript"]:
                            self.get_logger().warning("Abbruch: Beim zweiten Versuch Person zu verstehen keine Sprache erkannt.")
                            break
                        if audio["voice_emb"]:
                            prev_voice_emb = last_voice_emb
                            last_voice_emb = audio["voice_emb"]
                    spoken_text = audio["transcript"]

                context = self.fetch_context(person.guest_id, spoken_text) if (person.found and person.guest_id) else ""
                user_message = spoken_text if not conversation_text else f"{spoken_text}\nZuvor gesprochener Text: {conversation_text}"
                reply, done = self.llm.chat(user_message, context)
                if not self.speak(reply):
                    self.get_logger().warning("Abbruch: TTS für LLM-Antwort fehlgeschlagen.")
                    break
                conversation_text += f"Nutzer: {spoken_text}\nRoboter: {reply}\n"
                if len(conversation_text) > 30000:  # Prompt nicht zu lang werden lassen
                    conversation_text = conversation_text[-30000:]
                if done:
                    self.get_logger().info("Gespräch regulär beendet (LLM hat [ENDE] gesendet).")
                    break

            if conversation_text.strip():
                voice_emb_to_save = prev_voice_emb if prev_voice_emb else last_voice_emb
                self.save_and_finish(person, conversation_text, voice_emb_to_save)
            else:
                self.get_logger().info("Kein Gespräch geführt. Nichts gespeichert.")
        except Exception as e:
            self.get_logger().error(f"run_conversation fehlgeschlagen: {e}")
        finally:
            # Immer ausführen, auch bei Exception. Sonst bleibt conversation_running = True und recognition wartet ewig
            self.conversation_running = False
            self.conv_done_pub.publish(String())

    def record(self) -> dict | None:
        """
        Nimmt über den Mikrofon-Service eine Äußerung auf.
        Gibt bei Erfolg {"transcript": str, "voice_emb": list} zurück, sonst None (z.B. bei Timeout)
        """
        result = self.call_service(self.mic_client, Listen.Request(), timeout=60.0)  # Länger warten, da Person vielleicht lange spricht
        if result and result.success:
            return {
                "transcript": result.transcript,
                "voice_emb": list(result.voice_emb),
            }
        return None

    def fetch_context(self, guest_id: str, user_text: str) -> str:
        """
        Holt RAG-Kontext zur aktuellen Nutzeräußerung aus der Datenbank (per retrieve_context-Service) und verkettet
        die gefundenen Einträge zu einem einzigen String, der dem LLM als Zusatzkontext übergeben wird. Leerer
        String, wenn kein Kontext gefunden wurde oder der Service nicht antwortet.
        """
        req = RetrieveContext.Request()
        req.guest_id = guest_id
        req.search_text = user_text
        result = self.call_service(self.db_context_client, req, timeout=5.0)
        if result and result.contexts:
            return " ".join(result.contexts)
        return ""

    def save_and_finish(self, person: PersonDetected, conversation: str, voice_emb: list) -> None:
        """
        Speichert das Gespräch als neuen oder aktualisierten Gast-Eintrag in der Datenbank.
        Extrahiert zuerst per RAG Memories und Episode aus dem Gesprächstext. Wurde die Person zu Beginn nicht
        erkannt, wird hier nochmal mit dem (meist längeren, besseren) Voice-Embedding aus dem Gespräch versucht sie
        zu erkennen, bevor entschieden wird, ob ein bestehender Gast aktualisiert (update_guest) oder ein neuer
        angelegt wird (create_guest).
        """
        memory_json = self.rag.extract_memory(conversation)
        episode_json = self.rag.extract_episode(conversation)

        # Wenn Person anfangs nicht erkannt wurde, mit anderem Embedding aus dem Gespräch nochmal versuchen
        if not person.found and voice_emb:
            req = FindByVoice.Request()
            req.voice_emb = voice_emb
            result = self.call_service(self.db_find_guest_client, req, timeout=3.0)
            if result and result.found:
                person.found = True
                person.guest_id = result.guest_id
                person.name = result.name
                self.get_logger().info(f"Person nachträglich erkannt: {result.name}")

        if person.found and person.guest_id:
            req = UpdateGuest.Request()
            req.guest_id = person.guest_id
            req.voice_emb = voice_emb
            req.name = person.name
            req.episode_json = episode_json
            req.memory_json = memory_json
            self.call_service(self.db_update_guest_client, req, timeout=5)
            self.get_logger().info(f"Gast aktualisiert: name={person.name!r}, guest_id={person.guest_id},\nmemory={memory_json},\nepisode={episode_json}")
        else:
            req = CreateGuest.Request()
            req.voice_emb = voice_emb
            req.name = person.name
            req.episode_json = episode_json
            req.memory_json = memory_json
            self.call_service(self.db_create_guest_client, req, timeout=5)
            self.get_logger().info(f"Neuer Gast gespeichert: name={person.name!r},\nmemory={memory_json},\nepisode={episode_json}")

        self.get_logger().info("Gespräch gespeichert! Warte auf nächste Person...")

    def speak(self, text: str) -> bool:
        """
        Lässt den Text per TTS-Service aussprechen.
        Gibt True bei Erfolg zurück, False wenn der TTS-Service fehlschlägt oder nicht antwortet. run_conversation
        bricht das Gespräch dann ab, da der Roboter nichts gesagt hat.
        """
        self.get_logger().info(f"TTS: {text}")
        req = TTS.Request()
        req.text = text
        # Fixer Timeout passt nicht, da die Sprechdauer von der Textlänge abhängt (LLM-Antworten sind unterschiedlich lang).
        # Schätzung nach durchschnittlichem Sprechtempo (120 Wörter/Minute (https://de.wikipedia.org/wiki/Sprechtempo)) mit
        # durchschnittlich 5,5 Zeichen/Wort (geschätzt): 120 * 5,5 / 60s = 11 Zeichen/Sekunde. Außerdem 10s Puffer.
        timeout = len(text) / 11 + 10
        result = self.call_service(self.speaker_client, req, timeout=timeout)
        if not result or not result.success:
            self.get_logger().error("TTS fehlgeschlagen. Gespräch wird abgebrochen.")
            return False
        time.sleep(0.4)  # Kurz warten damit Mikrofon nicht den eventuellen Nachhall von der Sprachausgabe vom Roboter aufnimmt. Wert ist experimentell, da es stark von der Umgebung abhängt.
        return True

    def call_service(self, client, request, timeout=15.0):
        """Ruft einen ROS2-Service asynchron auf und wartet blockierend bis zum Timeout auf die Antwort."""
        future = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.result() if future.done() else None


def main(args=None):
    rclpy.init(args=args)
    node = VoiceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
