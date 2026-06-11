import threading
import time
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from interfaces.msg import PersonDetected
from interfaces.srv import CreateGuest, UpdateGuest, Listen, RetrieveContext, TTS
from llm.llm_chat import LLMChatModule
from llm.llm_rag import RAGModule


class VoiceNode(Node):
    def __init__(self):
        super().__init__("voice_ai")
        self.llm = LLMChatModule(self.get_logger())
        self.rag = RAGModule(self.get_logger())

        self.mic_client = self.create_client(Listen, "/microphone/stt")
        self.speaker_client = self.create_client(TTS, "/speaker/tts")
        self.db_create_guest_client = self.create_client(CreateGuest, "/db/create_guest")
        self.db_update_guest_client = self.create_client(UpdateGuest, "/db/update_guest")
        self.db_context_client = self.create_client(RetrieveContext, "/db/retrieve_context")
        self.conv_done_pub = self.create_publisher(String, "/llm/conversation_done", 1)  # 1 = Queue-Größe
        self.create_subscription(PersonDetected, "/recognition/person_detected", self.person_detected, 1)
        self.conversation_running = False
        self.get_logger().info("Voice AI node gestartet")

    def person_detected(self, person: PersonDetected):
        if self.conversation_running:
            return
        self.conversation_running = True
        # Eigener Thread nötig: run_conversation blockiert mit time.sleep() in call_service.
        # Läuft run_conversation direkt im Callback, kann der Executor die Service-Response nicht verarbeiten → Deadlock.
        threading.Thread(target=self.run_conversation, args=(person,), daemon=True).start()

    def run_conversation(self, person: PersonDetected):
        self.llm.init_with_context(person.name, person.last_seen)
        conversation_text = ""
        last_voice_emb = []

        while True:
            audio = self.record()
            if not audio:
                self.get_logger().error("Keine Audio-Daten erhalten")
                break

            if audio["voice_emb"]:
                last_voice_emb = audio["voice_emb"]

            if not audio["transcript"]:
                self.speak("Entschuldigung, ich habe dich leider nicht verstanden. Kannst du das wiederholen?")
                audio = self.record()
                if not audio or not audio["transcript"]:
                    break
                if audio["voice_emb"]:
                    last_voice_emb = audio["voice_emb"]

            spoken_text = audio["transcript"]
            context = self.fetch_context(person.guest_id, spoken_text) if (person.found and person.guest_id) else ""
            user_message = spoken_text if not conversation_text else f"{spoken_text}\nZuvor gesprochener Text: {conversation_text}"
            reply, done = self.llm.chat(user_message, context)
            self.speak(reply)
            conversation_text += f"Nutzer: {spoken_text}\nRoboter: {reply}\n"
            if len(conversation_text) > 30000:  # Prompt nicht zu lang werden lassen
                conversation_text = conversation_text[-30000:]
            if done:
                break

        self.save_and_finish(person, conversation_text, last_voice_emb)

    def record(self) -> dict | None:
        result = self.call_service(self.mic_client, Listen.Request(), timeout=60.0) # Länger warten, da Person vielleicht lange spricht
        if result and result.success:
            return {
                "transcript": result.transcript,
                "voice_emb": list(result.voice_emb),
            }
        return None

    def fetch_context(self, guest_id: str, user_text: str) -> str:
        req = RetrieveContext.Request()
        req.guest_id = guest_id
        req.search_text = user_text
        result = self.call_service(self.db_context_client, req, timeout=5.0)
        if result and result.contexts:
            return " ".join(result.contexts)
        return ""

    def save_and_finish(self, person: PersonDetected, conversation: str, voice_emb: list):
        memory_json = self.rag.extract_memory(conversation)
        episode_json = self.rag.extract_episode(conversation)

        if person.found and person.guest_id:
            req = UpdateGuest.Request()
            req.guest_id = person.guest_id
            req.voice_emb = voice_emb
            req.name = person.name
            req.episode_json = episode_json
            req.memory_json = memory_json
            self.call_service(self.db_update_guest_client, req)
        else:
            req = CreateGuest.Request()
            req.voice_emb = voice_emb
            req.name = person.name
            req.episode_json = episode_json
            req.memory_json = memory_json
            self.call_service(self.db_create_guest_client, req)

        self.conversation_running = False
        self.conv_done_pub.publish(String())  # data nicht gesetzt → data="" (leer)
        self.get_logger().info("Gespräch gespeichert! Warte auf nächste Person...")

    def speak(self, text: str):
        req = TTS.Request()
        req.text = text
        self.call_service(self.speaker_client, req)

    def call_service(self, client, request, timeout=15.0):
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
