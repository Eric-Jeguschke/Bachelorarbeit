import time
import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from interfaces.msg import PersonDetected
from interfaces.srv import FindByVoice, Listen


class RecognitionNode(Node):
    def __init__(self):
        super().__init__("recognition")
        self.start_listening = True
        self.db_client  = self.create_client(FindByVoice, "/db/find_by_voice")
        self.mic_client = self.create_client(Listen, "/microphone/stt")
        self.person_detected = self.create_publisher(PersonDetected, "/recognition/person_detected", 1)
        self.create_subscription(String, "/llm/conversation_done", self.on_conversation_done, 1)

        # identify_loop ist kein ROS-Callback, daher startet MultiThreadedExecutor ihn nicht automatisch.
        # Manueller Thread nötig, damit spin() im Haupt-Thread weiterläuft und ROS-Callbacks (z.B. on_conversation_done) empfangen kann.
        # https://docs.ros.org/en/foxy/Concepts/About-Executors.html
        threading.Thread(target=self.identify_loop, daemon=True).start()  # identify_loop blockiert (Service-Calls), daher eigener Thread – spin() im Haupt-Thread bleibt so für ROS-Callbacks erreichbar
        self.get_logger().info("Recognition node gestartet...")

    def identify_loop(self):
        while rclpy.ok():
            if not self.start_listening:
                time.sleep(0.1)
                continue
            self.start_listening = False

            self.get_logger().info("Nehme Stimme auf...")
            rec = self.call_service(self.mic_client, Listen.Request(), timeout=40.0)
            if not rec or not rec.success or not rec.voice_emb:
                self.start_listening = True # Aufnahme ist fehlgeschlagen, versuche nochmal
                continue

            req = FindByVoice.Request()
            req.voice_emb = rec.voice_emb
            result = self.call_service(self.db_client, req, timeout=5.0)

            msg = PersonDetected()
            if result and result.found:
                msg.found = True
                msg.guest_id = result.guest_id
                msg.name = result.name
                msg.last_seen = result.last_seen
                self.get_logger().info(f"Bekannte Person gefunden: {result.name}")
            else:
                msg.found = False
                self.get_logger().info("Unbekannte Person")
            self.person_detected.publish(msg)

    def on_conversation_done(self, msg: String):
        self.start_listening = True
        self.get_logger().info(f"Gespräch beendet. Warte auf nächste Person...")

    def call_service(self, client, request, timeout: float):
        future = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.result() if future.done() else None

def main(args=None):
    rclpy.init(args=args)
    node = RecognitionNode()
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
