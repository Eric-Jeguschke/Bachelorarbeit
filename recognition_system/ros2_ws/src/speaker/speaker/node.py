import subprocess
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from openai import OpenAI

from interfaces.srv import TTS

# https://git.mylab.th-luebeck.de/kia/tutorials/mylab-examples/-/blob/main/notebooks/Model_Overview.ipynb?ref_type=heads
class SpeakerNode(Node):
    def __init__(self):
        super().__init__("speaker") # Name von ROS Node

        self.client = OpenAI(base_url="https://models.mylab.th-luebeck.dev/v1", api_key="-")

        self.create_service(TTS, "~/tts", self.handle_speak)
        self.get_logger().info("Speaker node gestartet")

    def handle_speak(self, request, response):
        try:
            self.speak(request.text)
            response.success = True
        except Exception as error:
            self.get_logger().error(f"Sprachausgabe fehlgeschlagen: {error}")
            response.success = False
        return response

    def speak(self, text: str):
        self.get_logger().info("Warte auf antwort von TTS...")
        t0 = time.time()
        audio = self.client.audio.speech.create(
            model="xtts-v2",
            voice="alloy", # https://www.youtube.com/watch?v=dG2H7dHRopc und https://wiki.mylab.th-luebeck.de/books/mylab-ki-plattform/page/andere-modelle-image-audio-etc
            input=text,
        )
        self.get_logger().info(f"TTS antwort erhalten ({time.time() - t0:.1f}s). Spiele Audio...")
        t0 = time.time()
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
            input=audio.content,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.get_logger().info(f"Audio abgespielt ({time.time() - t0:.1f}s)")


def main(args=None):
    rclpy.init(args=args)
    node = SpeakerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
