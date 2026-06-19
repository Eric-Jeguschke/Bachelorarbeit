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
        # https://github.com/remsky/Kokoro-FastAPI/wiki/HowTo-Opus
        self.get_logger().info("Spiele Audio...")
        ffplay = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"], # "-" → Audio-Daten von stdin lesen
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        t0 = time.time()
        with self.client.audio.speech.with_streaming_response.create(
            model="tts-1-hd",
            voice="coral",
            speed=1.1,
            response_format="opus", # opus ist ein effizienter Codec für Streaming-Audio, der niedrige Latenz und gute Qualität bietet
            input=text,
        ) as response:
            first_chunk = True
            for chunk in response.iter_bytes(chunk_size=1024):
                if first_chunk:
                    self.get_logger().info(f"Audio-Stream gestartet nach {time.time() - t0:.1f}s")
                    first_chunk = False
                ffplay.stdin.write(chunk)
                ffplay.stdin.flush()
            ffplay.stdin.close()
            ffplay.wait()
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
