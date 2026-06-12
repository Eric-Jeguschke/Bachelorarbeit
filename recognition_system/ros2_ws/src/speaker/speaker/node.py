import subprocess

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
        speech_file_path = "/tmp/speech.mp3"
        with self.client.audio.speech.with_streaming_response.create(
            model="xtts-v2",
            voice="fable",
            speed=1.1,
            input=text,
        ) as response:
            response.stream_to_file(speech_file_path)
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", speech_file_path])


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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

