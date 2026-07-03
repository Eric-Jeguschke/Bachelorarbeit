import io
import subprocess
import time
import wave
from pathlib import Path
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from piper import PiperVoice
from interfaces.srv import TTS

# https://huggingface.co/Thorsten-Voice/Piper/tree/main
MODEL_PATH = Path(__file__).resolve().parent / "models" / "de_DE-thorsten-medium.onnx"


class SpeakerNode(Node):
    """
    PiperVoice Nutzung orientiert sich an:
    https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
    """
    def __init__(self):
        super().__init__("speaker") # Name von ROS Node
        self.voice = PiperVoice.load(MODEL_PATH)
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
        self.get_logger().info("Erzeuge TTS-Audio über Piper...")
        t0 = time.time()
        buffer = io.BytesIO() # Puffer für WAV-Daten im Arbeitsspeicher
        with wave.open(buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        self.get_logger().info(f"TTS-Audio erzeugt ({time.time() - t0:.1f}s). Spiele Audio...")

        # getvalue() statt getbuffer(): relativ egal hier, da nicht ständig geschrieben wird (kein Random Access)
        # https://stackoverflow.com/questions/61319551/when-should-one-use-bytesio-getvalue-instead-of-getbuffer
        # ffplay hat knacken beim Abspielen der Audio verursacht, deswegen zu ffmpeg gewechselt.
        # -f wav -i -: WAV von stdin lesen (Bytes-Puffer statt Datei)
        # -af apad=pad_dur=0.8: Stille anhängen, sonst schneidet ffmpeg beim Beenden das letzte Wort ab
        # -f pulse default: Ausgabe direkt an PulseAudio. "default" ist der Stream-Name und Pflicht zu setzen.
        subprocess.run(
            ["ffmpeg", "-f", "wav", "-i", "-", "-af", "apad=pad_dur=0.8", "-f", "pulse", "default"],
            input=buffer.getvalue(),
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
