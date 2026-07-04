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
        # ffplay hat knacken beim Abspielen der Audio verursacht, deswegen zu ffmpeg gewechselt. Hat auch noch knacken, aber weniger.
        # -f wav -i -: WAV von stdin lesen
        # -af apad=pad_dur=0.7: 0.7s Stille anhängen, damit der Puffer sicher leerläuft, bevor ffmpeg endet.
        # -f pulse : Ausgabe direkt an PulseAudio.
        # -buffer_duration 500: fester Puffer (500ms) statt PulseAudio selbst entscheiden zu lassen.
        #   pad_dur (0.7s) ist bewusst größer als buffer_duration (500ms), damit Ende der Audio auch sauber zu Ende geht.
        # -default: Ist der Stream-Name und Pflichtparameter bei -f pulse.
        subprocess.run(
            [
                "ffmpeg", "-f", "wav", "-i", "-", "-af", "apad=pad_dur=0.7",
                "-f", "pulse", "-buffer_duration", "500", "default"
            ],
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
