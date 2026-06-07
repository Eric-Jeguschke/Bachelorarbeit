import time
import numpy as np
import rclpy
import sounddevice as sd
import soundfile as sf

from openai import OpenAI
from rclpy.node import Node
from interfaces.srv import STT
from microphone.voice_embedder import VoiceEmbedder
from rclpy.executors import MultiThreadedExecutor

# https://python-sounddevice.readthedocs.io
class MicrophoneNode(Node):
    def __init__(self):
        super().__init__("microphone")
        self.client = OpenAI(base_url="https://models.mylab.th-luebeck.dev/v1", api_key="-")
        self.voice_embedder = VoiceEmbedder()
        self.voice_embedder.load()
        self.create_service(STT, "~/stt", self.handle_stt)
        self.get_logger().info("Microphone node gestartet")

    def handle_stt(self, request, response):
        try:
            if request.audio_path:
                audio, _ = sf.read(request.audio_path, dtype="float32") # der zweite Rückgabewert ist die Sample-Rate, die wir hier aber nicht brauchen, da wir davon ausgehen, dass alle Audios 16kHz haben
                file_path = request.audio_path
            else:
                frames, speech_detected = self.record_audio()
                if not speech_detected:
                    self.get_logger().info("Keine Sprache erkannt. Transkription übersprungen")
                    # success = True, weil es technisch ja geklappt hat, nur halt kein Audio da war.
                    # Und leeres Transkript + leeres Voice-Embedding zurückgeben, damit die anderen Nodes damit umgehen können.
                    response.success = True
                    response.transcript = ""
                    response.voice_emb = []
                    return response
                audio = np.concatenate(frames) # frames ist eine Liste von 1024-Sample-Blöcken. "concatenate" fügt sie zu einem einzigen Liste zusammen
                sf.write("recording.wav", audio, 16000)
                file_path = "recording.wav"

            response.transcript = self.transcribe(file_path)
            response.voice_emb = self.voice_embedder.embed_from_pcm(audio, 16000).tolist()
            response.success = True
        except Exception as error:
            self.get_logger().error(f"STT fehlgeschlagen: {error}")
            response.success = False
            response.voice_emb = []
        return response

    def transcribe(self, file_path: str) -> str:
        with open(file_path, "rb") as f:  # rb = read binary, nötig für Audio-Dateien
            # Andere Sprachen sollten auch erkannt werden, da wir mit "language" dem Modell eigentlich nur einen Tipp geben
            return self.client.audio.transcriptions.create(model="whisper-3-large", file=f, language="de").text

    # https://stackoverflow.com/questions/18406570/python-record-audio-on-detected-sound
    # https://python-sounddevice.readthedocs.io/en/0.3.15/examples.html
    def record_audio(self):
        """
        Das Mikrofon misst Luftdruckschwankungen. Schall ist eine Druckwelle die abwechselnd Überdruck und Unterdruck erzeugt. Überdruck → positive Werte, Unterdruck → negative Werte. Stille = Luftdruck
        konstant = 0.
        """
        frames = []
        started = False
        end = time.time() + 30

        with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=1024) as stream:
            while time.time() < end:
                data, _ = stream.read(1024)  # blockiert bis 1024 Samples bereit sind, dann eine Iteration
                if np.abs(data).mean() > 0.01: # abs = Betrag, ohne wäre der Mittelwert immer nahe 0, da positive und negative Werte sich ausgleichen würden. Schwellenwert bei float32 zwischen 0.0 und 1.0
                    if not started:
                        started = True
                        self.get_logger().info("Sprache erkannt. Aufnahme läuft...")
                    end = time.time() + 3 # Nach Ende der Sprache noch 3 Sekunden weiter aufnehmen, um sicherzugehen, dass am Ende nichts abgeschnitten wird
                    frames.append(data)
                elif started:
                    frames.append(data)

        return frames, started


def main(args=None):
    rclpy.init(args=args)
    node = MicrophoneNode()
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
