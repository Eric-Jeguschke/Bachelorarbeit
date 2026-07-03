import math
import subprocess
import time
import numpy as np
import rclpy
import sounddevice as sd
import soundfile as sf

from openai import OpenAI
from rclpy.node import Node
from interfaces.srv import Listen
from microphone.voice_embedder import VoiceEmbedder
from rclpy.executors import MultiThreadedExecutor

class MicrophoneNode(Node):
    def __init__(self):
        super().__init__("microphone")
        self.client = OpenAI(base_url="https://models.mylab.th-luebeck.dev/v1", api_key="-")
        self.warmup_transcription()
        self.voice_embedder = VoiceEmbedder()
        self.voice_embedder.load()
        self.create_service(Listen, "~/stt", self.handle_stt)
        self.get_logger().info("Microphone node gestartet")

    def warmup_transcription(self) -> None:
        """
        Erste Anfrage an das gehostete Whisper-Modell ist oft deutlich langsamer.
        Deshalb einmal beim Start mit stillem Audio transkribieren, damit die erste echte Aufnahme schnell verarbeitet wird.
        :return:
        """
        try:
            t0 = time.time()
            warmup_path = "/tmp/warmup.wav"
            sf.write(warmup_path, np.zeros(16000, dtype=np.float32), 16000)
            with open(warmup_path, "rb") as f:
                self.client.audio.transcriptions.create(model="whisper-3-large", file=f, language="de")
            self.get_logger().info(f"Transkriptions-Warmup abgeschlossen ({time.time() - t0:.2f}s)")
        except Exception as error:
            self.get_logger().error(f"Transkriptions-Warmup fehlgeschlagen: {error}")

    def handle_stt(self, request, response):
        try:
            if request.audio_path:
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
                sf.write("/tmp/recording.wav", audio, 16000)
                file_path = "/tmp/recording.wav"

            trimmed_path = self.trim_silence(file_path)
            duration = sf.info(trimmed_path).duration
            if duration < 0.4:  # Kürzer als 0,4s = kein echter Satz
                self.get_logger().info(f"Audio nach Stille-Trim zu kurz ({duration:.2f}s). Übersprungen.")
                response.success = True
                response.transcript = ""
                response.voice_emb = []
                return response

            self.get_logger().info("Transkribiere...")
            t0 = time.time()
            response.transcript = self.transcribe(trimmed_path)
            if not response.transcript:
                # Kein Embedding berechnen: ein Stille-Vektor würde die DB verfälschen
                response.success = True
                response.voice_emb = []
                return response
            self.get_logger().info(f"Transkription: {response.transcript} ({time.time() - t0:.2f}s). Berechne Voice-Embedding...")
            trimmed_audio, _ = sf.read(trimmed_path, dtype="float32") # der zweite Rückgabewert ist die Sample-Rate, die wir hier aber nicht brauchen, da wir davon ausgehen, dass alle Audios 16kHz haben
            response.voice_emb = self.voice_embedder.embed_from_pcm(trimmed_audio, 16000).tolist()
            self.get_logger().info("Voice-Embedding berechnet.")
            response.success = True
        except Exception as error:
            self.get_logger().error(f"Listen fehlgeschlagen: {error}")
            response.success = False
            response.voice_emb = []
        return response

    def trim_silence(self, file_path: str) -> str:
        trimmed_path = "/tmp/recording_trimmed.wav"
        # Stille am Anfang und Ende entfernen, bevor Whisper und das Voice-Embedding sie verarbeiten.
        # Verhindert Whisper-Halluzinationen (z.B. "Vielen Dank.") und verfälschte Embeddings bei stillen Aufnahmen.
        # https://stackoverflow.com/questions/25697596/using-ffmpeg-with-silencedetect-to-remove-audio-silence
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", file_path,
            # 1. silenceremove: Stille am Anfang entfernen. 2. areverse: Audio umkehren, damit das Ende vorne liegt.
            # 3. silenceremove: Stille am  Ende entfernen(liegt jetzt vorne). 4. areverse: Audio zurückdrehen.
            # start_periods=1: Filter entfernt nur den ersten Stille-Abschnitt am Anfang und stoppt danach – Pausen in der Mitte werden nie erreicht.
            # start_threshold= -40dB, alles leisere wird als keine Audio wahrgenommen
            # start_silence=0.15: 150ms Puffer vor dem ersten Laut lassen, damit Anlaute nicht abgeschnitten werden.
            # detection=peak: Lautstärke wird als Spitzenwert gemessen, nicht als Durchschnitt.
            "-af", ("silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.15:detection=peak,"
                    "areverse,silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.15:detection=peak,areverse"),
            trimmed_path
        ], check=True)  # check=True: überprüft Status-Code und wirft Exception bei Fehler, wird von handle_stt try/except aufgefangen
        return trimmed_path

    def transcribe(self, trimmed_path: str) -> str:
        with open(trimmed_path, "rb") as f:
            # Andere Sprachen sollten auch erkannt werden, da wir mit "language" dem Modell nur einen Tipp geben
            text = self.client.audio.transcriptions.create(model="whisper-3-large", file=f, language="de").text
        # Whisper halluziniert bei stillem Audio häufig "Vielen Dank."
        # Kommt "Vielen Dank" dagegen in einem längeren, echten Satz vor (z.B. "Vielen Dank, das war hilfreich"), ist es wahrscheinlich keine Halluzination.
        if "Vielen Dank" in text and len(text) < 15:
            self.get_logger().info(f"Wahrscheinlich Whisper-Halluzination erkannt und verworfen: {text}")
            return ""
        return text

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
        log_counter = 0

        with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=1024) as stream:
            while time.time() < end:
                data, _ = stream.read(1024)  # blockiert bis 1024 Samples bereit sind, dann eine Iteration
                sound_level = np.abs(data).mean()
                # https://dsp.stackexchange.com/questions/8785/how-to-compute-dbfs
                dbfs = 20 * math.log10(sound_level)  # dBFS: Skala mit 0 als Obergrenze , Stille geht gegen minus unendlich
                log_counter += 1
                if log_counter % 5 == 0:  # nur jeden 5ten Block loggen, sonst zu viel Output
                    self.get_logger().info(f"Pegel: {dbfs:.1f} dBFS")
                    log_counter = 0
                if sound_level > 0.015:  # ungefähr -36,5 dBFS (20 * log10(0.015) = -36,5 dBFS). Alles unterhalb von -36,5 dBFS wird als Stille betrachtet.
                    if not started:
                        started = True
                        self.get_logger().info("Sprache erkannt. Aufnahme läuft...")
                    end = time.time() + 2.0 # Nach Ende der Sprache noch 1,5 Sekunden weiter aufnehmen
                    frames.append(data)
                elif started:
                    frames.append(data)

        if started:
            self.get_logger().info("Aufnahme beendet.")
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
        rclpy.shutdown()
