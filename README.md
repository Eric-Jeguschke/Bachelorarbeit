# Erkennungssystem – Bachelorarbeit

Sprachbasiertes Erkennungssystem für den PIB-Roboter. Identifiziert Personen anhand der Stimme, führt ein LLM-gestütztes Gespräch auf Deutsch und speichert Gesprächserinnerungen in einer Vektordatenbank.

Läuft vollständig auf CPU, kein CUDA erforderlich.

### Nodes

- `recognition/node.py` – nimmt kontinuierlich Stimme auf, berechnet ein Stimm-Embedding und fragt die DB ab, publiziert die erkannte Person.
- `db/node.py` – LanceDB-Vektordatenbank, sucht Personen per Stimm-Embedding (Cosine Distance, Threshold 0.4), verwaltet Gesprächsepisoden und Erinnerungen. 
- `microphone/node.py` – nimmt Audio über ALSA auf (VAD via Lautstärkeschwelle), transkribiert mit Whisper (OpenAI API) und berechnet das Stimm-Embedding mit `pyannote/wespeaker-voxceleb-resnet34-LM` (256-dim, CPU). 
- `llm/voice_node.py` – führt nach erkannter Person ein Gespräch auf Deutsch (LLM: `chat-large`), begrüßt bekannte Personen mit Namen und zeigt an, wie lange sie zuletzt gesehen wurden. Speichert am Ende Gesprächsprotokoll, Erinnerungen und aktualisiertes Stimm-Embedding (EMA).
- `speaker/node.py` – Text-to-Speech mit OpenAI TTS API + ffplay.

## Voraussetzungen

- Ubuntu-Terminal 22.04 in WSL2
- Docker

## System starten

Produktiv läuft das System auf dem PIB-Roboter über `docker-compose`, für die Entwicklung unter Windows im Ubuntu-Terminal (WSL2). Beide nutzen PulseAudio für Mikrofon/Lautsprecher, nur die Mounts dorthin unterscheiden sich.

### PIB-Roboter

Roboter am LAN-Kabel anschließen und IP-Adresse am Display ablesen. Dann von Windows-Terminal:

```powershell
ssh pib@<IP-ADRESSE>
```

Passwort: `pib` (Standard-Login des PIB-Roboters)

Repository klonen (einmalig):

```bash
git clone git@git.mylab.th-luebeck.de:eric.jeguschke/bachelorarbeit.git
cd <dein-repo>/recognition_system
```

System starten:

```bash
docker compose up --build
```

Im Hintergrund laufen lassen bzw. stoppen:

```bash
docker compose up -d --build
docker compose down
```

### Ubuntu-Terminal unter Windows (WSL2)

Der Standard-WSL2-Kernel hat keinen USB-Audio-Treiber, direktes USB-Passthrough funktioniert also nicht. Stattdessen nutzt WSLg einen automatisch installierten PulseAudio-Server, der das Windows-Standard-Audiogerät per RDP-Protokoll durchreicht.

Headset in Windows als Standard-Audiogerät setzen (Eingabe + Ausgabe), dann prüfen ob PulseAudio verfügbar ist. Danach:

```bash
cd "/mnt/c/Users/ericj/OneDrive/Dokumente/Studium/Semester 6/Bachelorarbeit/Repository/recognition_system"
docker build -t ros2_ws_image .

docker stop ros2_system; docker rm ros2_system; docker run -it --name ros2_system \
  -v hf_cache:/root/.cache/huggingface \
  -v $HOME/robot-db:/db/storage \
  -e PULSE_SERVER=unix:/mnt/wslg/runtime-dir/pulse/native \
  -v /mnt/wslg/runtime-dir/pulse:/mnt/wslg/runtime-dir/pulse \
  -v "/mnt/c/Studium/Semester 6/Bachelorarbeit/Datensatz/voxceleb":/data/voxceleb \
  --user $(id -u):$(id -g) \
  ros2_ws_image
```

Modelle (~1–2 GB) werden beim ersten Start heruntergeladen und im Volume gecacht.

## Benchmark (Sprechererkennung)

In neuem Terminal Container:
```bash
docker exec ros2_system bash -c "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash && ros2 run benchmark run_benchmark"

```

## Datenbank anzeigen

Während der Container läuft:

```bash
docker exec ros2_system python3 /ros2_ws/src/db/db/show_db.py
```
