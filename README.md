# Wiedererkennungssystem - Bachelorarbeit

Sprachbasiertes Erkennungssystem für den PIB-Roboter. Identifiziert Personen anhand der Stimme, führt ein LLM-basiertes Gespräch auf Deutsch und speichert Gesprächserinnerungen in einer Vektordatenbank.

Läuft vollständig auf CPU, kein CUDA erforderlich.

### Nodes

- **recognition/node.py** - Steuert den Erkennungsablauf: ruft **microphone/stt** auf, fragt die DB per Stimm-Embedding ab und publiziert das Ergebnis auf **/recognition/person_detected**. Wartet während eines laufenden Gesprächs auf **/voice_ai/conversation_done**.  
- **db/node.py** - LanceDB-Vektordatenbank, sucht Personen per Stimm-Embedding mittels Cosine Distance und speichert alles zu Personen und vorherigen Gesprächen. Text-Embeddings für RAG mit **intfloat/multilingual-e5-small** (384-dim).  
- **microphone/node.py** - Nimmt Audio über PulseAudio auf, transkribiert mit **whisper-3-large** von mylab und berechnet das Stimm-Embedding mit **pyannote/wespeaker-voxceleb-resnet34-LM** (256-dim).  
- **voice_ai/voice_node.py** - Führt nach erkannter Person ein Gespräch auf Deutsch mit einem LLM von mylab. Ruft bei bekannten Personen gespeicherten Kontext mit RAG ab. Speichert am Ende Gesprächsprotokoll, Erinnerungen und aktualisiertes Stimm-Embedding (EMA, 60/40-Gewichtung).  
- **speaker/node.py** - Text-to-Speech lokal auf der CPU mit **Piper** (de_DE-thorsten-medium). Audioausgabe läuft über ffmpeg.

### Voraussetzungen

- Ubuntu-Terminal 22.04 in WSL2
- Docker

## Auf PIB-Roboter System starten

Roboter am LAN-Kabel anschließen und IP-Adresse am Display ablesen. Dann von Windows-Terminal:
```powershell
ssh pib@<IP-ADRESSE>
```
Passwort: `pib`

Repository klonen (einmalig:
```bash
git clone git@git.mylab.th-luebeck.de:eric.jeguschke/bachelorarbeit.git # kann sein, dass es auch über https geklont werden muss 
cd bachelorarbeit/recognition_system
```

System starten:
```bash
docker compose up --build
```

# Ubuntu-Terminal unter Windows (WSL2) System starten

Der Standard-WSL2-Kernel hat keinen USB-Audio-Treiber, direktes USB durchreichen funktioniert also nicht. Stattdessen nutzt WSLg einen PulseAudio-Server, der das Windows-Standard-Audiogerät durchreicht.
Headset in Windows als Standard-Audiogerät setzen (Eingabe + Ausgabe). Danach:

```bash
cd "/mnt/c/Users/ericj/OneDrive/Dokumente/Studium/Semester 6/Bachelorarbeit/Repository/recognition_system"
```
```bash
docker build -t ros2_recognition_system . && docker stop ros2_recognition_system && docker rm ros2_recognition_system && docker run -it --name ros2_recognition_system \
  -v hf_cache:/root/.cache/huggingface \
  -v $HOME/robot-db:/db/storage \
  -e PULSE_SERVER=unix:/mnt/wslg/runtime-dir/pulse/native \
  -v /mnt/wslg/runtime-dir/pulse:/mnt/wslg/runtime-dir/pulse \
  -v "/mnt/c/Studium/Semester 6/Bachelorarbeit/Datensatz/voxceleb":/data/voxceleb \
  --user $(id -u):$(id -g) \
  ros2_recognition_system
```

## Evaluation (Sprechererkennungssystem)

Benötigt den VoxCeleb1+2-Datensatz unter **/data/voxceleb** im Container (siehe Mount im WSL2-Docker-run-Befehl oben). Funktioniert deshalb nur, wenn der Container über WSL2 gestartet wurde und der Datensatz lokal verfügbar ist. Nicht auf dem PIB-Roboter, da dort weder der Datensatz noch der Mount vorhanden sind.

In neuem Terminal Container:
```bash
docker exec ros2_recognition_system bash -c "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash && ros2 run evaluation run_evaluation"
```

## Datenbank anzeigen

Während Container läuft in neuem Terminal:

```bash
docker exec ros2_recognition_system python3 /ros2_ws/src/db/db/show_db.py
```

# Framework-Vergleich der Sprecher-Embedding-Modelle (benchmark_voice_embedding)

Separater Benchmark (außerhalb des ROS2-Systems), das NeMo-, Pyannote- und SpeechBrain-Modelle auf den VoxCeleb1-Verifikations-Trials vergleicht (Laufzeit, RAM/VRAM, EER, minDCF, TAR@FAR).
Das Durchlaufen dauert recht lange. Bei mir hat es ungefähr 3 Stunden gedauert. 

Voraussetzungen:
- VoxCeleb1-Datensatz lokal verfügbar, Pfad in **benchmark/load_dataset.py** gegebenenfalls anpassen
- **.env**-Datei mit **HUGGINGFACE_TOKEN=...** im Root nötig für Pyannote Modelle
- benchmark_voice_embedding/requirements.txt installieren. Bin mir nicht sicher, ob das alles funktioniert, da ich die Stück für Stück installiert habe und es bei einigen Sachen auch Probleme gab. Ich habe die Versionen, die bei mir funktioniert haben, in requirements.txt festgelegt.

Start:
```bash
cd benchmark_voice_embedding
pip install -r requirements.txt
python run_benchmark.py
```

Ergebnisse landen in `results/summary.json`.
