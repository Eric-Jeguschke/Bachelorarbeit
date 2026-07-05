bool found
string guest_id
string name          # Leer wenn neu oder "anonym" wenn Name nicht bekannt
int64 last_seen
string transcript    # Erstes Transkript aus Recognition
float32[] voice_emb  # Voice-Embedding aus Recognition (Wird genutzt, falls kein weiteres Gespräch stattfindet)
