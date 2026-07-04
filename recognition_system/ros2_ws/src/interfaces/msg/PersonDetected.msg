bool found
string guest_id      # Eindeutige ID der Person (leer wenn neu)
string name          # Leer wenn neu oder "anonym" wenn Name nicht bekannt
int64 last_seen      # Timestamp des letzten Besuchs (0 wenn neu)
string transcript    # Erstes Transkript aus Recognition - VoiceNode nutzt es als erste Gesprächsrunde
float32[] voice_emb  # Voice-Embedding aus Recognition - Fallback falls kein weiteres Gespräch stattfindet
