import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sentence_transformers import SentenceTransformer

from interfaces.srv import CreateGuest, UpdateGuest, FindByVoice, RetrieveContext

from db.guest_store import clean_old_entries, find_by_voice, create_guest, update_guest, retrieve_context


class GuestDBNode(Node):
    """ROS2-Node, der die LanceDB-Services (Suchen, Anlegen, Aktualisieren, Kontext abrufen) bereitstellt."""

    def __init__(self):
        super().__init__("db")
        # Text Embedding Model: https://aimultiple.com/multilingual-embedding-models, 384 Dimensionen
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-small")

        self.create_service(FindByVoice, "~/find_by_voice", self.handle_find)
        self.create_service(CreateGuest, "~/create_guest", self.handle_create)
        self.create_service(UpdateGuest, "~/update_guest", self.handle_update)
        self.create_service(RetrieveContext, "~/retrieve_context", self.handle_retrieve)
        self.create_timer(86400.0, clean_old_entries) # Einmal täglich alte Einträge bereinigen
        self.get_logger().info("GuestDB node gestartet")

    def embed_passage(self, text: str) -> list[float]:
        """
        Wandelt einen zu speichernden Text (Fakt, Keyword) in ein Embedding um. Wird als text_embedder an
        create_guest/update_guest übergeben. Das "passage: "-Prefix ist Teil des E5-Modell-Formats. e5-Modelle
        wurden mit unterschiedlichen Prefixen für gespeicherte Texte ("passage: ") und Suchanfragen ("query: ")
        trainiert, was die Retrieval-Qualität verbessern soll. Deshalb hier ein anderes Prefix als in embed_query.
        """
        return self.embedder.encode("passage: " + text, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Wandelt eine Suchanfrage in ein Embedding um, das beim RAG-Retrieval (retrieve_context) mit den per
        embed_passage gespeicherten Embeddings verglichen wird. Nutzt das "query: "-Prefix des E5-Modells
        (siehe embed_passage für Details).
        """
        return self.embedder.encode("query: " + text, normalize_embeddings=True).tolist()

    def handle_find(self, request, response):
        """Service-Handler: Sucht einen Gast per Stimm-Embedding"""
        try:
            result = find_by_voice(list(request.voice_emb))
        except Exception as e:
            self.get_logger().error(f"find_by_voice fehlgeschlagen: {e}")
            response.found = False
            return response
        if result:
            response.found = True
            response.guest_id = result.get("guest_id", "")
            response.name = result.get("name", "")
            response.last_seen = int(result.get("last_seen", 0))
        else:
            response.found = False
        return response

    def handle_create(self, request, response):
        """Service-Handler: Legt einen neuen Gast an"""
        guest_id = create_guest(
            name=request.name,
            voice_emb=list(request.voice_emb),
            episode_json=request.episode_json,
            memory_json=request.memory_json,
            text_embedder=self.embed_passage,
        )
        response.guest_id = guest_id
        if guest_id:
            response.success = True
        else:
            response.success = False
            self.get_logger().error("create_guest fehlgeschlagen")
        return response

    def handle_update(self, request, response):
        """Service-Handler: Aktualisiert einen bestehenden Gast"""
        guest_id = update_guest(
            guest_id=request.guest_id,
            name=request.name,
            voice_emb=list(request.voice_emb),
            episode_json=request.episode_json,
            memory_json=request.memory_json,
            text_embedder=self.embed_passage,
        )
        response.guest_id = guest_id
        if guest_id:
            response.success = True
        else:
            response.success = False
            self.get_logger().error(f"update_guest fehlgeschlagen für guest_id {request.guest_id}")
        return response

    def handle_retrieve(self, request, response):
        """Service-Handler: Liefert RAG-Kontext zu einem Gast"""
        response.contexts = retrieve_context(
            request.guest_id,
            self.embed_query(request.search_text),
        )
        self.get_logger().info(f"Aus DB geholter Kontext für guest_id {request.guest_id}: {response.contexts}")
        if not response.contexts:
            self.get_logger().warning(f"Kein Kontext gefunden für guest_id {request.guest_id}")
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GuestDBNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
