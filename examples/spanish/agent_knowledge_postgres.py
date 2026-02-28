"""
Recuperación de conocimiento (RAG) con PostgreSQL y búsqueda híbrida (vector + texto completo).

Diagrama:

 Entrada ──▶ Agente ──────────────────▶ LLM ──▶ Respuesta
               │                        ▲
               │  buscar con entrada    │ conocimiento relevante
               ▼                        │
         ┌────────────┐                 │
         │   Base de   │────────────────┘
         │ conocimiento│
         │ (Postgres)  │
         └────────────┘

Este ejemplo usa pgvector para búsqueda por similitud vectorial y el
tsvector nativo de PostgreSQL para búsqueda de texto completo, combinándolos
con Reciprocal Rank Fusion (RRF) para recuperación híbrida. El agente busca
en la base de conocimiento *antes* de consultar al LLM — sin necesidad de
llamar a una herramienta.

Requisitos:
  - PostgreSQL con extensión pgvector (ver docker-compose.yml)
  - Un modelo de embeddings (GitHub Models, Azure OpenAI u OpenAI)

Ver también: agent_knowledge_sqlite.py para una versión más simple solo con SQLite (búsqueda por palabras clave).
"""

import asyncio
import logging
import os
import sys
from typing import Any

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector

from agent_framework import Agent, AgentSession, BaseContextProvider, Message, SessionContext, SupportsAgentRun
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
from azure.identity import get_bearer_token_provider as sync_get_bearer_token_provider
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from rich import print
from rich.logging import RichHandler

# ── Logging ──────────────────────────────────────────────────────────
handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Clientes OpenAI (chat + embeddings) ──────────────────────────────
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://admin:LocalPasswordOnly@db:5432/postgres")
EMBEDDING_DIMENSIONS = 256  # Dimensión reducida para eficiencia

async_credential = None
if API_HOST == "azure":
    # Credencial asíncrona para el cliente de chat del framework
    async_credential = DefaultAzureCredential()
    async_token_provider = get_bearer_token_provider(async_credential, "https://cognitiveservices.azure.com/.default")
    # Credencial síncrona para el cliente de embeddings (SDK de OpenAI)
    sync_credential = SyncDefaultAzureCredential()
    sync_token_provider = sync_get_bearer_token_provider(sync_credential, "https://cognitiveservices.azure.com/.default")
    chat_client = OpenAIChatClient(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT']}/openai/v1/",
        api_key=async_token_provider,
        model_id=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    )
    embed_client = OpenAI(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT']}/openai/v1/",
        api_key=sync_token_provider(),
    )
    embed_model = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
elif API_HOST == "github":
    chat_client = OpenAIChatClient(
        base_url="https://models.github.ai/inference",
        api_key=os.environ["GITHUB_TOKEN"],
        model_id=os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini"),
    )
    embed_client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=os.environ["GITHUB_TOKEN"],
    )
    embed_model = "text-embedding-3-small"
else:
    chat_client = OpenAIChatClient(
        api_key=os.environ["OPENAI_API_KEY"], model_id=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    )
    embed_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embed_model = "text-embedding-3-small"


def get_embedding(text: str) -> list[float]:
    """Obtiene un vector de embedding para el texto dado."""
    response = embed_client.embeddings.create(input=text, model=embed_model, dimensions=EMBEDDING_DIMENSIONS)
    return response.data[0].embedding


# ── Base de conocimiento (PostgreSQL + pgvector) ─────────────────────

PRODUCTS = [
    {
        "name": "Botas de Senderismo TrailBlaze",
        "category": "Calzado",
        "price": 149.99,
        "description": (
            "Botas de senderismo impermeables con suelas Vibram, soporte de tobillo "
            "y forro transpirable Gore-Tex. Ideales para senderos rocosos y condiciones húmedas."
        ),
    },
    {
        "name": "Mochila SummitPack 40L",
        "category": "Mochilas",
        "price": 89.95,
        "description": (
            "Mochila ligera de 40 litros con compartimento para hidratación, cubierta de lluvia "
            "y cinturón de cadera ergonómico. Perfecta para excursiones de un día o con pernocta."
        ),
    },
    {
        "name": "Chaqueta de Plumón ArcticShield",
        "category": "Ropa",
        "price": 199.00,
        "description": (
            "Chaqueta de plumón de ganso 800-fill con clasificación de -28°C. "
            "Incluye carcasa resistente al agua, diseño comprimible y capucha ajustable."
        ),
    },
    {
        "name": "Remo para Kayak RiverRun",
        "category": "Deportes Acuáticos",
        "price": 74.50,
        "description": (
            "Remo de fibra de vidrio para kayak con férula ajustable y anillos antigoteo. "
            "Ligero (795 g), apto para kayak recreativo y de travesía."
        ),
    },
    {
        "name": "Bastones de Trekking TerraFirm",
        "category": "Accesorios",
        "price": 59.99,
        "description": (
            "Bastones de trekking plegables de fibra de carbono con empuñaduras de corcho y puntas de tungsteno. "
            "Ajustables de 60 a 137 cm, con amortiguación anti-vibración."
        ),
    },
    {
        "name": "Binoculares ClearView 10x42",
        "category": "Óptica",
        "price": 129.00,
        "description": (
            "Binoculares de prisma de techo con aumento 10x y lentes objetivos de 42 mm. "
            "Cargados con nitrógeno y resistentes al agua. Ideales para observación de aves y fauna."
        ),
    },
    {
        "name": "Linterna Frontal LED NightGlow",
        "category": "Iluminación",
        "price": 34.99,
        "description": (
            "Linterna frontal recargable de 350 lúmenes con modo de luz roja y haz ajustable. "
            "Clasificación IPX6 de resistencia al agua, hasta 40 horas en modo bajo."
        ),
    },
    {
        "name": "Saco de Dormir CozyNest",
        "category": "Camping",
        "price": 109.00,
        "description": (
            "Saco de dormir tipo momia para tres estaciones, con clasificación de -6°C. "
            "Aislamiento sintético, saco de compresión incluido. Pesa 1.1 kg."
        ),
    },
]


def create_knowledge_db(conn: psycopg.Connection) -> None:
    """Crea el catálogo de productos en PostgreSQL con pgvector e índices de texto completo."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    conn.execute("DROP TABLE IF EXISTS products")
    conn.execute(
        f"""
        CREATE TABLE products (
            id          SERIAL PRIMARY KEY,
            name        TEXT NOT NULL,
            category    TEXT NOT NULL,
            price       REAL NOT NULL,
            description TEXT NOT NULL,
            embedding   vector({EMBEDDING_DIMENSIONS})
        )
        """
    )
    # Índice GIN para búsqueda de texto completo sobre nombre + descripción
    conn.execute(
        "CREATE INDEX ON products USING GIN (to_tsvector('spanish', name || ' ' || description))"
    )

    logger.info("[📚 Conocimiento] Generando embeddings para %d productos...", len(PRODUCTS))
    for product in PRODUCTS:
        text_for_embedding = f"{product['name']} - {product['category']}: {product['description']}"
        embedding = get_embedding(text_for_embedding)
        conn.execute(
            "INSERT INTO products (name, category, price, description, embedding) VALUES (%s, %s, %s, %s, %s)",
            (product["name"], product["category"], product["price"], product["description"], embedding),
        )

    conn.commit()
    logger.info("[📚 Conocimiento] Catálogo de productos cargado con embeddings.")


# ── Proveedor de contexto personalizado para recuperación híbrida ────

# SQL de búsqueda híbrida usando Reciprocal Rank Fusion (RRF)
# Combina resultados de similitud vectorial y búsqueda de texto completo
HYBRID_SEARCH_SQL = f"""
WITH semantic_search AS (
    SELECT id, RANK() OVER (ORDER BY embedding <=> %(embedding)s::vector({EMBEDDING_DIMENSIONS})) AS rank
    FROM products
    ORDER BY embedding <=> %(embedding)s::vector({EMBEDDING_DIMENSIONS})
    LIMIT 20
),
keyword_search AS (
    SELECT id, RANK() OVER (ORDER BY ts_rank_cd(to_tsvector('spanish', name || ' ' || description), query) DESC)
    FROM products, plainto_tsquery('spanish', %(query)s) query
    WHERE to_tsvector('spanish', name || ' ' || description) @@ query
    ORDER BY ts_rank_cd(to_tsvector('spanish', name || ' ' || description), query) DESC
    LIMIT 20
)
SELECT
    COALESCE(semantic_search.id, keyword_search.id) AS id,
    COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
    COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score
FROM semantic_search
FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
ORDER BY score DESC
LIMIT %(limit)s
"""


class PostgresKnowledgeProvider(BaseContextProvider):
    """Recupera conocimiento relevante mediante búsqueda híbrida (vector + texto completo) con RRF.

    Usa pgvector para similitud semántica y tsvector de PostgreSQL para
    coincidencia por palabras clave, combinando resultados con Reciprocal
    Rank Fusion (RRF). Esto da mejor recuperación que cualquier método solo.
    """

    def __init__(self, conn: psycopg.Connection, max_results: int = 3):
        super().__init__(source_id="postgres-knowledge")
        self.conn = conn
        self.max_results = max_results

    def _search(self, query: str) -> list[dict]:
        """Ejecuta búsqueda híbrida (vector + texto completo) y devuelve productos coincidentes."""
        query_embedding = get_embedding(query)

        cursor = self.conn.execute(
            HYBRID_SEARCH_SQL,
            {"embedding": query_embedding, "query": query, "k": 60, "limit": self.max_results},
        )
        result_ids = [row[0] for row in cursor.fetchall()]
        if not result_ids:
            return []

        # Obtener detalles completos de los productos encontrados
        products = []
        for product_id in result_ids:
            row = self.conn.execute(
                "SELECT name, category, price, description FROM products WHERE id = %s",
                (product_id,),
            ).fetchone()
            if row:
                products.append({"name": row[0], "category": row[1], "price": row[2], "description": row[3]})
        return products

    def _format_results(self, results: list[dict]) -> str:
        """Formatea los resultados de búsqueda como texto para el contexto del LLM."""
        lines = ["Información relevante de productos de nuestro catálogo:\n"]
        for product in results:
            lines.append(
                f"- **{product['name']}** ({product['category']}, ${product['price']:.2f}): "
                f"{product['description']}"
            )
        return "\n".join(lines)

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Busca en la base de conocimiento con el último mensaje del usuario e inyecta resultados."""
        user_text = next((msg.text for msg in reversed(context.input_messages) if msg.role == "user" and msg.text), None)
        if not user_text:
            return

        results = self._search(user_text)
        if not results:
            logger.info("[📚 Conocimiento] No se encontraron productos para: %s", user_text)
            return

        logger.info("[📚 Conocimiento] Se encontraron %d producto(s) para: %s", len(results), user_text)

        context.extend_messages(
            self.source_id,
            [Message(role="user", text=self._format_results(results))],
        )


# ── Configuración ───────────────────────────────────────────────────


def setup_db() -> psycopg.Connection:
    """Conecta a PostgreSQL y carga la base de conocimiento."""
    conn = psycopg.connect(POSTGRES_URL)
    create_knowledge_db(conn)
    return conn


conn = setup_db()
knowledge_provider = PostgresKnowledgeProvider(conn=conn)

agent = Agent(
    client=chat_client,
    instructions=(
        "Eres un asistente de compras de equipo para actividades al aire libre de la tienda 'TrailBuddy'. "
        "Responde las preguntas del cliente usando SOLO la información de productos proporcionada en el contexto. "
        "Si no se encuentran productos relevantes en el contexto, di que no tienes información sobre ese artículo. "
        "Incluye precios al recomendar productos."
    ),
    context_providers=[knowledge_provider],
)


async def main() -> None:
    """Demuestra búsqueda híbrida RAG con varias consultas."""
    print("\n[bold]=== Recuperación de Conocimiento (RAG) con Búsqueda Híbrida en PostgreSQL ===[/bold]")

    # Consulta 1: Debería encontrar botas de senderismo y bastones de trekking
    print("[blue]Usuario:[/blue] Estoy planeando una excursión. ¿Qué botas y bastones me recomiendan?")
    response = await agent.run("Estoy planeando una excursión. ¿Qué botas y bastones me recomiendan?")
    print(f"[green]Agente:[/green] {response.text}\n")

    # Consulta 2: Coincidencia semántica — "artículos para observar fauna" → binoculares
    print("[blue]Usuario:[/blue] Quiero artículos para observar fauna silvestre")
    response = await agent.run("Quiero artículos para observar fauna silvestre")
    print(f"[green]Agente:[/green] {response.text}\n")

    conn.close()

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[agent], auto_open=True)
    else:
        asyncio.run(main())
