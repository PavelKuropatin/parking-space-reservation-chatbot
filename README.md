# Parking Space Reservation Chatbot

A conversational AI assistant for parking space reservations, built with LangGraph. It handles natural-language Q&A about parking facilities and guides users through the full reservation flow — including human-in-the-loop admin approval — backed by a vector store (Weaviate) for static knowledge and a relational database (PostgreSQL) for live availability, booking data, and conversation checkpoints.

## Architecture

The agent is implemented as a LangGraph state machine with the following stages:

```
User input
  └─► Input guardrail (Presidio PII scan)
        ├─► [blocked] → rejection message → Output guardrail
        └─► [ok] → Intent classifier
                    ├─► information_request → RAG (Weaviate) + DB tools ────────┐
                    └─► reservation        → detail extraction → confirmation   │
                                                 └─► [confirmed] → save reservation (Postgres)
                                                        └─► interrupt: request_admin_approval
                                                               └─► admin approves/rejects (admin.py, out-of-band)
                                                                      └─► update reservation status ─┘
                                                                                                       │
                                                                                                       ▼
                                                                                            Output guardrail (Presidio PII scan)
```

Reservations pause the graph mid-run via LangGraph's `interrupt()`. A PostgreSQL-backed checkpointer persists the paused state so a separate `admin.py` process can review the request and resume the customer's thread once a decision is made. The two processes exchange requests/decisions through a simple filesystem-based JSON mailbox (`chatbot/notifier.py`).

![LangGraph](langgraph.png)

## Tech stack

| Component | Technology |
|-----------|-----------|
| Agent framework | LangGraph |
| LLM | OpenAI-compatible endpoint (configurable) |
| Vector store | Weaviate 1.38 |
| Vectorizer | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Relational DB | PostgreSQL 17 |
| Conversation checkpointing | PostgreSQL (`langgraph-checkpoint-postgres`) |
| Admin notification transport | Filesystem JSON mailbox (`chatbot/notifier.py`) |
| PII guardrail | Microsoft Presidio + spaCy `en_core_web_lg` |

## Project structure

```
.
├── client.py                            # Customer-facing chatbot CLI entry point
├── admin.py                             # Admin CLI — approves/rejects pending reservations
├── assets/
│   ├── static/                          # Markdown knowledge base (ingested into Weaviate)
│   └── dynamic/                         # SQL seed files (loaded into PostgreSQL on first start)
├── chatbot/
│   ├── database/
│   │   ├── retriever.py                 # Weaviate hybrid-search retriever
│   │   └── sql_store.py                 # PostgreSQL access (pricing, hours, availability, reservations)
│   ├── guardrail/
│   │   └── filtering.py                 # Presidio-based PII filtering
│   ├── scripts/
│   │   ├── ingest_static_data.py        # Load markdown assets into Weaviate
│   │   ├── init_checkpoiter.py          # Create the LangGraph checkpointer tables in PostgreSQL
│   │   └── run_evaluation.py            # RAG + LLM quality evaluation
│   ├── utils/                           # Shared utilities and evaluation helpers
│   ├── graph.py                         # LangGraph state graph definition
│   ├── nodes.py                         # LangGraph node implementations
│   ├── notifier.py                      # Filesystem-based request/response mailbox (client ↔ admin)
│   ├── prompts.py                       # Prompt templates
│   ├── states.py                        # Graph state, enums, structured-output schemas
│   ├── settings.py                      # Pydantic settings (reads .env)
│   └── logging.py                       # App-wide logger (writes to app.log)
├── docker-compose.yaml
└── pyproject.toml
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package and venv manager)
- Python 3.13+

## Setup

### 1. Start infrastructure services

```bash
docker compose up -d
```

This starts five services:

| Service | Description | Port |
|---------|-------------|------|
| `weaviate` | Vector database | 8080 (HTTP), 50051 (gRPC) |
| `t2v-transformers` | Text-to-vector inference (`multi-qa-MiniLM-L6-cos-v1`) | internal |
| `reranker-transformers` | Reranker inference (`ms-marco-MiniLM-L-6-v2`) | internal |
| `postgres` | Relational database (seeded from `assets/dynamic/`) — hosts both the app tables and, by default, the LangGraph checkpointer tables | 5432 |
| `mcp-server` | MCP server (`chatbot/mcp/mcp_server.py`) exposing the `submit_reservation` tool used by the graph | 8088 |

### 2. Create virtual environment and install dependencies

```bash
uv venv
uv sync
```

### 3. Download the spaCy model

The PII guardrail (Presidio) requires the `en_core_web_lg` spaCy model:

```bash
uv run python -m spacy download en_core_web_lg
```

### 4. Configure environment variables

Copy `.env.template` to `.env` in the project root and fill in your values:

```dotenv
# LLM (OpenAI-compatible endpoint — OpenAI, LM Studio, etc.)
OPENAI_LLM_URL=http://localhost:1234/v1
OPENAI_LLM_API_KEY=your-api-key
OPENAI_LLM_MODEL=openai/gpt-oss-20b

# Weaviate
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_COLLECTION=parking_info
WEAVIATE_INIT_DATA_PATH=assets/static

# RAG tuning
RAG_TOP_K=5
RAG_CHUNK_SIZE=200
RAG_CHUNK_OVERLAP=100

# PostgreSQL (app data)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=parking_db
POSTGRES_USER=parking_db_user
POSTGRES_PSWD=your-password
POSTGRES_POOL_MIN_SIZE=1
POSTGRES_POOL_MAX_SIZE=2

# LangGraph checkpointer (conversation state — can reuse the same Postgres instance/db as above)
CHECKPOINTER_HOST=localhost
CHECKPOINTER_PORT=5432
CHECKPOINTER_DB=parking_db
CHECKPOINTER_USER=parking_db_user
CHECKPOINTER_PSWD=your-password

# Admin notification mailbox (shared filesystem path between client.py and admin.py)
NOTIFICATION_PATH=/tmp/parking_notification
```

### 5. Initialize the checkpointer tables

LangGraph needs its own tables in PostgreSQL to persist conversation state across turns (and across the client/admin approval handoff):

```bash
uv run python -m chatbot.scripts.init_checkpoiter
```

### 6. Ingest static knowledge base

Load the markdown files from `assets/static/` into Weaviate:

```bash
uv run python -m chatbot.scripts.ingest_static_data
```

This parses and chunks the markdown by heading, assigns a category (`general`, `booking`, `policies`, `faq`), and batch-imports all chunks into the configured Weaviate collection.

## Running the chatbot

The demo runs as two cooperating CLIs that share the same PostgreSQL checkpointer and the filesystem `NOTIFICATION_PATH` mailbox: a customer-facing chat session and a human admin console that approves or rejects reservations out-of-band.

### Customer chat

```bash
uv run python client.py                  # starts a fresh thread with a new random id
uv run python client.py <thread-id>      # resumes (or starts) the given thread id
```

Each conversation is tied to a LangGraph thread id. Pass one explicitly to resume a specific customer's session (e.g. after restarting the CLI, or to continue a thread left waiting on admin approval); omit it to start a brand-new session each run.

### Admin approval console

In a second terminal:

```bash
uv run python admin.py
```

`admin.py` polls `NOTIFICATION_PATH` for pending reservation requests and prompts you to approve (`a`) or reject (`r`) each one. Its decision is written back to the mailbox, and the customer's paused graph thread is resumed automatically — completing the reservation and updating its status in PostgreSQL.

Exit either CLI with `quit`, `exit`, or `Ctrl+C`.

```
you> What parking space types are available?
bot> CityPark offers Standard, Oversized and EV Charging.
you> I'd like to book a space
bot> Sure! What date and time do you need the space?
...
bot> Please confirm:
      - Customer: Jane Doe
      - Parking level: B1
      - Space type: STANDARD
      - Start time: 2026-07-05 09:00
      - End time: 2026-07-05 18:00
      - Plate: ABC-1234
     Book it? (yes / cancel)
you> yes
system> Waiting for admin decision for request admin-request-42...
bot> Your reservaton #42 was approved
```

Meanwhile, in the admin terminal:

```
Waiting for admin notification...

=== Admin Notification START ===
Notification received:
 - Request ID: admin-request-42
 - Calling Thread ID: client-id-23b4dd38-10c3-4e6f-a7de-22ba336c8265
 - Reservation ID: 42
Details:
 - Customer: Jane Doe
 ...
Please approve (approved / rejected)
Response (a/r): a
Notification sent.
=== Admin Notification END ===
```

## Evaluation

The evaluation script measures RAG retrieval quality and LLM answer quality over a built-in dataset of 30 question–answer pairs:

```bash
uv run python -m chatbot.scripts.run_evaluation
```

**Retrieval metrics** (computed per chunk-size / overlap / top-k combination):

| Metric | Description |
|--------|-------------|
| Recall@K | Fraction of relevant categories retrieved |
| Precision@K | Fraction of retrieved chunks that are relevant |
| Hit@K | Whether at least one relevant chunk was retrieved |
| MRR | Mean Reciprocal Rank of the first relevant result |

**LLM answer metrics** (LLM-as-judge):

| Metric | Description |
|--------|-------------|
| Faithfulness | Are all claims in the answer grounded in the retrieved context? |
| Answer correctness | Does the answer match the reference answer? |

To test different chunking strategies, edit the parameters in `chatbot/scripts/run_evaluation.py`:

```python
results = run_evaluations(
    evaluation_dataset=EVALUATION_DATASET,
    collection="evaluation_collection",
    chunk_sizes=[200, 350, 500],
    chunk_overlaps=[50, 100],
    top_k_values=[3, 5],
)
```
