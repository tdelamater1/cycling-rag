# Cycling RAG — AI Training Assistant

## Project Overview
A RAG (Retrieval Augmented Generation) system that analyses cycling training data from intervals.icu
and answers natural language questions about performance, trends, and training recommendations.

## Architecture

### Data Layer
- **PostgreSQL** — structured storage for all activity metrics (power, HR, TSS, distance, duration, dates)
- **ChromaDB** — vector store of natural language ride summaries for semantic retrieval
- **intervals.icu API** — primary data source, polled automatically on a schedule

### Application Layer
- **Python 3.13**
- **FastAPI** — exposes a `/query` endpoint that Open WebUI can call as a tool
- **Ollama** — local LLM inference (qwen3:8b at http://192.168.4.93:11434)
- **ChromaDB Python client** — vector search
- **psycopg2 or asyncpg** — Postgres connection

### Interface
- **CLI** — for local development and testing
- **Open WebUI tool** — primary end-user interface via FastAPI endpoint

## Project Structure
```
cycling-rag/
├── CLAUDE.md
├── README.md
├── .env.example
├── .env                    # never commit this
├── requirements.txt
├── docker-compose.yml      # for Proxmox deployment
├── src/
│   ├── __init__.py
│   ├── config.py           # env vars, settings
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── intervals_client.py   # intervals.icu API client
│   │   ├── sync.py               # fetch and store new activities
│   │   └── embedder.py           # chunk activities, embed, store in ChromaDB
│   ├── db/
│   │   ├── __init__.py
│   │   ├── postgres.py           # postgres connection + queries
│   │   └── chroma.py             # chromadb connection + queries
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py          # hybrid retrieval (postgres + chroma)
│   │   ├── prompts.py            # prompt templates
│   │   └── query_engine.py      # main RAG pipeline
│   └── api/
│       ├── __init__.py
│       └── main.py               # FastAPI app
├── scripts/
│   └── sync_now.py               # manual one-off sync
└── tests/
    └── test_query.py
```

## Data Model

### PostgreSQL — activities table
```sql
CREATE TABLE activities (
    id              BIGINT PRIMARY KEY,      -- intervals.icu activity id
    name            TEXT,
    sport_type      TEXT,                    -- 'Ride', 'VirtualRide' etc
    start_date      TIMESTAMPTZ,
    duration_seconds INTEGER,
    distance_meters  FLOAT,
    elevation_meters FLOAT,
    avg_power_watts  FLOAT,
    normalized_power FLOAT,
    avg_hr           FLOAT,
    max_hr           FLOAT,
    tss              FLOAT,                  -- Training Stress Score
    ctl              FLOAT,                  -- Chronic Training Load (fitness)
    atl              FLOAT,                  -- Acute Training Load (fatigue)
    tsb              FLOAT,                  -- Training Stress Balance (form)
    intensity_factor FLOAT,
    avg_cadence      FLOAT,
    kilojoules       FLOAT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_activities_start_date ON activities(start_date);
CREATE INDEX idx_activities_sport_type ON activities(sport_type);
```

### ChromaDB — ride embeddings
Each document is a natural language summary of a single activity:

```
"Outdoor ride on 2024-03-15. Duration: 1h 45m. Distance: 52km.
Normalized power: 210W. Average HR: 148bpm. TSS: 112.
High intensity effort. Fitness (CTL): 68, Fatigue (ATL): 78, Form (TSB): -10."
```

Metadata stored alongside each embedding:
- activity_id
- start_date
- sport_type
- tss
- ctl, atl, tsb

## RAG Query Pipeline

For each user question:
1. **Classify** the query type (trend comparison, recommendation, general)
2. **Postgres query** — pull exact metrics for relevant date ranges
3. **ChromaDB retrieval** — semantic search for similar training blocks
4. **Prompt assembly** — combine structured data + retrieved summaries + user question
5. **LLM call** — send to qwen3:8b via Ollama API
6. **Return response** — stream back to user

## Chunking Strategy
- One chunk = one activity summary (natural language, ~100 words)
- Only cycling activities are embedded (filter out runs, swims etc)
- Activities with missing power data get a degraded summary (HR-only)
- Re-embed if an activity is updated in intervals.icu

## Key Queries to Support

### Phase 1
1. **Period comparison** — "Compare my last N days of riding to the same period last year"
   - Postgres: aggregate avg power, TSS, volume for both periods
   - ChromaDB: retrieve similar rides from both periods for qualitative context
   - LLM: summarise the delta and trend

2. **Interval recommendation** — "Recommend an interval session based on my recent training"
   - Postgres: check current CTL, ATL, TSB (form)
   - ChromaDB: find similar training blocks and what followed them
   - LLM: recommend appropriate session type and intensity

### Phase 2 (future)
- Power curve analysis
- Peak performance identification
- Recovery recommendations
- Race readiness assessment

## intervals.icu API
- Base URL: `https://intervals.icu/api/v1`
- Auth: HTTP Basic — username is `API_KEY`, password is your intervals.icu API key
- Key endpoints:
  - `GET /athlete/{id}/activities` — list activities with metrics
  - `GET /athlete/{id}/wellness` — CTL/ATL/TSB wellness data
- Athlete ID and API key stored in `.env`
- Rate limit: be respectful, max 1 request/second

## Environment Variables (.env)
```
INTERVALS_ATHLETE_ID=your_athlete_id
INTERVALS_API_KEY=your_api_key
POSTGRES_URL=postgresql://user:password@localhost:5432/cycling_rag
CHROMA_PATH=./chroma_db
OLLAMA_BASE_URL=http://192.168.4.93:11434
OLLAMA_MODEL=qwen3:8b
```

## Development vs Production

### Local (laptop)
- Postgres runs in Docker: `docker-compose up postgres`
- ChromaDB runs local file mode (no server needed)
- Ollama points at mini PC: `http://192.168.4.93:11434`
- Run sync manually: `python scripts/sync_now.py`
- Query via CLI: `python -m src.rag.query_engine "compare last 90 days to last year"`

### Production (Proxmox Debian Docker VM)
- Everything runs in Docker Compose
- Postgres persisted via Docker volume
- ChromaDB persisted via Docker volume
- Sync runs on cron schedule (daily at 6am)
- FastAPI exposed on port 8000
- Open WebUI connects to FastAPI as a tool endpoint

## Coding Standards
- Type hints on all functions
- Docstrings on all public functions and classes
- Environment variables via python-dotenv, never hardcoded
- Async where it makes sense (FastAPI routes, DB calls)
- Keep RAG pipeline, DB layer, and API layer clearly separated
- Prefer simple over clever — this is a learning project

## Current Status
- [x] Project scaffolding
- [x] intervals.icu API client
- [x] Postgres schema and connection
- [x] Activity sync script
- [x] Basic RAG query engine (period comparison + recommendation + general)
- [x] Prompt templates
- [x] CLI interface
- [x] FastAPI endpoint (`GET /health`, `POST /query`)
- [ ] ChromaDB connection (`db/chroma.py`)
- [ ] ChromaDB embedder (`ingest/embedder.py`) — natural language summaries → vector store
- [ ] Hybrid retriever (`rag/retriever.py`) — wire ChromaDB into query engine
- [ ] Interval recommendation query (needs ChromaDB context)
- [ ] Open WebUI tool integration
- [ ] Docker Compose for Proxmox deployment
- [ ] Tests
