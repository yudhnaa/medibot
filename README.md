# MediBot Django Backend

Django REST backend for MediBot, a medical assistant platform with cookie-based JWT authentication, intake-aware chatbot sessions, retrieval-augmented generation, vector search, and chest X-ray vision analysis.

## Skills & Badges

<p align="center">
    <img src="https://go-skill-icons.vercel.app/api/icons?i=python,django,postgres,redis,docker,githubactions,git" alt="Skills" />
</p>

<p align="center">
    <a href="https://www.djangoproject.com/">
        <img src="https://img.shields.io/badge/Django-6.0.1-092E20.svg" alt="Django">
    </a>
    <a href="https://www.django-rest-framework.org/">
        <img src="https://img.shields.io/badge/DRF-3.16.1-red.svg" alt="Django REST Framework">
    </a>
    <a href="https://github.com/pgvector/pgvector">
        <img src="https://img.shields.io/badge/PostgreSQL-pgvector-blue.svg" alt="PostgreSQL pgvector">
    </a>
    <a href="https://docs.celeryq.dev/">
        <img src="https://img.shields.io/badge/Celery-5.6.2-green.svg" alt="Celery">
    </a>
    <a href="https://www.docker.com/">
        <img src="https://img.shields.io/badge/platform-Docker%20%7C%20macOS%20%7C%20Linux-blue" alt="Platform">
    </a>
</p>

## Features

- Cookie-based JWT authentication with access and refresh token endpoints
- Intake-aware chatbot sessions with Server-Sent Events streaming at `/api/v1/chatbot/chat/`
- Retrieval-augmented generation using LangChain, LangGraph, Google Gemini, OpenRouter, and pgvector-backed storage
- Vector-store APIs for embedding and medical document retrieval workflows
- Chest X-ray vision module for image analysis, embedding, and similarity search
- Vietnamese medical NLP services for NER, negation handling, and text normalization
- Celery worker, beat scheduler, Redis broker, and Flower monitoring support
- Swagger and Redoc API documentation generated with drf-yasg
- Docker Compose profiles for local development, staging, and production deployment
- Offline RAG benchmark tooling with RAGAS datasets and evaluation commands

## Acknowledgements

- [Django](https://www.djangoproject.com/) and [Django REST Framework](https://www.django-rest-framework.org/) for the API foundation
- [LangChain](https://www.langchain.com/), [LangGraph](https://www.langchain.com/langgraph), and [Google Gemini](https://ai.google.dev/) for chatbot orchestration
- [pgvector](https://github.com/pgvector/pgvector) for vector search in PostgreSQL
- [Celery](https://docs.celeryq.dev/) and [Redis](https://redis.io/) for background processing
- [torchxrayvision](https://github.com/mlmed/torchxrayvision) for X-ray model support

## Installation

Clone the repository, create a virtual environment, install dependencies, and copy environment defaults.

```bash
python -m venv env
source env/bin/activate
pip install -r requirements/dev.txt
cp .env.example .env
```

Update `.env` with local secrets and API keys:

```bash
SECRET_KEY=your-secret-key-here
GOOGLE_API_KEY=your-google-api-key
OPENROUTER_API_KEY=your-openrouter-key
DB_DEV_NAME=medibot
DB_DEV_USER=postgres
DB_DEV_PASSWORD=postgres
DB_DEV_HOST=localhost
DB_DEV_PORT=5432
```

Run migrations and start local server:

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Start background services when chatbot or scheduled jobs need them:

```bash
redis-server
celery -A django_template.celery worker -l info -E
celery -A django_template.celery beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
celery -A django_template.celery flower --port=5555
```

Run full development stack with Docker Compose:

```bash
docker compose -f docker-compose.dev.yml up --build
```

Production-style compose uses published images and `.env` values:

```bash
export MEDIBOT_BACKEND_IMAGE=ghcr.io/<owner>/<repo>
export IMAGE_TAG=<commit-sha>
docker compose -f docker-compose.yml pull
docker compose -f docker-compose.yml up -d
```

## API Documentation

After server starts, open generated API docs:

```text
http://127.0.0.1:8000/swagger/
http://127.0.0.1:8000/redoc/
http://127.0.0.1:8000/swagger.json
```

Public API routes are mounted under `/api/v1/`:

- `/api/v1/auth/` — authentication and session endpoints
- `/api/v1/chatbot/` — chatbot sessions, intake flow, and streaming chat
- `/api/v1/vector-store/` — embedding and retrieval endpoints
- `/api/v1/vision/` — image analysis and similarity endpoints

## Testing

Run unit tests:

```bash
python manage.py test
```

Run coverage-backed tests:

```bash
coverage run --source='.' manage.py test
coverage report
coverage html
```

Run formatting and lint checks:

```bash
flake8 .
black .
isort .
```

Install and run pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

## Useful Commands

Load default customer fixture:

```bash
python manage.py loaddata authentication/fixtures/customer.json --app authentication.customer
```

Clean expired JWT tokens:

```bash
python manage.py flushexpiredtokens
```

Collect static files:

```bash
python manage.py collectstatic --no-input --clear
```

## Contributing

Contributions are welcome.

Before opening a pull request:

1. Create a focused branch from `develop`.
2. Keep changes surgical and scoped to one purpose.
3. Run formatting, linting, and tests.
4. Update API docs or README details when behavior changes.
5. Do not commit `.env`, local credentials, model weights, generated coverage HTML, or local database files.

CI installs `requirements/test.txt`, runs migrations, collects static files, loads fixtures, checks `flake8`, `black`, `isort`, and runs coverage-backed Django tests.

## License

MediBot Project License.
