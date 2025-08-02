# AI Agent Application

A Python-based AI agent application with Streamlit frontend and FastAPI backend, featuring GPT-4o-mini integration and currency rate tools.

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd agent
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   - Frontend (Streamlit): http://localhost:8501
   - Backend API (FastAPI): http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## API Documentation

When running, visit http://localhost:8000/docs for interactive API documentation.

## Troubleshooting

- Ensure your OpenAI API key is set in the `.env` file
- Check that Docker and Docker Compose are installed
- Verify ports 8000 and 8501 are available
- Check logs with `docker-compose logs -f`