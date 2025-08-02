# Quick Setup Guide

## Prerequisites
- Docker and Docker Compose installed
- OpenAI API key

## Setup Steps

1. **Get your OpenAI API key**:
   - Go to https://platform.openai.com/api-keys
   - Create a new secret key
   - Copy the key

2. **Create environment file**:
   ```bash
   copy env-example.txt .env
   ```
   Then edit `.env` and replace `your_openai_api_key_here` with your actual API key.

3. **Build and run the application**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   - **Chat Interface**: http://localhost:8501
   - **API Documentation**: http://localhost:8000/docs
   - **Backend Health**: http://localhost:8000/health

## Test the AI Agent

The agent can handle complex, natural language queries about currency rates:

**Simple queries:**
- "What is the EUR rate today?"
- "Show me USD exchange rate"

**Complex analysis:**
- "Compare EUR and USD rates"
- "Which is stronger, GBP or EUR?"

**Natural language:**
- "I need to exchange dollars, what's the rate?"
- "Planning a trip to Europe, what's the Euro rate?"