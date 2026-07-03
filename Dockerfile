# quorum -- OpenAI-compatible deliberation proxy.
# Build:  docker build -t quorum .
# Run:    docker run --rm -p 8802:8802 -e QUORUM_OPENROUTER_KEY=sk-or-... \
#           -v "$PWD/config.yaml:/app/config.yaml:ro" quorum
# Then point any OpenAI client's base_url at http://localhost:8802/v1
FROM python:3.12-slim

WORKDIR /app

# Install the package (+ its deps) from source.
COPY pyproject.toml README.md ./
COPY quorum ./quorum
RUN pip install --no-cache-dir .

# Provide config.yaml (council/providers) via a mount and the API key via -e.
# Binds to all interfaces inside the container so the mapped port is reachable.
EXPOSE 8802
ENTRYPOINT ["quorum"]
CMD ["serve", "--api", "--host", "0.0.0.0", "--port", "8802"]
