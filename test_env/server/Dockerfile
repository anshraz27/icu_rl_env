FROM python:3.10-slim

# HF Spaces heavily recommends a non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy files with user ownership
COPY --chown=user:user . .

# Install dependencies directly (matching local requirements)
RUN pip install --upgrade pip --no-cache-dir \
	&& pip install --no-cache-dir openenv-core fastapi uvicorn pydantic

# Hugging Face Spaces routes traffic specifically to 7860
EXPOSE 7860

# Point to HF port and correct python module path from root
# In the built container, the code lives directly under /app with
# a top-level "server" package, not a "test_env" package.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
