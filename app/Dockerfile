FROM python:3.12.12

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app .

# Create .streamlit directory for config
RUN mkdir -p .streamlit

# Streamlit config to disable file watching and set server settings
RUN echo '\
[server]\n\
headless = true\n\
port = 8500\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > .streamlit/config.toml

EXPOSE 8500

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8500/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]