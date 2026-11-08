FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY vigileye ./vigileye
COPY config ./config
RUN pip install --no-cache-dir ".[reddit,tokens]"
RUN useradd -m vigil && mkdir -p /data && chown vigil /data
USER vigil
ENV VIGILEYE_STORE=/data/cases.db
VOLUME ["/data"]
EXPOSE 8765
ENTRYPOINT ["vigileye"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765"]
