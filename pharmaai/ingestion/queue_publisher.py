"""
pharmaai/ingestion/queue_publisher.py
Async RabbitMQ publisher using aio-pika.
Publishes normalised Document JSON to the appropriate queue.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aio_pika
from aio_pika import Message, DeliveryMode

from pharmaai.core.config import get_settings
from pharmaai.core.schemas import Document

logger = logging.getLogger("pharmaai.ingestion.queue")


class QueuePublisher:
    def __init__(self):
        self._connection: aio_pika.Connection | None = None
        self._channel: aio_pika.Channel | None = None
        self._settings = get_settings()

    async def connect(self) -> None:
        url = self._settings.rabbitmq.url
        self._connection = await aio_pika.connect_robust(url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=100)

        # Declare queues
        for qname in self._settings.rabbitmq.queues.values():
            await self._channel.declare_queue(
                qname, durable=True, arguments={"x-message-ttl": 86_400_000}
            )
        logger.info("Connected to RabbitMQ and queues declared.")

    async def publish(
        self,
        document: Document,
        queue: str | None = None,
    ) -> None:
        if self._channel is None:
            await self.connect()
        target = queue or self._settings.rabbitmq.queues.get("ingestion", "pharmaai.ingestion")
        body = document.model_dump_json().encode()
        await self._channel.default_exchange.publish(
            Message(body, delivery_mode=DeliveryMode.PERSISTENT),
            routing_key=target,
        )

    async def publish_batch(
        self, documents: list[Document], queue: str | None = None
    ) -> None:
        for doc in documents:
            await self.publish(doc, queue)

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()


class DirectProcessor:
    """
    Fallback processor when queues are disabled.
    Directly embeds and indexes documents.
    """

    async def process_batch(self, documents: list[Document]) -> int:
        from pharmaai.embeddings.models import embedding_service
        from pharmaai.embeddings.index import faiss_index
        from pharmaai.core.database import document_store

        if not documents:
            return 0

        texts = [d.content for d in documents]
        embeddings = embedding_service.embed(texts)  # (N, 2304)

        faiss_indices = faiss_index.add(
            embeddings, [d.id for d in documents]
        )
        for doc, idx in zip(documents, faiss_indices):
            await document_store.upsert(doc, faiss_idx=idx)

        # Periodically save index
        if faiss_index.size % 1000 < len(documents):
            faiss_index.save()

        return len(documents)