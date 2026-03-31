import asyncio
import logging

import discord
from bot.formatter import chunk_response
from rag.retriever import query as retrieve
from llm.client import answer

logger = logging.getLogger(__name__)

_ERROR_RESPONSE = (
    "Raphael's calculations encountered an anomaly. "
    "This one shall attempt to answer when systems stabilise."
)


def setup_events(client: discord.Client) -> None:

    @client.event
    async def on_ready() -> None:
        logger.info("Raphael is online as %s", client.user)

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot:
            return

        content = message.content.strip()
        if not content.endswith("?"):
            return
        if len(content) > 500:
            return

        try:
            loop = asyncio.get_event_loop()
            async with message.channel.typing():
                chunks = await loop.run_in_executor(None, retrieve, content)
                response = await loop.run_in_executor(None, answer, content, chunks)

            for part in chunk_response(response):
                await message.channel.send(part)
        except Exception:
            logger.exception(
                "Error handling message from user_id=%s in channel %s",
                message.author.id,
                message.channel.id,
            )
            await message.channel.send(_ERROR_RESPONSE)
