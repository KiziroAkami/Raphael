import logging
import os

import discord
from dotenv import load_dotenv
from bot.events import setup_events

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

setup_events(client)

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set in .env")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set in .env")
    client.run(token)
