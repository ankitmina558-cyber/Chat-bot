import os
import asyncio
import logging
from collections import defaultdict, deque

import httpx
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

TELEGRAM_TOKEN = (
    os.getenv("TELEGRAM_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("BOT_TOKEN")
)
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Sabse fast model pehle, na chale to backup
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
MODELS = list(dict.fromkeys([PRIMARY_MODEL, "gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-flash-latest"]))

# true  = bot group ke har message pe reply karega
# false = sirf mention / reply karne pe
REPLY_TO_ALL = os.getenv("REPLY_TO_ALL", "false").lower() == "true"

HISTORY_SIZE = 10

SYSTEM_PROMPT = """Tu ek Telegram group ka normal member hai, bot jaisa mat bol.
Rules:
- Jis member ne abhi message kiya hai, uski language aur style copy kar:
  agar wo Hinglish mein likhe to Hinglish, Hindi mein to Hindi, English mein to English.
- Uska tone match kar: wo mazaaki hai to mazaak kar, serious hai to serious reh,
  slang/short forms (bro, yaar, bhai, lol) use kare to tu bhi kar sakta hai.
- Emojis tabhi use kar jab wo karta ho, aur kam.
- Reply bahut chhota rakh (1-2 line), jaise log chat mein likhte hain.
- "As an AI", lecture, ya lambi list mat de, jab tak koi explicitly na maange.
- Group ki pichli baatein dhyan se padh, aur naam lekar baat kar sakta hai.
- Gaali-galoch, hate ya kisi ko hurt karne wali baat mat kar.
- Agar koi kaam ka sawal pooche (code, padhai, info) to sahi jawab de, but usi casual style mein.
- Sirf reply ka text likh, apne naam ka prefix mat laga.
"""

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))

# Ek hi connection baar baar use hoga (har message pe naya connection nahi banega = fast)
client = httpx.AsyncClient(timeout=15)


async def call_gemini(model: str, body: dict) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = await client.post(url, headers={"x-goog-api-key": GEMINI_API_KEY}, json=body)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


async def ask_ai(chat_log: str, name: str) -> str:
    prompt = f"Group chat ka log:\n{chat_log}\n\nAb {name} ko reply de, usi ke style mein."
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 150,
            "thinkingConfig": {"thinkingBudget": 0},  # soch-vichar band = fast reply
        },
    }

    last_err = None
    for model in MODELS:
        try:
            return await call_gemini(model, body)
        except httpx.HTTPStatusError as e:
            last_err = e
            logging.warning("Gemini %s -> HTTP %s", model, e.response.status_code)
        except Exception as e:
            last_err = e
            logging.warning("Gemini %s error: %s", model, type(e).__name__)
        await asyncio.sleep(0.2)  # turant agle model pe
    raise last_err


async def keep_typing(bot, chat_id, stop: asyncio.Event):
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, "typing")
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not msg.text:
        return

    chat = update.effective_chat
    user = update.effective_user
    name = user.first_name if user else "Someone"
    text = msg.text

    history[chat.id].append(f"{name}: {text}")

    bot_username = (context.bot.username or "").lower()
    mentioned = f"@{bot_username}" in text.lower()
    replied_to_bot = (
        msg.reply_to_message is not None
        and msg.reply_to_message.from_user is not None
        and msg.reply_to_message.from_user.id == context.bot.id
    )
    is_private = chat.type == "private"

    if not (REPLY_TO_ALL or mentioned or replied_to_bot or is_private):
        return

    stop = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(context.bot, chat.id, stop))
    try:
        reply = await ask_ai("\n".join(history[chat.id]), name)
    except Exception as e:
        logging.error("AI failed: %s", type(e).__name__)
        reply = "Abhi thoda busy hu yaar, 1 min baad dobara bol 😅"
    finally:
        stop.set()
        await typing_task

    history[chat.id].append(f"Bot: {reply}")
    await msg.reply_text(reply)


def main():
    # concurrent_updates: group mein kai log ek saath bolein to bhi sabko turant jawab
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.run_polling()


if __name__ == "__main__":
    main()
