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

# Pehla model try hoga, na chale to agle (backup) try honge
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
MODELS = [PRIMARY_MODEL, "gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-2.5-flash-lite"]
MODELS = list(dict.fromkeys(MODELS))  # duplicates hata do

# true  = bot group ke har message pe reply karega
# false = sirf mention / reply karne pe (free limit bachti hai)
REPLY_TO_ALL = os.getenv("REPLY_TO_ALL", "false").lower() == "true"

HISTORY_SIZE = 20

SYSTEM_PROMPT = """Tu ek Telegram group ka normal member hai, bot jaisa mat bol.
Rules:
- Jis member ne abhi message kiya hai, uski language aur style copy kar:
  agar wo Hinglish mein likhe to Hinglish, Hindi mein to Hindi, English mein to English.
- Uska tone match kar: wo mazaaki hai to mazaak kar, serious hai to serious reh,
  slang/short forms (bro, yaar, bhai, lol) use kare to tu bhi kar sakta hai.
- Emojis tabhi use kar jab wo karta ho, aur kam.
- Reply chhota rakh (1-3 line), jaise log chat mein likhte hain.
- "As an AI", lecture, ya lambi list mat de, jab tak koi explicitly na maange.
- Group ki pichli baatein dhyan se padh, aur naam lekar baat kar sakta hai.
- Gaali-galoch, hate ya kisi ko hurt karne wali baat mat kar.
- Agar koi kaam ka sawal pooche (code, padhai, info) to sahi jawab de, but usi casual style mein.
- Sirf reply ka text likh, apne naam ka prefix mat laga.
"""

logging.basicConfig(level=logging.INFO)
# Token log mein na dikhe isliye httpx ke INFO logs band
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))


async def call_gemini(client: httpx.AsyncClient, model: str, body: dict) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = await client.post(url, headers={"x-goog-api-key": GEMINI_API_KEY}, json=body)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def ask_ai(chat_log: str, name: str) -> str:
    prompt = (
        "Group chat ka log:\n"
        f"{chat_log}\n\n"
        f"Ab {name} ko reply de, usi ke style mein."
    )
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 300},
    }

    last_err = None
    async with httpx.AsyncClient(timeout=30) as client:
        for model in MODELS:
            for attempt in range(2):  # har model ko 2 baar try
                try:
                    return await call_gemini(client, model, body)
                except httpx.HTTPStatusError as e:
                    last_err = e
                    code = e.response.status_code
                    logging.warning("Gemini %s -> HTTP %s (try %d)", model, code, attempt + 1)
                    if code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    break  # 404/400 etc: ye model chhodo, agla try karo
                except Exception as e:
                    last_err = e
                    logging.warning("Gemini %s error: %s", model, type(e).__name__)
                    await asyncio.sleep(1)
    raise last_err


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

    await context.bot.send_chat_action(chat.id, "typing")

    try:
        reply = await ask_ai("\n".join(history[chat.id]), name)
    except Exception as e:
        logging.error("AI failed: %s", type(e).__name__)
        await msg.reply_text("Abhi thoda busy hu yaar, 1 min baad dobara bol 😅")
        return

    history[chat.id].append(f"Bot: {reply}")
    await msg.reply_text(reply)


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.run_polling()


if __name__ == "__main__":
    main()
