import os
import time
import random
import base64
import asyncio
import logging
from io import BytesIO
from urllib.parse import quote
from collections import defaultdict, deque

import httpx
from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

try:
    import edge_tts  # voice reply ke liye (na mile to bot text mein reply karega)
except ImportError:
    edge_tts = None

# ====================== CONFIG ======================
TELEGRAM_TOKEN = (
    os.getenv("TELEGRAM_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("BOT_TOKEN")
)
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
API = "https://generativelanguage.googleapis.com/v1beta"

PREFERRED = os.getenv("GEMINI_MODEL", "").strip()
MODELS = ([PREFERRED] if PREFERRED else []) + ["gemini-flash-latest"]

# true = group ke har message/voice/sticker pe reply
REPLY_TO_ALL = os.getenv("REPLY_TO_ALL", "true").lower() == "true"
HISTORY_SIZE = 10

# Awaaz (variables se badal sakte ho). Ladki ke liye: hi-IN-SwaraNeural, en-IN-NeerjaNeural
VOICE_HI = os.getenv("VOICE_HI", "hi-IN-MadhurNeural")
VOICE_EN = os.getenv("VOICE_EN", "en-IN-PrabhatNeural")

SHAYARI_COOLDOWN = 30  # ek insaan har 30 sec mein ek baar /shayari maang sakta hai
DRAW_GAP = 16          # free image service har ~15 sec mein 1 image deti hai

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

VOICE_RULES = """
Ye jawab bolke sunaya jayega (voice note), isliye:
- 1-2 chhote vakya, emoji ya special symbols bilkul mat use kar.
- Agar banda Hindi/Hinglish mein bol raha hai to jawab Devanagari (हिंदी) script mein likh.
- Agar English mein bol raha hai to English mein likh.
"""

SHAYARI_SYSTEM = """Tu ek shayar hai.
- 4 line ki nayi, original shayari likh. Kisi mashhoor shayar, kitaab ya gaane ki line copy mat kar.
- Roman Hinglish (Hindi, English letters mein) mein likh. Agar topic Devanagari mein ho to Devanagari mein likh.
- Mood/topic ke hisaab se likh. Gaali ya ashleel baat nahi.
- Sirf shayari likh, koi intro ya outro nahi.
"""

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
client = httpx.AsyncClient(timeout=20)


# ====================== GEMINI ======================
async def discover_models():
    """Google se pooch ke dekho is key pe kaunse flash models chalte hain."""
    global MODELS
    try:
        r = await client.get(f"{API}/models?pageSize=200", headers={"x-goog-api-key": GEMINI_API_KEY})
        r.raise_for_status()
        found = []
        for m in r.json().get("models", []):
            name = m["name"].removeprefix("models/")
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            if "flash" not in name:
                continue
            if any(x in name for x in ("image", "tts", "live", "audio", "embedding", "robotics", "computer", "thinking")):
                continue
            found.append(name)
        logging.info("Available Gemini models: %s", found)
        found.sort(key=lambda n: (0 if "lite" in n else 1, 0 if "latest" in n else 1, n))
        order = ([PREFERRED] if PREFERRED else []) + found + ["gemini-flash-latest"]
        MODELS = list(dict.fromkeys(order))
        logging.info("Model order: %s", MODELS)
    except Exception as e:
        logging.warning("Model list fetch failed: %s", type(e).__name__)


async def call_gemini(model: str, body: dict) -> str:
    url = f"{API}/models/{model}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY}
    r = await client.post(url, headers=headers, json=body)
    if r.status_code == 400 and "thinkingConfig" in body["generationConfig"]:
        gc = {k: v for k, v in body["generationConfig"].items() if k != "thinkingConfig"}
        r = await client.post(url, headers=headers, json={**body, "generationConfig": gc})
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


async def gemini_generate(system: str, parts: list, max_tokens: int = 150, temperature: float = 0.9) -> str:
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    last_err = None
    for model in MODELS[:4]:
        try:
            return await call_gemini(model, body)
        except httpx.HTTPStatusError as e:
            last_err = e
            logging.warning("Gemini %s -> HTTP %s", model, e.response.status_code)
        except Exception as e:
            last_err = e
            logging.warning("Gemini %s error: %s", model, type(e).__name__)
    raise last_err


# ====================== HELPERS ======================
def should_respond(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str = "") -> bool:
    msg = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if user and user.is_bot:
        return False  # doosre bots ko ignore (loop se bachne ke liye)
    if REPLY_TO_ALL or chat.type == "private":
        return True
    if text and f"@{(context.bot.username or '').lower()}" in text.lower():
        return True
    r = msg.reply_to_message
    return bool(r and r.from_user and r.from_user.id == context.bot.id)


async def keep_typing(bot, chat_id, stop: asyncio.Event, action: str = "typing"):
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, action)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


# ====================== TEXT CHAT ======================
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not msg.text:
        return
    user = update.effective_user
    if user and user.is_bot:
        return

    chat = update.effective_chat
    name = user.first_name if user else "Someone"
    history[chat.id].append(f"{name}: {msg.text}")

    if not should_respond(update, context, msg.text):
        return

    stop = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(context.bot, chat.id, stop))
    try:
        prompt = "Group chat ka log:\n" + "\n".join(history[chat.id]) + f"\n\nAb {name} ko reply de, usi ke style mein."
        reply = await gemini_generate(SYSTEM_PROMPT, [{"text": prompt}])
    except Exception as e:
        logging.error("AI failed: %s", type(e).__name__)
        reply = "Abhi thoda busy hu yaar, 1 min baad dobara bol 😅"
    finally:
        stop.set()
        await typing_task

    history[chat.id].append(f"Bot: {reply}")
    await msg.reply_text(reply)


# ====================== VOICE ======================
def pick_voice(text: str) -> str:
    return VOICE_HI if any("\u0900" <= ch <= "\u097F" for ch in text) else VOICE_EN


async def text_to_speech(text: str):
    if edge_tts is None:
        return None
    clean = text.strip()[:500]
    if not clean:
        return None

    async def _run():
        buf = BytesIO()
        comm = edge_tts.Communicate(clean, pick_voice(clean))
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf

    try:
        buf = await asyncio.wait_for(_run(), timeout=25)
        if buf.tell() == 0:
            return None
        buf.seek(0)
        buf.name = "reply.mp3"
        return buf
    except Exception as e:
        logging.warning("TTS failed: %s", type(e).__name__)
        return None


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not msg.voice:
        return
    if not should_respond(update, context):
        return

    chat = update.effective_chat
    user = update.effective_user
    name = user.first_name if user else "Someone"

    if msg.voice.duration and msg.voice.duration > 60:
        await msg.reply_text("Itna lamba voice nahi sun sakta yaar, 1 min tak ka bhej 😅")
        return

    stop = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(context.bot, chat.id, stop, "record_voice"))
    try:
        tg_file = await msg.voice.get_file()
        audio = bytes(await tg_file.download_as_bytearray())
        log = "\n".join(history[chat.id])
        parts = [
            {"text": f"Group chat ka log:\n{log}\n\n{name} ne ye voice note bheja hai. Sun ke usi ke style mein jawab de."},
            {"inlineData": {"mimeType": "audio/ogg", "data": base64.b64encode(audio).decode()}},
        ]
        reply = await gemini_generate(SYSTEM_PROMPT + VOICE_RULES, parts, max_tokens=200)
        voice_buf = await text_to_speech(reply)
    except Exception as e:
        logging.error("Voice failed: %s", type(e).__name__)
        reply, voice_buf = "Abhi thoda busy hu yaar, 1 min baad dobara bol 😅", None
    finally:
        stop.set()
        await typing_task

    history[chat.id].append(f"{name}: [voice note]")
    history[chat.id].append(f"Bot: {reply}")
    if voice_buf:
        await msg.reply_voice(voice=voice_buf)
    else:
        await msg.reply_text(reply)  # voice na bane to text mein


# ====================== STICKER ======================
sticker_cache = {}


async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not msg.sticker:
        return
    if not should_respond(update, context):
        return

    chat = update.effective_chat
    user = update.effective_user
    name = user.first_name if user else "Someone"
    st = msg.sticker
    history[chat.id].append(f"{name}: [sticker {st.emoji or ''}]")

    chosen = st.file_id  # pack na mile to wahi sticker wapas
    if st.set_name:
        try:
            if st.set_name not in sticker_cache:
                if len(sticker_cache) > 30:
                    sticker_cache.clear()
                sticker_cache[st.set_name] = (await context.bot.get_sticker_set(st.set_name)).stickers
            pack = sticker_cache[st.set_name]
            others = [s for s in pack if s.file_unique_id != st.file_unique_id]
            same_emoji = [s for s in others if s.emoji == st.emoji]
            pool = same_emoji or others
            if pool:
                chosen = random.choice(pool).file_id
        except Exception as e:
            logging.warning("Sticker pack error: %s", type(e).__name__)
    try:
        await msg.reply_sticker(chosen)
    except Exception as e:
        logging.warning("Sticker send failed: %s", type(e).__name__)


# ====================== /shayari ======================
shayari_last = {}


async def shayari(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    uid = user.id if user else 0

    left = SHAYARI_COOLDOWN - (time.monotonic() - shayari_last.get(uid, -999))
    if left > 0:
        await msg.reply_text(f"Thoda ruk yaar, {int(left) + 1} sec baad dobara maang 😄")
        return
    shayari_last[uid] = time.monotonic()

    topic = " ".join(context.args).strip() if context.args else ""
    if not topic and msg.reply_to_message:
        topic = (msg.reply_to_message.text or msg.reply_to_message.caption or "").strip()
    topic = topic[:300] or random.choice(["pyaar", "dosti", "zindagi", "udaasi", "umeed", "yaadein"])

    stop = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(context.bot, chat.id, stop))
    try:
        text = await gemini_generate(SHAYARI_SYSTEM, [{"text": f"Topic/mood: {topic}"}], max_tokens=300, temperature=1.0)
    except Exception as e:
        logging.error("Shayari failed: %s", type(e).__name__)
        text = "Abhi mood nahi ban raha, thodi der baad try kar 😅"
        shayari_last.pop(uid, None)
    finally:
        stop.set()
        await typing_task
    await msg.reply_text(text)


# ====================== /getdraw ======================
draw_lock = asyncio.Lock()
last_draw = 0.0


async def getdraw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_draw
    msg = update.effective_message
    chat = update.effective_chat

    prompt = ""
    if msg.reply_to_message:
        prompt = msg.reply_to_message.text or msg.reply_to_message.caption or ""
    if not prompt and context.args:
        prompt = " ".join(context.args)
    prompt = prompt.strip()[:400]

    if not prompt:
        await msg.reply_text("Kisi text message ke reply mein /getdraw likho, ya /getdraw ke baad batao kya banana hai 🎨")
        return

    wait_msg = await msg.reply_text("Bana raha hu, 10-20 sec ruk 🎨")

    async with draw_lock:
        gap = DRAW_GAP - (time.monotonic() - last_draw)
        if gap > 0:
            await asyncio.sleep(gap)

        url = (
            f"https://image.pollinations.ai/prompt/{quote(prompt)}"
            f"?width=1024&height=1024&model=flux&nologo=true&seed={random.randint(1, 10**6)}"
        )
        data = None
        for attempt in range(2):
            try:
                await context.bot.send_chat_action(chat.id, "upload_photo")
                async with httpx.AsyncClient(timeout=90, follow_redirects=True) as c:
                    r = await c.get(url)
                last_draw = time.monotonic()
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                    data = r.content
                    break
                logging.warning("Draw HTTP %s (try %d)", r.status_code, attempt + 1)
                await asyncio.sleep(DRAW_GAP)
            except Exception as e:
                last_draw = time.monotonic()
                logging.warning("Draw error: %s", type(e).__name__)
                await asyncio.sleep(3)

    try:
        await wait_msg.delete()
    except Exception:
        pass

    if data:
        await msg.reply_photo(photo=BytesIO(data), caption=prompt[:200])
    else:
        await msg.reply_text("Image nahi ban payi, thodi der baad dobara try kar 😅")


# ====================== /start /help ======================
HELP_TEXT = (
    "Mai tumhare group ka AI dost hu 😎\n\n"
    "Commands:\n"
    "/start - intro\n"
    "/help - ye list\n"
    "/getdraw - kisi text ke reply mein likho, mai us par image banaunga (ya /getdraw sher jungle mein)\n"
    "/shayari - nayi shayari (mood bhi de sakte ho: /shayari pyaar, dosti, udaas, funny)\n\n"
    "Aur bhi:\n"
    "• Kuch bhi likho, mai tumhare style mein reply dunga\n"
    "• Voice note bhejo, mai voice mein jawab dunga 🎙️\n"
    "• Sticker bhejo, mai bhi sticker bhejunga 😄"
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    name = user.first_name if user else "dost"
    await update.effective_message.reply_text(f"Hey {name}! 👋\n\n{HELP_TEXT}")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(HELP_TEXT)


# ====================== MAIN ======================
async def on_start(app):
    await discover_models()
    try:
        await app.bot.set_my_commands([
            BotCommand("start", "Bot ka intro"),
            BotCommand("help", "Saare commands"),
            BotCommand("getdraw", "Text se image banao"),
            BotCommand("shayari", "Nayi shayari suno"),
        ])
    except Exception as e:
        logging.warning("set_my_commands failed: %s", type(e).__name__)


def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)
        .post_init(on_start)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("getdraw", getdraw))
    app.add_handler(CommandHandler("shayari", shayari))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.Sticker.ALL, handle_sticker))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.run_polling()


if __name__ == "__main__":
    main()
