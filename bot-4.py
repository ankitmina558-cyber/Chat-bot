import os
import time
import random
import base64
import asyncio
import logging
from io import BytesIO
from urllib.parse import quote
from collections import deque

import httpx
from telegram import Update, BotCommand
from telegram.error import BadRequest, TimedOut
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
cooldown = {}        # model -> kab tak skip karna hai (429 "limit" aane par)
COOLDOWN_429 = 300   # 5 minute

# false (default) = group mein bot sirf tab bolega jab koi uske message ka reply kare ya @mention kare
# true = group ke har message/voice/sticker pe reply
REPLY_TO_ALL = os.getenv("REPLY_TO_ALL", "false").lower() == "true"
HISTORY_SIZE = 10  # har insaan ke last itne messages yaad rahenge
HISTORY_MINUTES = int(os.getenv("HISTORY_MINUTES", "15"))  # itni der chup rahe to bot purani baat bhool jayega
HISTORY_TTL = HISTORY_MINUTES * 60

# Awaaz (variables se badal sakte ho). Ladki ke liye: hi-IN-SwaraNeural, en-IN-NeerjaNeural
VOICE_HI = os.getenv("VOICE_HI", "hi-IN-MadhurNeural")
VOICE_EN = os.getenv("VOICE_EN", "en-IN-PrabhatNeural")

POLLINATIONS_KEY = os.getenv("POLLINATIONS_KEY", "").strip()  # optional, free key se limit kam hoti hai
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
- Har insaan se tu alag baat karta hai. Sirf usi insaan ki baat ka jawab de jisne abhi likha.
- Agar baat ka log mein sirf ek hi message hai, to ye nayi baat hai. Kisi aur ka ya pichla topic mat uthana.
- Is insaan ki pichli baatein dhyan se padh, aur naam lekar baat kar sakta hai.
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

history = {}  # (chat_id, user_id) -> deque[(time, line)]


def hist_key(update: Update):
    user = update.effective_user
    return (update.effective_chat.id, user.id if user else 0)


def add_hist(key, line: str):
    if len(history) > 500:  # memory bachane ke liye purane insaan hata do
        now = time.monotonic()
        for k in [k for k, d in history.items() if not d or now - d[-1][0] > HISTORY_TTL]:
            history.pop(k, None)
    history.setdefault(key, deque(maxlen=HISTORY_SIZE)).append((time.monotonic(), line))


def get_hist(key) -> str:
    dq = history.get(key)
    if not dq:
        return ""
    now = time.monotonic()
    while dq and now - dq[0][0] > HISTORY_TTL:  # TTL se purani baatein bhool jao
        dq.popleft()
    if not dq:
        history.pop(key, None)
        return ""
    return "\n".join(line for _, line in dq)


def reply_context(msg, bot_id: int) -> str:
    """Agar user ne kisi aur ke message ka reply kiya hai to wo text bhi do."""
    r = msg.reply_to_message
    if not r or (r.from_user and r.from_user.id == bot_id):
        return ""
    t = (r.text or r.caption or "").strip()
    if not t:
        return ""
    who = r.from_user.first_name if r.from_user else "kisi"
    return f"\n(Ye message {who} ke is message ka reply hai: \"{t[:200]}\")"
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
    now = time.monotonic()
    ready = [m for m in MODELS if cooldown.get(m, 0) <= now]  # jo model 429 de raha hai use kuch der skip
    for model in (ready or MODELS)[:5]:
        try:
            return await call_gemini(model, body)
        except httpx.HTTPStatusError as e:
            last_err = e
            code = e.response.status_code
            logging.warning("Gemini %s -> HTTP %s", model, code)
            if code == 404 and len(MODELS) > 1 and model in MODELS:
                MODELS.remove(model)  # ye model is key pe nahi chalta, aage skip
            elif code == 429:
                cooldown[model] = time.monotonic() + COOLDOWN_429
        except Exception as e:
            last_err = e
            logging.warning("Gemini %s error: %s", model, type(e).__name__)
    raise last_err


# ====================== HELPERS ======================
BUSY_MSG = "Abhi thoda busy hu yaar, 1 min baad dobara bol 😅"
QUOTA_MSG = "Meri free AI limit abhi khatam ho gayi yaar, thodi der baad bol 😅"


def busy_text(err) -> str:
    """Gemini fail hone par sahi wajah ka message (limit vs busy)."""
    if isinstance(err, httpx.HTTPStatusError) and err.response.status_code == 429:
        return QUOTA_MSG
    return BUSY_MSG


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


SEND_METHODS = {
    "text": "send_message",
    "photo": "send_photo",
    "voice": "send_voice",
    "sticker": "send_sticker",
}


async def reply_safe(msg, context, kind, **kw):
    """Reply karo. Agar original message delete ho gaya ho (kai groups mein
    cleanup bots commands turant delete kar dete hain) to bina reply ke seedha bhejo."""
    try:
        return await getattr(msg, f"reply_{kind}")(**kw)
    except BadRequest as e:
        if "not found" in str(e).lower():
            return await getattr(context.bot, SEND_METHODS[kind])(chat_id=msg.chat_id, **kw)
        raise


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
    key = hist_key(update)

    # Group mein log aapas mein baat karte hain, unke messages bot ki memory mein nahi jaane chahiye
    if not should_respond(update, context, msg.text):
        return
    add_hist(key, f"{name}: {msg.text}")

    stop = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(context.bot, chat.id, stop))
    try:
        prompt = (
            f"{name} ke saath ab tak ki baat (sirf isi insaan ki):\n{get_hist(key)}"
            f"{reply_context(msg, context.bot.id)}\n\n"
            f"Ab {name} ke aakhri message ka reply de, usi ke style mein."
        )
        reply = await gemini_generate(SYSTEM_PROMPT, [{"text": prompt}])
    except Exception as e:
        logging.error("AI failed: %s", type(e).__name__)
        reply = busy_text(e)
    finally:
        stop.set()
        await typing_task

    add_hist(key, f"Bot: {reply}")
    await reply_safe(msg, context, "text", text=reply)


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
        await reply_safe(msg, context, "text", text="Itna lamba voice nahi sun sakta yaar, 1 min tak ka bhej 😅")
        return

    stop = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(context.bot, chat.id, stop, "record_voice"))
    try:
        audio = None
        for attempt in range(2):  # download dobara karna safe hai, to ek baar retry
            try:
                tg_file = await msg.voice.get_file()
                audio = bytes(await tg_file.download_as_bytearray())
                break
            except TimedOut:
                logging.warning("Voice download timed out (try %d)", attempt + 1)
        if audio is None:
            raise TimedOut("voice download")
        key = hist_key(update)
        log = get_hist(key)
        parts = [
            {"text": f"{name} ke saath ab tak ki baat (sirf isi insaan ki):\n{log}\n\n{name} ne ye voice note bheja hai. Sun ke usi ke style mein jawab de."},
            {"inlineData": {"mimeType": "audio/ogg", "data": base64.b64encode(audio).decode()}},
        ]
        reply = await gemini_generate(SYSTEM_PROMPT + VOICE_RULES, parts, max_tokens=200)
        voice_buf = await text_to_speech(reply)
    except Exception as e:
        logging.error("Voice failed: %s", type(e).__name__)
        if isinstance(e, TimedOut):
            reply = "Tera voice Telegram se nahi aa paya (net slow hai), ek baar dobara bhej 🙏"
        else:
            reply = busy_text(e)
        voice_buf = None
    finally:
        stop.set()
        await typing_task

    key = hist_key(update)
    add_hist(key, f"{name}: [voice note]")
    add_hist(key, f"Bot: {reply}")
    if voice_buf:
        await reply_safe(msg, context, "voice", voice=voice_buf)
    else:
        await reply_safe(msg, context, "text", text=reply)  # voice na bane to text mein


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
    add_hist(hist_key(update), f"{name}: [sticker {st.emoji or ''}]")

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
        await reply_safe(msg, context, "sticker", sticker=chosen)
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
        await reply_safe(msg, context, "text", text=f"Thoda ruk yaar, {int(left) + 1} sec baad dobara maang 😄")
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
    await reply_safe(msg, context, "text", text=text)


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
        await reply_safe(msg, context, "text", text="Kisi text message ke reply mein /getdraw likho, ya /getdraw ke baad batao kya banana hai 🎨")
        return

    wait_msg = await reply_safe(msg, context, "text", text="Bana raha hu, 10-20 sec ruk 🎨")

    async with draw_lock:
        gap = DRAW_GAP - (time.monotonic() - last_draw)
        if gap > 0:
            await asyncio.sleep(gap)

        seed = random.randint(1, 10**6)
        q = quote(prompt)
        urls = []
        if POLLINATIONS_KEY:  # key ho to pehle wahi (zyada limit)
            urls.append((
                f"https://gen.pollinations.ai/image/{q}?model=flux&width=768&height=768&nologo=true&seed={seed}&key={POLLINATIONS_KEY}",
                {"Authorization": f"Bearer {POLLINATIONS_KEY}"},
            ))
        urls.append((
            f"https://image.pollinations.ai/prompt/{q}?width=768&height=768&model=flux&nologo=true&seed={seed}",
            {},
        ))
        data = None
        for url, hdrs in urls:
            try:
                await context.bot.send_chat_action(chat.id, "upload_photo")
                async with httpx.AsyncClient(timeout=90, follow_redirects=True) as c:
                    r = await c.get(url, headers=hdrs)
                last_draw = time.monotonic()
                is_img = r.headers.get("content-type", "").startswith("image")
                # khali ya chhoti file ko image na maano (isi se "File must be non-empty" aata tha)
                if r.status_code == 200 and is_img and len(r.content) > 2000:
                    data = r.content
                    break
                note = "image" if is_img else r.text[:100].replace("\n", " ")
                logging.warning("Draw HTTP %s, %d bytes, %s", r.status_code, len(r.content), note)
            except Exception as e:
                last_draw = time.monotonic()
                logging.warning("Draw error: %s", type(e).__name__)

    try:
        await wait_msg.delete()
    except Exception:
        pass

    if data:
        try:
            await reply_safe(msg, context, "photo", photo=BytesIO(data), caption=prompt[:200])
        except TimedOut:
            # Telegram ne jawab dene mein der ki, par photo aksar baad mein pahunch jati hai
            logging.warning("Photo send timed out")
            await reply_safe(msg, context, "text", text="Telegram thoda slow hai, image kuch der mein aa sakti hai ⏳ (na aaye to dobara /getdraw karo)")
        except Exception as e:
            logging.warning("Photo send failed: %s: %s", type(e).__name__, e)
            low = str(e).lower()
            if "forbidden" in low or "rights" in low:
                note = "Image ban gayi par yahan bhej nahi paya, group mein media permission check karo 😅"
            else:
                note = "Image bhejte waqt error aaya, dobara try kar 😅"
            await reply_safe(msg, context, "text", text=note)
    else:
        await reply_safe(msg, context, "text", text="Free image service abhi busy ya limit pe hai, thodi der baad dobara try kar 😅")


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
    await reply_safe(update.effective_message, context, "text", text=f"Hey {name}! 👋\n\n{HELP_TEXT}")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply_safe(update.effective_message, context, "text", text=HELP_TEXT)


async def on_error(update, context):
    logging.error("Handler error: %s: %s", type(context.error).__name__, context.error)


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
        .read_timeout(90)
        .write_timeout(120)
        .connect_timeout(20)
        .pool_timeout(20)
        .post_init(on_start)
        .build()
    )
    app.add_error_handler(on_error)
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
