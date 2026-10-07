"""Daily kids Shorts pipeline: pick niche+topic -> script -> voice -> visuals -> 9:16 video + captions."""
import asyncio, datetime, json, os, pathlib, random, subprocess, time

import edge_tts
import requests
import yaml

ROOT = pathlib.Path(__file__).parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
TOPICS = yaml.safe_load(open(ROOT / "topics.yaml", encoding="utf-8"))
STATE_F = ROOT / "data" / "state.json"
CONTENT_POOL_F = ROOT / "data" / "content_pool.json"
W, H, FPS = 1080, 1920, 30
GEMINI_WORKING_MODEL = None

# ---------- helpers ----------
def sh(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError("command failed: " + " ".join(map(str, cmd[:3])))
    return r.stdout


def duration(path):
    return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", str(path)]).strip())


def gemini(prompt):
    """Call Gemini with retries, model fallback, and per-run model caching."""

    global GEMINI_WORKING_MODEL

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    configured_models = CFG.get(
        "gemini_models",
        ["gemini-3.8-flash"]
    )

    # If a model already worked during this run, use it first.
    if GEMINI_WORKING_MODEL:
        models = [
            GEMINI_WORKING_MODEL,
            *[
                model
                for model in configured_models
                if model != GEMINI_WORKING_MODEL
            ],
        ]
    else:
        models = configured_models

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.9,
        },
    }

    retryable_statuses = {408, 429, 500, 502, 503, 504}
    max_attempts = 2

    for model in models:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent"
        )

        for attempt in range(1, max_attempts + 1):
            try:
                print(
                    f"Gemini request: model={model}, "
                    f"attempt={attempt}/{max_attempts}"
                )

                response = requests.post(
                    url,
                    json=body,
                    timeout=90,
                    headers={
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                )

                if response.status_code in retryable_statuses:
                    print(
                        f"Gemini temporary error: "
                        f"HTTP {response.status_code}, model={model}"
                    )

                    if attempt < max_attempts:
                        delay = 5 * (2 ** (attempt - 1))
                        jitter = random.uniform(0, 2)

                        print(
                            f"Retrying {model} in "
                            f"{delay + jitter:.1f}s..."
                        )

                        time.sleep(delay + jitter)

                    continue

                response.raise_for_status()

                data = response.json()

                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError(
                        f"Gemini returned no candidates: {data}"
                    )

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise RuntimeError(
                        f"Gemini returned no content parts: {data}"
                    )

                text = parts[0].get("text")
                if not text:
                    raise RuntimeError(
                        f"Gemini returned empty text: {data}"
                    )

                result = json.loads(text)

                # Remember the model that successfully generated content.
                GEMINI_WORKING_MODEL = model

                print(
                    f"Gemini success: model={model} "
                    f"(cached for this run)"
                )

                return result

            except requests.RequestException as e:
                print(
                    f"Gemini network error: model={model}, "
                    f"attempt={attempt}: {e}"
                )

                if attempt < max_attempts:
                    delay = 5 * (2 ** (attempt - 1))
                    jitter = random.uniform(0, 2)

                    print(
                        f"Retrying {model} in "
                        f"{delay + jitter:.1f}s..."
                    )

                    time.sleep(delay + jitter)

            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Gemini returned invalid JSON from {model}: {e}"
                ) from e

            except Exception as e:
                raise RuntimeError(
                    f"Gemini request failed for {model}: {e}"
                ) from e

        print(f"Gemini model exhausted: {model}")

    raise RuntimeError(
        "All configured Gemini models failed. "
        f"Tried: {', '.join(models)}"
    )


# ---------- state ----------
def load_state():
    if STATE_F.exists():
        state = json.load(open(STATE_F, encoding="utf-8"))
    else:
        state = {
            "day": 0,
            "used": [],
            "extra": {},
            "used_content": []
        }

    # Keep compatibility with older state.json files.
    state.setdefault("day", 0)
    state.setdefault("used", [])
    state.setdefault("extra", {})
    state.setdefault("used_content", [])

    return state


def save_state(s):
    STATE_F.parent.mkdir(exist_ok=True)
    json.dump(s, open(STATE_F, "w"), indent=2)


def load_content_pool():
    """Load the static emergency content pool."""
    if not CONTENT_POOL_F.exists():
        print("Content pool not found:", CONTENT_POOL_F)
        return {}

    try:
        with open(CONTENT_POOL_F, encoding="utf-8") as f:
            pool = json.load(f)

        if not isinstance(pool, dict):
            raise ValueError("Content pool must contain a JSON object.")

        return pool

    except (OSError, json.JSONDecodeError, ValueError) as e:
        print("Content pool could not be loaded:", e)
        return {}


def get_unused_pool_content(niche, state):
    """Return one unused content item for the requested niche."""
    pool = load_content_pool()
    items = pool.get(niche, [])

    if not isinstance(items, list):
        print(f"Invalid content pool for niche: {niche}")
        return None

    used_ids = set(state.get("used_content", []))

    available = [
        item
        for item in items
        if isinstance(item, dict)
        and item.get("id")
        and item["id"] not in used_ids
    ]

    if not available:
        return None

    return random.choice(available)


def mark_pool_content_used(state, content):
    """Mark a fallback content item as consumed."""
    content_id = content.get("id")

    if not content_id:
        return

    state.setdefault("used_content", [])

    if content_id not in state["used_content"]:
        state["used_content"].append(content_id)

    print(f"Using fallback content: {content_id}")

# ---------- stage 1: trends + topic ----------
def trend_hints():
    """Optional: recent popular kids video titles, used as THEME inspiration only."""
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        return []
    try:
        since = (datetime.datetime.utcnow() - datetime.timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
        r = requests.get("https://www.googleapis.com/youtube/v3/search", timeout=30, params={
            "part": "snippet", "q": "kids learning shorts", "type": "video", "order": "viewCount",
            "publishedAfter": since, "maxResults": 10, "key": key})
        return [i["snippet"]["title"] for i in r.json().get("items", [])]
    except Exception as e:
        print("trend lookup skipped:", e)
        return []


def pick_topic(state, hints):
    niches = CFG["niche_rotation"]
    niche = niches[state["day"] % len(niches)]

    pool = TOPICS.get(niche, []) + state["extra"].get(niche, [])
    unused = [t for t in pool if t not in state["used"]]

    if unused:
        return niche, unused[0], None

    # Normal topic list is exhausted.
    # Try Gemini first.
    try:
        new = gemini(
            f"Suggest 10 fresh, original video topics for a kids YouTube Shorts channel, "
            f"niche: {niche}, ages {CFG['target_age']}. "
            f"Already used: {state['used'][-40:]}. "
            f"Trending kids themes this week (inspiration only): {hints}. "
            "Return a JSON array of 10 short strings."
        )

        new = [
            t for t in new
            if isinstance(t, str) and t not in state["used"]
        ]

        if new:
            state["extra"].setdefault(niche, []).extend(new)
            return niche, new[0], None

    except RuntimeError as e:
        print("Gemini topic generation failed:", e)

    # Gemini failed.
    # Use the emergency content pool.
    fallback = get_unused_pool_content(niche, state)

    if fallback:
        print(
            f"Using content pool topic because Gemini topic generation failed: "
            f"{fallback.get('topic')}"
        )
        return niche, fallback["topic"], fallback

    raise RuntimeError(
        f"No unused topics and no unused content-pool content for niche '{niche}'."
    )


# ---------- stage 2: script ----------
def make_script(niche, topic, hints, state, fallback=None):
    # If pick_topic() already selected a fallback item,
    # use it directly instead of calling Gemini.
    if fallback is not None:
        print(
            f"Using ready-made content from pool: "
            f"{fallback.get('id', 'unknown')}"
        )
        return fallback, fallback

    n = CFG["niches"][niche]

    prompt = f"""You write ORIGINAL scripts for a YouTube Shorts channel for kids aged {CFG['target_age']}.
Niche: {niche}. Format: {n['style']}
Topic: {topic}
Trending themes (inspiration only, never copy): {hints}

Rules:
- Totally original. No existing characters, brands, or copyrighted songs.
- Safe, positive, simple words. Total narration 60 to 90 words.
- Exactly {n['scenes']} scenes. Scene 1 is a hook (a fun question or surprise).
- The last scene's final line should flow naturally back into the hook so the video loops.
- Each scene: narration (1 or 2 short sentences) and image_query (2 to 4 simple English words, a concrete photographable thing).

Return JSON: {{"title": "max 60 chars, keyword first", "description": "1 to 2 sentences",
"hashtags": ["4 tags without #"], "scenes": [{{"narration": "...", "image_query": "..."}}]}}"""

    try:
        return gemini(prompt), None

    except RuntimeError as e:
        print("Gemini script generation failed:", e)

        fallback = get_unused_pool_content(niche, state)

        if fallback:
            print(
                f"Falling back to content pool: "
                f"{fallback.get('id', 'unknown')}"
            )
            return fallback, fallback

        raise RuntimeError(
            f"Gemini failed and no unused content-pool content is available "
            f"for niche '{niche}'."
        )


# ---------- stage 3: voice with word timings ----------
async def tts(text, path):
    com = edge_tts.Communicate(text, CFG["voice"], rate=CFG["voice_rate"], boundary="WordBoundary")
    words = []
    with open(path, "wb") as f:
        async for c in com.stream():
            if c["type"] == "audio":
                f.write(c["data"])
            elif c["type"] == "WordBoundary":
                words.append((c["text"], c["offset"] / 1e7, (c["offset"] + c["duration"]) / 1e7))
    return words


# ---------- stage 4: visuals ----------
def get_image(query, path):
    """Try Pixabay (illustrations first, then photos), then Pexels. Returns None if nothing works."""
    pk = os.getenv("PIXABAY_API_KEY")
    if pk:
        for kind in ("illustration", "photo"):
            try:
                r = requests.get("https://pixabay.com/api/", timeout=30, params={
                    "key": pk, "q": query, "image_type": kind, "orientation": "vertical",
                    "safesearch": "true", "per_page": 10})
                hits = r.json().get("hits", [])
                if hits:
                    url = random.choice(hits[:6])["largeImageURL"]
                    open(path, "wb").write(requests.get(url, timeout=60).content)
                    return path
            except Exception as e:
                print("pixabay error:", e)
    key = os.getenv("PEXELS_API_KEY")
    if key:
        try:
            r = requests.get("https://api.pexels.com/v1/search", timeout=30,
                             headers={"Authorization": key},
                             params={"query": query, "orientation": "portrait", "per_page": 8})
            photos = r.json().get("photos", [])
            if photos:
                url = random.choice(photos[:5])["src"]["large2x"]
                open(path, "wb").write(requests.get(url, timeout=60).content)
                return path
        except Exception as e:
            print("pexels error:", e)
    return None


# ---------- stage 5: render ----------
def render_clip(img, dur, out, idx):
    frames = int(dur * FPS) + 1
    if img:
        zoom = "min(1+0.0006*on,1.2)" if idx % 2 == 0 else "max(1.2-0.0006*on,1)"
        vf = (f"scale=1296:2304:force_original_aspect_ratio=increase,crop=1296:2304,"
              f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS},"
              "format=yuv420p")
        cmd = ["ffmpeg", "-y", "-i", str(img), "-vf", vf, "-frames:v", str(frames)]
    else:  # no image -> bright solid colour so the video still renders
        colour = random.choice(["0x4aa3ff", "0xff8fab", "0x7bd88f", "0xffd166", "0xb28dff"])
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s={W}x{H}:r={FPS}",
               "-frames:v", str(frames), "-pix_fmt", "yuv420p"]
    sh(cmd + ["-c:v", "libx264", "-preset", "veryfast", "-r", str(FPS), str(out)])


def ass_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(words, path):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,DejaVu Sans,86,&H0000FFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,9,3,2,60,60,520,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = []
    chunks = [words[i:i + 3] for i in range(0, len(words), 3)]
    for i, ch in enumerate(chunks):
        start = ch[0][1]
        end = chunks[i + 1][0][1] if i + 1 < len(chunks) else ch[-1][2] + 0.3
        text = " ".join(w[0] for w in ch).upper().replace("\n", " ")
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}")
    open(path, "w", encoding="utf-8").write(head + "\n".join(lines) + "\n")


# ---------- main ----------
def main():
    out = ROOT / "output" / datetime.date.today().isoformat()
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    state = load_state()
    hints = trend_hints()

    niche, topic, fallback = pick_topic(state, hints)

    print(
        f"Day {state['day'] + 1}: "
        f"niche={niche} topic={topic}"
    )

    script, used_pool_content = make_script(
        niche,
        topic,
        hints,
        state,
        fallback
    )

    # If a fallback item was used, make sure the metadata topic
    # matches the actual content from the pool.
    if used_pool_content:
        topic = used_pool_content["topic"]
        mark_pool_content_used(state, used_pool_content)

    json.dump(
        script,
        open(out / "script.json", "w", encoding="utf-8"),
        indent=2
    )

    all_words, clips, wavs, offset = [], [], [], 0.0
    for i, scene in enumerate(script["scenes"]):
        mp3, wav, img, clip = tmp / f"s{i}.mp3", tmp / f"s{i}.wav", tmp / f"s{i}.jpg", tmp / f"c{i}.mp4"
        words = asyncio.run(tts(scene["narration"], mp3))
        sh(["ffmpeg", "-y", "-i", str(mp3), "-af", "apad=pad_dur=0.3", "-ar", "44100", "-ac", "1", str(wav)])
        dur = duration(wav)
        all_words += [(t, s + offset, e + offset) for t, s, e in words]
        render_clip(get_image(scene["image_query"], img), dur, clip, i)
        clips.append(clip)
        wavs.append(wav)
        offset += dur
    print(f"Total length: {offset:.1f}s")

    open(tmp / "clips.txt", "w").write("".join(f"file '{c.name}'\n" for c in clips))
    open(tmp / "wavs.txt", "w").write("".join(f"file '{w.name}'\n" for w in wavs))
    build_ass(all_words, tmp / "captions.ass")

    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "clips.txt",
        "-f", "concat", "-safe", "0", "-i", "wavs.txt",
        "-vf", "ass=captions.ass", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-shortest", "../video.mp4"], cwd=tmp)

    tags = " ".join("#" + t.lstrip("#") for t in script["hashtags"])
    meta = {"title": script["title"][:70],
            "description": f"{script['description']}\n\n{tags} #shorts #kids",
            "made_for_kids": True, "niche": niche, "topic": topic}
    json.dump(meta, open(out / "meta.json", "w"), indent=2)

    # clean temp files, update state
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    state["used"].append(topic)
    state["day"] += 1
    save_state(state)
    print("Done:", out / "video.mp4")


if __name__ == "__main__":
    main()
