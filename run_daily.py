"""Daily kids Shorts pipeline: pick niche+topic -> script -> voice -> visuals -> 9:16 video + captions."""
import asyncio, datetime, json, os, pathlib, random, subprocess, time

import edge_tts
import requests
import yaml

ROOT = pathlib.Path(__file__).parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
TOPICS = yaml.safe_load(open(ROOT / "topics.yaml", encoding="utf-8"))
STATE_F = ROOT / "data" / "state.json"
W, H, FPS = 1080, 1920, 30


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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CFG['gemini_model']}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.9}}
    for attempt in range(4):
        try:
            r = requests.post(url, json=body, timeout=90,
                              headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
            r.raise_for_status()
            return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            print("Gemini retry:", e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Gemini failed")


# ---------- state ----------
def load_state():
    if STATE_F.exists():
        return json.load(open(STATE_F))
    return {"day": 0, "used": [], "extra": {}}


def save_state(s):
    STATE_F.parent.mkdir(exist_ok=True)
    json.dump(s, open(STATE_F, "w"), indent=2)


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
    if not unused:  # list exhausted -> ask Gemini for fresh topics
        new = gemini(
            f"Suggest 10 fresh, original video topics for a kids YouTube Shorts channel, niche: {niche}, "
            f"ages {CFG['target_age']}. Already used: {state['used'][-40:]}. "
            f"Trending kids themes this week (inspiration only): {hints}. "
            "Return a JSON array of 10 short strings.")
        new = [t for t in new if isinstance(t, str) and t not in state["used"]]
        state["extra"].setdefault(niche, []).extend(new)
        unused = new
    return niche, unused[0]


# ---------- stage 2: script ----------
def make_script(niche, topic, hints):
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
    return gemini(prompt)


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
    niche, topic = pick_topic(state, hints)
    print(f"Day {state['day'] + 1}: niche={niche} topic={topic}")

    script = make_script(niche, topic, hints)
    json.dump(script, open(out / "script.json", "w"), indent=2)

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
