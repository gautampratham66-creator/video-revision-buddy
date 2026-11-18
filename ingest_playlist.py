import os
import time
import uuid
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType, Filter, FieldCondition, MatchValue

load_dotenv()

# ---------------- put your links here ----------------
# Playlist links and single video links can be mixed.
SOURCES = [
    "PASTE_YOUR_PLAYLIST_LINK_HERE",
]
MAX_VIDEOS = 10          # safety limit per run
# -----------------------------------------------------

COLLECTION_NAME = "video_notes_ml"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SECONDS = 75
EMBEDDING_SIZE = 384
MAX_AUDIO_MB = 24        # Groq free limit is about 25 MB per file

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embedding_model = TextEmbedding(MODEL_NAME)

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )
for field in ("video_id", "playlist"):
    client.create_payload_index(COLLECTION_NAME, field, PayloadSchemaType.KEYWORD)


# ---------- 1. turn the links into a list of videos ----------
def list_videos(sources):
    videos = []
    for url in sources:
        with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        if info.get("entries"):                       # it is a playlist
            name = info.get("title", "Playlist")
            for e in info["entries"]:
                if e and e.get("id"):
                    videos.append((e["id"], name))
        else:                                         # it is one video
            videos.append((info["id"], "Single videos"))
    return videos


def already_done(video_id):
    points, _ = client.scroll(
        COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]),
        limit=1,
    )
    return len(points) > 0


# ---------- 2. download a small audio file ----------
def download_audio(video_id):
    options = {
        "format": "bestaudio[filesize<24M]/bestaudio[filesize_approx<24M]/worstaudio",
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        return info["title"], ydl.prepare_filename(info)


# ---------- 3. audio -> text with timestamps ----------
def transcribe(audio_path):
    with open(audio_path, "rb") as f:
        response = groq_client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    result = []
    for seg in response.segments:
        if isinstance(seg, dict):
            start, end, text = seg["start"], seg["end"], seg["text"]
        else:
            start, end, text = seg.start, seg.end, seg.text
        result.append({"start": start, "end": end, "text": text.strip()})
    return result


# ---------- 4. group segments into chunks ----------
def make_chunks(segments):
    groups, current = [], []
    for seg in segments:
        if current and seg["end"] - current[0]["start"] > CHUNK_SECONDS:
            groups.append(current)
            current = []
        current.append(seg)
    if current:
        groups.append(current)

    return [{
        "start": g[0]["start"],
        "end": g[-1]["end"],
        "text": " ".join(s["text"] for s in g),
        "segments": [{"start": round(s["start"], 1), "text": s["text"]} for s in g],
    } for g in groups]


# ---------- 5. store in Qdrant ----------
def store(video_id, title, playlist, chunks):
    embeddings = list(embedding_model.embed([c["text"] for c in chunks]))
    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}-{i}")),
            vector=embeddings[i].tolist(),
            payload={
                "text": chunk["text"],
                "start": chunk["start"],
                "end": chunk["end"],
                "segments": chunk["segments"],
                "video_id": video_id,
                "title": title,
                "playlist": playlist,
            },
        ))
    client.upsert(COLLECTION_NAME, points=points)
    return len(points)


# ---------- run ----------
videos = list_videos(SOURCES)[:MAX_VIDEOS]
print(f"Found {len(videos)} videos")

done, skipped, failed = 0, 0, 0
for number, (video_id, playlist) in enumerate(videos, start=1):
    print(f"\n[{number}/{len(videos)}] {video_id}")

    if already_done(video_id):
        print("  already stored, skipping")
        skipped += 1
        continue

    try:
        title, audio_path = download_audio(video_id)
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"  {title} ({size_mb:.1f} MB)")

        if size_mb > MAX_AUDIO_MB:
            print("  too long for the free Groq limit, skipping")
            skipped += 1
            continue

        segments = transcribe(audio_path)
        chunks = make_chunks(segments)
        count = store(video_id, title, playlist, chunks)
        print(f"  stored {count} chunks")
        done += 1
        time.sleep(5)                       # be gentle with the rate limit

    except Exception as error:
        print("  failed:", str(error)[:200])
        failed += 1

print(f"\nDone: {done} stored, {skipped} skipped, {failed} failed")
print("If some failed, wait a few minutes and run the file again. Finished videos are skipped.")
