import html
import re

import streamlit as st
from fastembed import TextEmbedding
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

st.set_page_config(page_title="Revision Buddy", page_icon="📌", layout="wide")

COLLECTION_NAME = "video_notes_ml"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NOT_KNOWN = "Apun ko iss bare mai kuch nahi pata"

# ---------------------------------------------------------------- design
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --paper:#F3F6FA; --ink:#13203A; --cobalt:#2B50F5; --mark:#FFD84D;
  --muted:#5C6B82; --line:#D9E1EC; --card:#FFFFFF;
}
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp textarea { font-family:'IBM Plex Sans', system-ui, sans-serif; }
[data-testid="stIconMaterial"] { font-family:'Material Symbols Rounded' !important; }
.stApp { background: var(--paper); color: var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }

h1.title {
  font-family:'Bricolage Grotesque', sans-serif; font-weight:800;
  font-size: clamp(2.2rem, 5vw, 3.6rem); line-height:1.02; letter-spacing:-0.02em;
  margin:0 0 .5rem 0; color:var(--ink);
}
p.sub { color:var(--muted); font-size:1.08rem; max-width:34rem; margin:0 0 1.6rem 0; line-height:1.55; }

/* search box */
div[data-testid="stTextInput"] input {
  font-size:1.12rem; padding:.95rem 1.1rem; border-radius:14px;
  border:2px solid var(--ink); background:var(--card); color:var(--ink);
}
div[data-testid="stTextInput"] input:focus { border-color:var(--cobalt); box-shadow:0 0 0 4px rgba(43,80,245,.15); }

/* buttons */
div[data-testid="stButton"] button {
  border-radius:999px; border:1.5px solid var(--ink); background:transparent;
  color:var(--ink); font-weight:500; padding:.35rem 1rem; transition:background .15s;
}
div[data-testid="stButton"] button:hover { background:var(--mark); border-color:var(--ink); color:var(--ink); }
div[data-testid="stButton"] button[kind="primary"] {
  background:var(--cobalt); border-color:var(--cobalt); color:#fff; padding:.6rem 1.6rem; font-weight:600;
}
div[data-testid="stButton"] button[kind="primary"]:hover { background:#1f3fd1; color:#fff; }

/* result: the moment */
.moment { display:flex; align-items:baseline; gap:1rem; flex-wrap:wrap; margin:1.8rem 0 .4rem 0; }
.moment .label { color:var(--muted); font-size:1rem; }
.moment .time {
  font-family:'Bricolage Grotesque', sans-serif; font-weight:800;
  font-size: clamp(3rem, 8vw, 5.2rem); line-height:1; letter-spacing:-0.03em;
  background: linear-gradient(transparent 58%, var(--mark) 58%, var(--mark) 92%, transparent 92%);
  padding:0 .15em;
}

/* timeline */
.track { position:relative; height:14px; background:#E3EAF3; border-radius:999px; margin:1rem 0 .35rem 0; }
.track .hit { position:absolute; top:0; height:14px; background:#9DB1F7; border-radius:999px; }
.track .pin {
  position:absolute; top:-7px; width:4px; height:28px; background:var(--ink);
  border-radius:2px; transform:translateX(-2px);
}
.track .pin::after {
  content:''; position:absolute; top:-6px; left:-5px; width:14px; height:14px;
  border-radius:50%; background:var(--mark); border:3px solid var(--ink);
}
.ticks { display:flex; justify-content:space-between; color:var(--muted); font-size:.85rem; }

/* cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius:16px !important; border:1.5px solid var(--line) !important; background:var(--card);
}
.card-title { font-family:'Bricolage Grotesque', sans-serif; font-weight:700; font-size:1.25rem; margin:0 0 .4rem 0; }
.notice {
  background:var(--card); border:2px dashed var(--ink); border-radius:16px;
  padding:1.4rem 1.6rem; margin-top:1.6rem; font-size:1.15rem;
}
.notice b { font-family:'Bricolage Grotesque', sans-serif; font-size:1.4rem; }
.meta { color:var(--muted); font-size:.9rem; }

@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- setup
@st.cache_resource(show_spinner="Loading the search model. The first start takes a minute.")
def load_model():
    return TextEmbedding(MODEL_NAME)


@st.cache_resource
def load_clients():
    qdrant = QdrantClient(url=st.secrets["QDRANT_URL"], api_key=st.secrets["QDRANT_API_KEY"])
    groq = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return qdrant, groq


model = load_model()
qdrant, groq_client = load_clients()


# ---------------------------------------------------------------- logic
def mmss(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def video_filter(video_id):
    if not video_id.strip():
        return None
    return Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id.strip()))])


def search(question, top_k, min_score, video_id):
    vector = list(model.embed([question]))[0].tolist()
    return qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=video_filter(video_id),
        limit=top_k,
        with_payload=True,
        score_threshold=min_score,
    ).points


@st.cache_data(ttl=3600)
def video_length(video_id):
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=video_filter(video_id),
        limit=200,
        with_payload=["end"],
    )
    return max(p.payload["end"] for p in points)


@st.cache_data(ttl=300)
def list_videos():
    seen, offset = {}, None
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME, limit=256, offset=offset,
            with_payload=["video_id", "title"],
        )
        for p in points:
            seen[p.payload["video_id"]] = p.payload.get("title", p.payload["video_id"])
        if offset is None:
            break
    return seen


def ask_llm(question, segments):
    context = ""
    for number, s in enumerate(segments, start=1):
        context += f"[{number}] ({mmss(s['start'])}) {s['text']}\n"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You help a student revise from a video transcript. "
                           "The context has numbered lines with timestamps. "
                           "The transcript is a Hindi and English mix (Hinglish) and has spelling mistakes. "
                           "Understand the meaning even when words are wrong. "
                           "Answer using only the context, in simple words. "
                           "Reply in this exact format:\n"
                           "START: <number of the line where the explanation of the question begins>\n"
                           "ANSWER: <your answer>\n"
                           "If the context has nothing related to the question, reply exactly: NOT_FOUND",
            },
            {"role": "user", "content": f"Context:\n{context}\nQuestion:\n{question}"},
        ],
    )
    return response.choices[0].message.content.strip()


def timeline_html(results, duration, moment):
    hits = ""
    for r in results:
        left = r.payload["start"] / duration * 100
        width = max((r.payload["end"] - r.payload["start"]) / duration * 100, 1)
        hits += f'<div class="hit" style="left:{left:.2f}%;width:{width:.2f}%"></div>'
    pin = f'<div class="pin" style="left:{moment / duration * 100:.2f}%"></div>'
    return (
        f'<div class="track">{hits}{pin}</div>'
        f'<div class="ticks"><span>00:00</span><span>{mmss(duration)}</span></div>'
    )


# ---------------------------------------------------------------- page
st.markdown(
    '<h1 class="title">Find the exact minute<br>your topic is explained.</h1>'
    '<p class="sub">Ask in English or Hinglish. You get a short answer and the lecture '
    "starts playing at the right second.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Parts of the video to read", 3, 17, 8)
    min_score = st.slider("Out-of-syllabus limit", 0.0, 0.8, 0.30, 0.05)
    videos = list_videos()
    choice = st.selectbox("Search in", ["All videos"] + sorted(videos.values()))
    video_id = "" if choice == "All videos" else next(v for v, t in videos.items() if t == choice)
    st.caption(f"{len(videos)} videos stored")
    st.caption("A higher limit rejects more questions as out of syllabus.")

examples = ["hashmap ki time complexity kya hai?", "what is a hash function?", "collision kya hota hai?"]

question = st.text_input(
    "Your question",
    key="q",
    placeholder="For example: hashmap internally kaise kaam karta hai?",
    label_visibility="collapsed",
)

row = st.columns([1.2, 3, 3, 3])
go = row[0].button("Find it", type="primary")
auto = False
for col, ex in zip(row[1:], examples):
    if col.button(ex, key=f"ex_{ex}", use_container_width=True):
        st.session_state["q"] = ex
        question = ex
        auto = True

if (go or auto) and question.strip():
    with st.spinner("Searching the lecture..."):
        results = search(question, top_k, min_score, video_id)
        reply = None
        segments = []
        if results:
            top_video = results[0].payload["video_id"]
            results = [r for r in results if r.payload["video_id"] == top_video]
            for r in sorted(results, key=lambda r: r.payload["start"]):
                for s in r.payload.get("segments", []):
                    segments.append({"start": s["start"], "text": s["text"], "chunk": r})
            reply = ask_llm(question, segments)

    if not results or reply is None or "NOT_FOUND" in reply:
        st.markdown(f'<div class="notice"><b>{NOT_KNOWN}</b><br>'
                    '<span class="meta">Try a topic that is taught in the video.</span></div>',
                    unsafe_allow_html=True)
        st.stop()

    chosen = None
    match = re.search(r"START:\s*(\d+)", reply)
    if match and 1 <= int(match.group(1)) <= len(segments):
        chosen = segments[int(match.group(1)) - 1]

    best = chosen["chunk"] if chosen else results[0]
    chunk = best.payload
    moment = max(0, int((chosen["start"] if chosen else chunk["start"])) - 2)
    answer = html.escape(reply.split("ANSWER:", 1)[-1].strip()).replace("\n", "<br>")

    st.markdown(
        f'<div class="moment"><span class="label">Your answer starts at</span>'
        f'<span class="time">{mmss(moment)}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(timeline_html(results, video_length(chunk["video_id"]), moment), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.video(f"https://www.youtube.com/watch?v={chunk['video_id']}", start_time=moment)
        st.markdown(f'<span class="meta">{html.escape(chunk.get("title", ""))}</span>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown('<p class="card-title">Short answer</p>', unsafe_allow_html=True)
            st.markdown(f"<div style='line-height:1.65'>{answer}</div>", unsafe_allow_html=True)
        with st.expander(f"Transcript of this part ({mmss(chunk['start'])} to {mmss(chunk['end'])})"):
            st.write(chunk["text"])
