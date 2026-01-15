import streamlit as st
import joblib
from pathlib import Path
import re
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2

# -----------------------------
# Heuristic + ML claim scoring
# -----------------------------
def hoax_signal_score(text: str) -> float:
    t = text.lower()
    cues = 0

    # Strong scam/danger keywords
    high_risk_words = ["scam", "fraud", "hack", "virus", "urgent"]
    cues += sum(5 for w in high_risk_words if w in t)

    # Urgency / call to action
    urgency_words = ["share", "forward", "must", "warning", "alert"]
    cues += sum(3 for w in urgency_words if w in t)

    # Sensational words
    sensational = ["shocking", "exposed", "unbelievable", "mind-blowing", "secret"]
    cues += sum(2 for w in sensational if w in t)

    # Unverified / vague attribution
    unverified_patterns = [
        "claims that",
        "said this",
        "no source",
        "people say",
        "it is said",
        "reportedly"
    ]
    cues += sum(2 for p in unverified_patterns if p in t)

    # Punctuation / ALL CAPS
    if text.count("!") >= 2 or text.count("?") >= 2:
        cues += 2
    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        caps_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if caps_ratio > 0.45:
            cues += 1

    # Imperatives / clickbait
    imperatives = ["click", "watch", "read", "share now", "send this", "dont ignore", "don't ignore"]
    cues += sum(1 for w in imperatives if w in t)

    # Emoji / repeated symbols
    cues += 1 if len(re.findall(r"[\U0001F300-\U0001FAFF]", text)) >= 3 else 0
    cues += 1 if re.search(r"[!?]{4,}", text) else 0

    # Normalize more aggressively
    score = min(cues / 20.0, 1.0)

    # Force Medium if very suspicious keywords
    if any(w in t for w in high_risk_words) and score < 0.45:
        score = 0.45

    # If text mentions claims or sources but score is very low, force Medium
    if any(p in t for p in ["claims", "no source", "reportedly"]) and score < 0.25:
        score = 0.25

    return score


# -----------------------------
# Media scoring
# -----------------------------
def media_risk_placeholder(image: Image.Image) -> tuple[float, list[str]]:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    gx = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
    gy = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
    sharpness = float((gx + gy) / 2.0)
    sharp_norm = min(sharpness / 20.0, 1.0)
    score = float(1.0 - sharp_norm)
    reasons = []
    if score > 0.60:
        reasons.append("Image looks low-detail or blurry.")
    else:
        reasons.append("Image has reasonable detail.")
    return score, reasons


def media_bucket(score: float) -> str:
    if score >= 0.35:
        return "High"
    if score >= 0.15:
        return "Medium"
    return "Low"


def confidence_for_bucket(score: float, level: str) -> float:
    if level == "Low":
        return (1 - score) * 100
    if level == "High":
        return score * 100
    return (0.5 + abs(score - 0.5)) * 100

def teen_explanation(overall_level: str, claim_level: str, media_level: str) -> str:
    if overall_level == "High":
        return "🔴 High-risk! Pause and check carefully before sharing."
    if overall_level == "Medium":
        return "🟠 Medium-risk. Some warning signs detected; verify first."
    return "🟢 Low-risk. Looks safe, but always double-check!"

# -----------------------------
# Deepfake model
# -----------------------------
MEDIA_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"

@st.cache_resource
def load_media_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MEDIA_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MEDIA_MODEL_ID)
    model.eval()
    model.to(device)
    if device == "mps":
        model = model.half()
    return processor, model, device

def _fake_prob_from_outputs(model, probs: np.ndarray) -> float:
    id2label = getattr(model.config, "id2label", None) or {}
    fake_idx = None
    real_idx = None
    for idx, lab in id2label.items():
        lab_l = str(lab).lower()
        if "fake" in lab_l or "deepfake" in lab_l:
            fake_idx = int(idx)
        if "real" in lab_l or "realism" in lab_l:
            real_idx = int(idx)
    if fake_idx is not None:
        return float(probs[fake_idx])
    if real_idx is not None and len(probs) == 2:
        return float(1.0 - probs[real_idx])
    if real_idx is not None and len(probs) > 1:
        other_idxs = [i for i in range(len(probs)) if i != real_idx]
        return float(np.max(probs[other_idxs]))
    return float(np.max(probs))

def detect_faces_pil(image: Image.Image):
    img_rgb = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    crops = []
    h, w = img_rgb.shape[:2]
    for (x, y, fw, fh) in faces:
        pad = int(0.20 * max(fw, fh))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + pad)
        crop = Image.fromarray(img_rgb[y1:y2, x1:x2])
        crops.append(crop)
    return crops

def media_risk_v2(image: Image.Image):
    processor, model, device = load_media_model()
    face_crops = detect_faces_pil(image)
    images = face_crops if len(face_crops) > 0 else [image.convert("RGB")]
    images = images[:2]  # limit for speed
    with torch.inference_mode():
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if device == "mps" and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
    fake_probs = [_fake_prob_from_outputs(model, probs[i]) for i in range(probs.shape[0])]
    score = float(np.max(fake_probs))
    reasons = []
    if len(face_crops) > 0:
        reasons.append(f"Deepfake model run on {len(images)} face crop(s); using max fake probability.")
    else:
        reasons.append("No face detected; model run on full image.")
    reasons.append(f"Estimated P(fake): {score:.2f}")
    reasons.append(f"Device: {device}")
    return score, reasons

# -----------------------------
# VIDEO DEEPFAKE SUPPORT
# -----------------------------
def video_risk(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_scores = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= 10:
            break
        frame_count += 1
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        score, _ = media_risk_v2(img)
        frame_scores.append(score)
    cap.release()
    if frame_scores:
        return max(frame_scores)
    return 0.0

def bucket(score: float) -> str:
    if score >= 0.55:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"

@st.cache_resource
def load_claim_model():
    vec = joblib.load(Path("models/claim_vectorizer.joblib"))
    model = joblib.load(Path("models/claim_model.joblib"))
    return vec, model

# Load claim model
try:
    vec, model = load_claim_model()
    model_loaded = True
except Exception:
    model_loaded = False

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="VeriTeen", page_icon="🛡️", layout="centered")
st.title("🛡️ VeriTeen – Misinformation Checker")
st.caption("Friendly guidance for safe sharing 👀")
st.warning("⚠️ This tool gives hints, not full answers. Always double-check before sharing!")

# -----------------------------
# Step 1: Claim
# -----------------------------
st.subheader("Please Enter The Post Text ")
claim = st.text_area("What does the post say?", placeholder="Type the post text here…")

# -----------------------------
# Step 2: Image
# -----------------------------
st.subheader("Upload an image here:")
uploaded_image = st.file_uploader("Upload a picture", type=["png", "jpg", "jpeg"])
if uploaded_image:
    st.image(uploaded_image, caption="Your uploaded picture", use_column_width=True)

# -----------------------------
# Step 3: Video
# -----------------------------
st.subheader("Upload a short video :")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov"])
if uploaded_video:
    st.video(uploaded_video)
# -----------------------------
# Analyze Button
# -----------------------------
analyze_btn = st.button(
    "🔍 Check this post!",
    type="primary",
    disabled=(not claim.strip() and uploaded_image is None and uploaded_video is None)
)

if analyze_btn:
    with st.spinner("Analyzing…"):
        # CLAIM SCORE
        if claim.strip():
            heur_score = hoax_signal_score(claim)
            claim_score = heur_score
            claim_level = bucket(claim_score)
            claim_conf = confidence_for_bucket(claim_score, claim_level)
        else:
            claim_score = 0.0
            claim_level = "Low"
            claim_conf = 100.0

        # MEDIA SCORE
        media_score = 0.0
        media_reasons = ["No media uploaded."]
        if uploaded_image is not None:
            try:
                media_score, media_reasons = media_risk_v2(Image.open(uploaded_image))
            except Exception as e:
                media_score, media_reasons = 0.2, [f"Media model error, using fallback. {e}"]
        if uploaded_video is not None:
            try:
                video_score = video_risk(uploaded_video)
                media_score = max(media_score, video_score)
                media_reasons.append(f"Video risk estimated: {video_score*100:.1f}%")
            except Exception as e:
                media_reasons.append(f"Video model error: {e}")
        media_level = media_bucket(media_score)
        media_conf = confidence_for_bucket(media_score, media_level)

        # OVERALL RISK
        if uploaded_image or uploaded_video:
            overall_score = max(
                claim_score,
                media_score,
                0.35  # safety floor when media exists
                )
        else:
            overall_score = claim_score


        # Push obvious scams/high-risk words automatically
        high_risk_words = ["scam", "fraud", "hack", "virus", "urgent"]
        if claim.strip() and any(w in claim.lower() for w in high_risk_words):
            overall_score = max(overall_score, 0.7)

        overall_level = bucket(overall_score)
        overall_conf = confidence_for_bucket(overall_score, overall_level)

        # RESULTS
        st.subheader("Step 4: Results")
        st.markdown(f"### Overall Risk: {overall_level} ({overall_score*100:.0f}%)")
        st.write(teen_explanation(overall_level, claim_level, media_level))

        st.markdown("### Details")
        reasons = []
        if claim.strip():
            reasons.append(f"Claim score: {claim_score*100:.1f}%")
        if uploaded_image or uploaded_video:
            reasons.extend(media_reasons)
        with st.expander("Why it might be risky ❓"):
            for r in reasons:
                st.write(f"- {r}")

        if (uploaded_image or uploaded_video) and media_score < 0.35:
            st.warning(
                "⚠️ This image/video looks realistic. "
                "Real-looking media can be harder for AI to detect, "
                "so low confidence does NOT mean it is safe."
            )
        st.subheader("Step 5: Quick Checklist")
        checklist_items = [
            "✅ Check source reliability",
            "🔎 Search on trusted websites",
            "📅 Look at upload date & context",
            "⏸️ Pause if it feels shocking",
            "📸 Compare images/videos with other sources",
            "👨‍🏫 Ask a trusted adult if unsure"
        ]
        with st.expander("Verify-before-share tips"):
            for item in checklist_items:
                st.checkbox(item, value=False, disabled=True)

        st.info("Tip: VeriTeen gives hints, not final answers. Always verify!")
