import streamlit as st
import joblib
from pathlib import Path
import re
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
from datetime import datetime

# -----------------------------
# Heuristic + ML claim scoring
# -----------------------------

if "last_result" not in st.session_state:
    st.session_state.last_result = None

def hoax_signal_score(text: str) -> tuple[float, list[str]]:
    """
    Returns:
      score: 0.0–1.0
      cues: list of explainable triggers
    """
    if not text or not text.strip():
        return 0.0, ["No claim text provided."]

    t = text.strip()
    tl = t.lower()

    cues = []
    total = 8  # cue groups

    # 1) urgency / share cues
    urgency_words = ["urgent", "share", "forward", "must", "warning", "alert"]
    if any(w in tl for w in urgency_words):
        cues.append("Urgency/share language detected (e.g., urgent/share/warning).")

    # 2) conspiracy-ish phrasing
    conspiracy = ["they don't want you to know", "they dont want you to know", "cover up", "hidden truth"]
    if any(p in tl for p in conspiracy):
        cues.append("Conspiracy-style phrasing detected (e.g., 'they don't want you to know').")

    # 3) sensational adjectives
    sensational = ["shocking", "exposed", "unbelievable", "mind-blowing", "secret"]
    if any(w in tl for w in sensational):
        cues.append("Sensational wording detected (e.g., shocking/exposed/secret).")

    # 4) lots of exclamation/question marks
    ex = t.count("!")
    qu = t.count("?")
    if ex >= 3 or qu >= 3:
        cues.append(f"Excessive punctuation detected (!={ex}, ?={qu}).")

    # 5) ALL CAPS ratio
    letters = re.findall(r"[A-Za-z]", t)
    if letters:
        caps = sum(1 for ch in letters if ch.isupper())
        caps_ratio = caps / len(letters)
        if caps_ratio > 0.45:
            cues.append(f"High ALL-CAPS ratio detected ({caps_ratio*100:.0f}%).")

    # 6) commands / imperatives
    imperatives = ["click", "watch", "read", "share now", "send this", "dont ignore", "don't ignore"]
    if any(p in tl for p in imperatives):
        cues.append("Call-to-action / imperative detected (e.g., click/share now/don’t ignore).")

    # 7) too many emojis
    emoji_count = len(re.findall(r"[\U0001F300-\U0001FAFF]", t))
    if emoji_count >= 3:
        cues.append(f"Many emojis detected (count={emoji_count}).")

    # 8) excessive punctuation sequences
    if re.search(r"[!?]{4,}", t):
        cues.append("Repeated !!!/??? sequence detected (viral bait pattern).")

    score = len(cues) / total
    if len(cues) == 0:
        cues = ["No obvious hoax-style cues detected in the writing style."]

    return score, cues



@st.cache_resource
def load_claim_model():
    vec = joblib.load(Path("models/claim_vectorizer.joblib"))
    model = joblib.load(Path("models/claim_model.joblib"))
    return vec, model


# -----------------------------
# Media scoring (placeholder)
# -----------------------------
def media_risk_placeholder(image: Image.Image) -> tuple[float, list[str]]:
    """
    Placeholder media risk scoring (0.0–1.0).
    This is NOT deepfake detection yet. It just checks basic "quality cues"
    that often correlate with misleading reposts (heavy compression / blur).
    Later, replace with a real CV model.
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)

    # Simple sharpness estimate using gradients
    gx = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
    gy = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
    sharpness = float((gx + gy) / 2.0)

    # Normalize sharpness to a rough 0–1 scale (empirical)
    # low sharpness -> higher risk (blurry/low detail can hide manipulation)
    sharp_norm = min(sharpness / 20.0, 1.0)

    # Invert: low sharpness => higher risk
    score = float(1.0 - sharp_norm)

    reasons = []
    if score > 0.60:
        reasons.append("Image looks low-detail or blurry (harder to verify).")
    else:
        reasons.append("Image has reasonable detail (easier to visually verify).")

    return score, reasons


def bucket(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.45:
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
        return (
            "This looks **high-risk**. Don’t share yet. "
            "Pause and verify using the checklist below."
        )
    if overall_level == "Medium":
        return (
            "This looks **medium-risk**. Some warning signs are present. "
            "It might be misleading, so verify before sharing."
        )
    return (
        "This looks **low-risk** based on our checks, but misinformation can still happen. "
        "If it’s important, verify anyway."
    )

# -----------------------------
# Media model loading + inference
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
    """
    Tries to compute P(fake) robustly from model labels.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    # Find label indices
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

    # If we know "real", then fake = 1 - P(real) (binary-ish case)
    if real_idx is not None and len(probs) == 2:
        return float(1.0 - probs[real_idx])

    # Fallback: treat "not real" as fake-ish (choose max non-real index)
    if real_idx is not None and len(probs) > 1:
        other_idxs = [i for i in range(len(probs)) if i != real_idx]
        return float(np.max(probs[other_idxs]))

    # Last fallback: just take max probability as "risk"
    return float(np.max(probs))

def detect_faces_pil(image: Image.Image):
    """
    Returns list of PIL face crops using OpenCV Haar cascade (lightweight).
    """
    img_rgb = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    crops = []
    h, w = img_rgb.shape[:2]
    for (x, y, fw, fh) in faces:
        # add padding around face
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

    # SPEED: limit crops (group photos can be slow)
    images = images[:2]  # change to [:1] for fastest

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
        reasons.append("No face detected; deepfake model run on full image (less reliable for face-only deepfakes).")

    reasons.append(f"Estimated P(fake): {score:.2f}")
    reasons.append(f"Device: {device}")

    return score, reasons


def build_report(
    claim: str,
    claim_level: str,
    claim_score: float,
    ml_score: float,
    heur_score: float,
    heur_cues: list[str],
    media_level: str,
    media_score: float,
    media_reasons: list[str],
    overall_level: str,
    overall_score: float,
    overall_conf: float,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("VeriTeen — Analysis Report")
    lines.append(f"Timestamp: {ts}")
    lines.append("")
    lines.append("INPUT")
    lines.append(f"Claim text: {claim if claim and claim.strip() else '(none)'}")
    lines.append("")
    lines.append("RESULTS")
    lines.append(f"Overall: {overall_level} ({overall_score*100:.1f}%) | Confidence: {overall_conf:.1f}%")
    lines.append(f"Claim Risk: {claim_level} ({claim_score*100:.1f}%)")
    lines.append(f"  - ML credibility (LIAR): {ml_score:.2f}")
    lines.append(f"  - Hoax-signal (heuristics): {heur_score:.2f}")
    lines.append(f"Media Risk: {media_level} ({media_score*100:.1f}%)")
    lines.append("")
    lines.append("WHY (Explainable cues)")
    if heur_cues:
        lines.append("Hoax-signal cues:")
        for c in heur_cues:
            lines.append(f"- {c}")
    if media_reasons:
        lines.append("")
        lines.append("Media analysis notes:")
        for mr in media_reasons:
            lines.append(f"- {mr}")
    lines.append("")
    lines.append("VERIFY-BEFORE-SHARE CHECKLIST")
    checklist_items = [
        "Check the source: official account or random repost?",
        "Search the claim keywords on 2–3 trusted sites.",
        "Look for the original upload date and full context.",
        "If the post is urgent/emotional, pause — this is a common hoax tactic.",
        "If it shows a person/event, look for other reliable clips/photos of the same moment.",
        "Ask a trusted adult/teacher if you’re unsure.",
    ]
    for item in checklist_items:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("DISCLAIMER")
    lines.append("This tool estimates risk. It does not prove something is real or fake. Always verify using trusted sources.")
    return "\n".join(lines)


# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="VeriTeen", page_icon="🛡️", layout="centered")
st.title("🛡️ VeriTeen")
st.caption("Overall risk = Claim Risk + Media Risk. Guidance only — not a definitive verdict.")

st.warning(
    "Limitations: This tool estimates risk and can be wrong. "
    "Deepfake detectors may fail on low-quality images, heavy compression, screenshots, or new manipulation methods. "
    "Always verify using trusted sources."
)

# Try loading trained model files
try:
    _vec, _model = load_claim_model()
    model_loaded = True
except Exception:
    model_loaded = False

mode = st.radio("Mode", ["Teen", "Parent/Teacher"], horizontal=True)
use_deepfake = st.checkbox("Use deepfake detector (slower)", value=True)
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["png", "jpg", "jpeg"])
claim = st.text_input("Enter the post caption / claim (optional)")

image_obj = None
if uploaded_file is not None:
    image_obj = Image.open(uploaded_file)
    st.image(image_obj, caption="Uploaded image", use_column_width=True)

analyze_btn = st.button("Analyze", type="primary", disabled=(uploaded_file is None and not claim.strip()))

if analyze_btn:
    if not model_loaded:
        st.error("Claim model not loaded. Train it first by running train_claim_model.py")
    else:
        claim_score = 0.0
        claim_level = "Low"
        claim_conf = 0.0
        media_score = 0.0
        media_level = "Low"
        media_conf = 0.0
        overall_score = 0.0
        overall_level = "Low"
        overall_conf = 0.0
        with st.spinner("Analyzing…"):
            # -------------------
            # CLAIM RISK
            # -------------------
            ml_score = 0.10
            if claim and claim.strip():
                vec, model = load_claim_model()
                X = vec.transform([claim])
                ml_score = float(model.predict_proba(X)[0][1])  # P(suspicious)

            heur_score, heur_cues = hoax_signal_score(claim)
            claim_score = 0.35 * ml_score + 0.65 * heur_score

            claim_level = bucket(claim_score)
            claim_conf = confidence_for_bucket(claim_score, claim_level)

            # -------------------
            # MEDIA RISK
            # -------------------
            media_score = 0.10
            media_reasons = ["No image uploaded."]
            media_level = bucket(media_score)
            if image_obj is not None:
                try:
                    if use_deepfake:
                        media_score, media_reasons = media_risk_v2(image_obj)
                        media_level = bucket(media_score)
                    else:
                        media_score, media_reasons = media_risk_placeholder(image_obj)  # fast fallback
                        media_level = bucket(media_score)
                except Exception as e:
                    media_score, media_reasons = 0.20, [f"Media model error, using fallback score. Details: {e}"]
                    media_level = bucket(media_score)

            media_conf = confidence_for_bucket(media_score, media_level)

            # -------------------
            # OVERALL RISK (simple fusion)
            # -------------------
            overall_score = max(claim_score, media_score)
            overall_level = bucket(overall_score)
            overall_conf = confidence_for_bucket(overall_score, overall_level)

        # -------------------
        # RESULTS
        # -------------------
        st.subheader("Overall Risk")
        if overall_level == "Low":
            st.success(f"🟢 Low Risk (Confidence: {overall_conf:.1f}%)")
        elif overall_level == "Medium":
            st.warning(f"🟠 Medium Risk (Confidence: {overall_conf:.1f}%)")
        else:
            st.error(f"🔴 High Risk (Confidence: {overall_conf:.1f}%)")

        # -------------------
        # Why it may be risky (REPLACE YOUR WHOLE SECTION WITH THIS)
        # -------------------
        st.markdown("### Why it may be risky")

        reasons = []

        # Claim explanations
        if claim and claim.strip():
            reasons.append(f"• Claim risk = 0.35×ML + 0.65×Heuristics → {claim_score:.2f}")
            reasons.append(f"  - ML credibility score (LIAR-trained): {ml_score:.2f}")
            reasons.append(f"  - Hoax-signal score (writing style): {heur_score:.2f}")

            reasons.append(f"• Hoax-signal cues triggered ({len(heur_cues)}/8):")
            for c in heur_cues:
                reasons.append(f"  - {c}")
        else:
            reasons.append("• No claim text provided, so claim-based cues are skipped.")

        # Media explanations
        if image_obj is not None and media_reasons:
            reasons.append("• Media analysis notes:")
            for mr in media_reasons:
                reasons.append(f"  - {mr}")

        # Display
        with st.expander("Why it may be risky (details)", expanded=True):
            for r in reasons:
                st.write(r)

        st.subheader("Verify-before-share checklist")
        checklist_items = [
            "Check the source: official account or random repost?",
            "Search the claim keywords on 2–3 trusted sites.",
            "Look for the original upload date and full context.",
            "If the post is urgent/emotional, pause — this is a common hoax tactic.",
            "If it shows a person/event, look for other reliable clips/photos of the same moment.",
            "Ask a trusted adult/teacher if you're unsure.",
        ]
        for i, item in enumerate(checklist_items):
            st.checkbox(
                item,
                value=False,
                disabled=True,
                key=f"verify_checklist_{i}"
            )

        st.info("Tip: VeriTeen estimates risk. It doesn't prove something is real or fake. Always verify before sharing.")

        report_text = build_report(
            claim=claim,
            claim_level=claim_level,
            claim_score=claim_score,
            ml_score=ml_score,
            heur_score=heur_score,
            heur_cues=heur_cues,
            media_level=media_level,
            media_score=media_score,
            media_reasons=media_reasons,
            overall_level=overall_level,
            overall_score=overall_score,
            overall_conf=overall_conf,
        )

        st.download_button(
            label="⬇️ Download analysis report (TXT)",
            data=report_text,
            file_name="veriteen_report.txt",
            mime="text/plain",
        )
