import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# =========================================================
# CLIENT DEMO APP: Proof-of-Delivery + Location Validation
# Streamlit-Cloud-safe version:
# - NO use_container_width / use_column_width on buttons
# - st.image uses use_column_width (most compatible)
# =========================================================

# ---------- CONFIG ----------
st.set_page_config(page_title="Delivery Vision Validation (Client Demo)", layout="wide")

# ✅ Repo-friendly default (recommended):
# DEFAULT_VIDEO = Path("assets/demo_delivery_clip.mp4")
# ✅ Your current mounted path (works in your local/container):
DEFAULT_VIDEO = Path("assets/demo_delivery_clip.mp4")

# ---------- HELPERS ----------
def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = frame_count / fps if fps else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": float(fps), "frames": frame_count, "duration_s": float(duration_s), "w": w, "h": h}

def read_frame_at_time(video_path: Path, t_sec: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_idx = int(max(0.0, float(t_sec)) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def resize_for_compare(img, size=(320, 320)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def hist_similarity(a_rgb, b_rgb):
    """
    Fast, dependency-free similarity:
    - Convert to HSV
    - Compare 2D histogram correlation
    Score: [-1..1] (higher is more similar)
    """
    a = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2HSV)
    b = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2HSV)

    hist_a = cv2.calcHist([a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

def detect_face_crop(img_rgb):
    """
    Demo-grade face crop using Haar cascade (OpenCV built-in).
    If no face detected, returns None.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return img_rgb[y:y + h, x:x + w]

def safe_confidence(score, min_s=-0.2, max_s=1.0):
    s = max(min(float(score), max_s), min_s)
    return float((s - min_s) / (max_s - min_s) * 100.0)

def bytes_to_rgb(uploaded_file):
    if uploaded_file is None:
        return None
    data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def init_session():
    st.session_state.setdefault("delivery_frame", None)
    st.session_state.setdefault("location_frame", None)
    st.session_state.setdefault("delivery_time", None)
    st.session_state.setdefault("location_time", None)
    st.session_state.setdefault("validation_log", [])

init_session()

# ---------- HEADER ----------
st.title("📦 Proof-of-Delivery Vision + Location Validation (Client Demo)")
st.caption("Video → Capture → Validate (recipient + location) → GenBI-style KPIs. Offline, demo-safe logic.")

# ---------- INPUTS ----------
left, right = st.columns([1.35, 1])

with left:
    st.subheader("🎥 Delivery Footage")
    video_file = st.file_uploader(
        "Upload a delivery video (optional). If not uploaded, we use the default clip.",
        type=["mp4", "mov", "m4v"]
    )

    if video_file is not None:
        upload_path = Path("uploaded_delivery_video.mp4")
        upload_path.write_bytes(video_file.read())
        video_path = upload_path
    else:
        video_path = DEFAULT_VIDEO

    if not video_path.exists():
        st.error(f"❌ Video not found at: {video_path}")
        st.stop()

    st.video(str(video_path))

    info = get_video_info(video_path)
    if info is None:
        st.error("❌ Could not open video.")
        st.stop()

    st.markdown(
        f"""
        **Video Info**
        - Resolution: `{info['w']} x {info['h']}`
        - FPS: `{info['fps']:.2f}`
        - Duration: `{info['duration_s']:.1f}s`
        """
    )

with right:
    st.subheader("📸 Capture: Delivery Proof + Location Proof")

    t1 = st.slider(
        "Select timestamp for **Delivery Proof** frame (handover / doorstep moment)",
        0.0, float(info["duration_s"]),
        min(5.0, float(info["duration_s"]) / 4.0),
        0.1
    )
    t2 = st.slider(
        "Select timestamp for **Location Proof** frame (house / entrance / door view)",
        0.0, float(info["duration_s"]),
        min(8.0, float(info["duration_s"]) / 3.0),
        0.1
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Capture Delivery Proof Frame"):
            frame = read_frame_at_time(video_path, t1)
            if frame is None:
                st.error("Could not read frame. Try another timestamp.")
            else:
                st.session_state["delivery_frame"] = frame
                st.session_state["delivery_time"] = float(t1)

    with c2:
        if st.button("🏠 Capture Location Proof Frame"):
            frame = read_frame_at_time(video_path, t2)
            if frame is None:
                st.error("Could not read frame. Try another timestamp.")
            else:
                st.session_state["location_frame"] = frame
                st.session_state["location_time"] = float(t2)

    df = st.session_state["delivery_frame"]
    lf = st.session_state["location_frame"]

    if df is not None:
        st.markdown(f"**Delivery Proof (t={st.session_state['delivery_time']:.1f}s)**")
        st.image(df, use_column_width=True)

    if lf is not None:
        st.markdown(f"**Location Proof (t={st.session_state['location_time']:.1f}s)**")
        st.image(lf, use_column_width=True)

    st.markdown("---")
    st.subheader("🧾 Validation Inputs (Client-friendly)")

    ref_loc = st.file_uploader(
        "Upload **Reference Location Image** (expected house/door photo)",
        type=["png", "jpg", "jpeg"],
        key="ref_loc"
    )
    ref_face = st.file_uploader(
        "Upload **Reference Recipient Image** (expected recipient selfie/ID photo)",
        type=["png", "jpg", "jpeg"],
        key="ref_face"
    )

    st.markdown("**Claimed Delivery Metadata (simulated)**")
    claimed_address = st.text_input("Claimed address", value="Unit 12, Example Street, City")
    claimed_lat = st.text_input("Claimed latitude", value="25.2048")
    claimed_lon = st.text_input("Claimed longitude", value="55.2708")

    st.markdown("---")
    st.subheader("🧠 Validate Delivery (Right person • Right place • Right time)")

    validate_btn = st.button("🔍 Run Validation", type="primary")

    if validate_btn:
        if df is None or lf is None:
            st.error("Capture BOTH frames first (Delivery Proof + Location Proof).")
        else:
            # IMPORTANT: file_uploader streams are one-time reads.
            # Convert immediately here.
            ref_loc_rgb = bytes_to_rgb(ref_loc)
            ref_face_rgb = bytes_to_rgb(ref_face)

            # --- Location similarity ---
            loc_status, loc_conf = "NOT CHECKED", None
            if ref_loc_rgb is not None:
                a = resize_for_compare(lf)
                b = resize_for_compare(ref_loc_rgb)
                loc_score = hist_similarity(a, b)
                loc_conf = safe_confidence(loc_score)
                loc_status = "PASS" if loc_conf >= 70 else "REVIEW"

            # --- Person similarity (demo-grade) ---
            person_status, person_conf = "NOT CHECKED", None
            df_face = None
            ref_face_crop = None

            if ref_face_rgb is not None:
                df_face = detect_face_crop(df)
                ref_face_crop = detect_face_crop(ref_face_rgb)

                if df_face is None or ref_face_crop is None:
                    person_status = "REVIEW"
                    person_conf = 40.0
                else:
                    a = resize_for_compare(df_face, (256, 256))
                    b = resize_for_compare(ref_face_crop, (256, 256))
                    person_score = hist_similarity(a, b)
                    person_conf = safe_confidence(person_score)
                    person_status = "PASS" if person_conf >= 72 else "REVIEW"

            # --- Time plausibility (simple demo rule) ---
            delivery_time = float(st.session_state["delivery_time"] or 0.0)
            location_time = float(st.session_state["location_time"] or 0.0)
            time_gap = abs(location_time - delivery_time)
            time_status = "PASS" if time_gap <= 10.0 else "REVIEW"
            time_conf = 90.0 if time_status == "PASS" else 55.0

            # --- Overall decision ---
            checks = []
            if loc_status != "NOT CHECKED":
                checks.append(loc_status)
            if person_status != "NOT CHECKED":
                checks.append(person_status)
            checks.append(time_status)

            overall = "PASS" if all(c == "PASS" for c in checks) else "REVIEW"

            record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "overall": overall,
                "location_status": loc_status,
                "location_conf": loc_conf,
                "person_status": person_status,
                "person_conf": person_conf,
                "time_status": time_status,
                "time_conf": time_conf,
                "claimed_address": claimed_address,
                "claimed_lat": claimed_lat,
                "claimed_lon": claimed_lon,
                "delivery_t": round(delivery_time, 2),
                "location_t": round(location_time, 2),
            }
            st.session_state["validation_log"].append(record)

            st.markdown("### ✅ Validation Result")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Overall", overall)
            k2.metric("Location", loc_status, f"{(loc_conf or 0):.0f}%")
            k3.metric("Recipient", person_status, f"{(person_conf or 0):.0f}%")
            k4.metric("Time Plausibility", time_status, f"{time_conf:.0f}%")

            with st.expander("Show evidence frames"):
                e1, e2 = st.columns(2)
                with e1:
                    st.caption("Delivery Proof Frame")
                    st.image(df, use_column_width=True)
                    if df_face is not None:
                        st.caption("Detected Face Crop (Delivery)")
                        st.image(df_face, use_column_width=True)
                with e2:
                    st.caption("Location Proof Frame")
                    st.image(lf, use_column_width=True)
                    if ref_loc_rgb is not None:
                        st.caption("Reference Location Image")
                        st.image(ref_loc_rgb, use_column_width=True)
                    if ref_face_rgb is not None:
                        st.caption("Reference Recipient Image")
                        st.image(ref_face_rgb, use_column_width=True)
                        if ref_face_crop is not None:
                            st.caption("Detected Face Crop (Reference)")
                            st.image(ref_face_crop, use_column_width=True)

            st.info(
                "Demo note: similarity checks are placeholders (offline). "
                "Production: face embeddings + liveness + geofencing + OCR/barcode + device attestation."
            )

# =========================================================
# GENBI-STYLE ANALYTICS + CHAT
# =========================================================
st.markdown("---")
st.subheader("📊 GenBI: Delivery Validation Analytics")

log = st.session_state["validation_log"]

if len(log) == 0:
    st.warning("No validations run yet. Capture frames and click **Run Validation** to populate analytics.")
else:
    total = len(log)
    passed = sum(1 for r in log if r["overall"] == "PASS")
    review = total - passed

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Deliveries Validated", total)
    c2.metric("Pass", passed, f"{(passed/total*100):.0f}%")
    c3.metric("Needs Review", review, f"{(review/total*100):.0f}%")

    pass_rate = []
    p = 0
    for i, r in enumerate(log, start=1):
        if r["overall"] == "PASS":
            p += 1
        pass_rate.append(p / i * 100.0)

    fig, ax = plt.subplots()
    ax.plot(range(1, total + 1), pass_rate, marker="o")
    ax.set_xlabel("Validation Run #")
    ax.set_ylabel("Cumulative Pass Rate (%)")
    ax.set_title("Proof-of-Delivery Validation Trend")
    st.pyplot(fig)

    st.markdown("**Recent Validation Events**")
    for r in log[-5:][::-1]:
        st.write(
            f"- `{r['ts']}` • **{r['overall']}** • "
            f"Location: {r['location_status']} ({(r['location_conf'] or 0):.0f}%) • "
            f"Recipient: {r['person_status']} ({(r['person_conf'] or 0):.0f}%) • "
            f"Time: {r['time_status']} • "
            f"Addr: {r['claimed_address']}"
        )

st.markdown("---")
st.subheader("💬 Ask GenBI (Offline)")
query = st.chat_input("Ask about validation outcomes… e.g., 'How many needed review?' or 'Why review?'")

if query:
    st.chat_message("user").write(query)
    q = query.lower().strip()

    if len(log) == 0:
        ans = "No validation events yet. Run at least one validation."
    elif "how many" in q and ("review" in q or "fail" in q):
        total = len(log)
        passed = sum(1 for r in log if r["overall"] == "PASS")
        review = total - passed
        ans = f"{review} out of {total} validations are marked **REVIEW**."
    elif "pass rate" in q or "success" in q:
        total = len(log)
        passed = sum(1 for r in log if r["overall"] == "PASS")
        ans = f"Current pass rate is **{(passed/total*100):.0f}%** ({passed}/{total})."
    elif "why" in q and "review" in q:
        drivers = {"Location": 0, "Recipient": 0, "Time": 0}
        for r in log:
            if r["overall"] == "REVIEW":
                if r["location_status"] == "REVIEW":
                    drivers["Location"] += 1
                if r["person_status"] == "REVIEW":
                    drivers["Recipient"] += 1
                if r["time_status"] == "REVIEW":
                    drivers["Time"] += 1
        top = max(drivers, key=drivers.get)
        ans = f"Most common review driver is **{top}**. Breakdown: {drivers}."
    else:
        ans = (
            "Offline mode — try:\n"
            "- How many needed review?\n"
            "- What is the pass rate?\n"
            "- Why review?\n"
            "- Show recent validation events"
        )

    st.chat_message("assistant").write(ans)
