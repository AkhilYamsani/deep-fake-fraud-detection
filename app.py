import streamlit as st
import os
import json
import time
from datetime import datetime
from ai_inference import predict_image

# ---------------- FILE SYSTEM SETUP ----------------
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

USERS_FILE = "results/users.json"
HISTORY_FILE = "results/history.json"

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "verifying" not in st.session_state:
    st.session_state.verifying = False

# ---------------- AUTH PAGE ----------------
if not st.session_state.logged_in:
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("## 🔐 Cloud Verification Access")
        st.caption("Sign up if you are new, or log in to continue")
        st.markdown("---")

        tab1, tab2 = st.tabs(["📝 Sign Up", "🔑 Login"])

        # -------- SIGN UP --------
        with tab1:
            new_user = st.text_input("👤 Create Username", key="su_user")
            new_pass = st.text_input("🔑 Create Password", type="password", key="su_pass")

            if st.button("Create Account", use_container_width=True):
                with open(USERS_FILE, "r") as f:
                    users = json.load(f)

                if new_user in users:
                    st.error("❌ Username already exists")
                elif new_user == "" or new_pass == "":
                    st.warning("⚠️ Please fill all fields")
                else:
                    users[new_user] = new_pass
                    with open(USERS_FILE, "w") as f:
                        json.dump(users, f, indent=2)

                    st.success("✅ Account created successfully")
                    st.session_state.logged_in = True
                    st.session_state.current_user = new_user
                    st.session_state.verifying = True
                    st.rerun()

        # -------- LOGIN --------
        with tab2:
            user = st.text_input("👤 Username", key="li_user")
            passwd = st.text_input("🔑 Password", type="password", key="li_pass")

            if st.button("Login", use_container_width=True):
                with open(USERS_FILE, "r") as f:
                    users = json.load(f)

                if user in users and users[user] == passwd:
                    st.success("✅ Login successful")
                    st.session_state.logged_in = True
                    st.session_state.current_user = user
                    st.session_state.verifying = True
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")

    st.stop()

# ---------------- VERIFICATION LOADING ----------------
if st.session_state.verifying:
    st.title("☁️ Initializing Cloud Verification Service")

    with st.spinner("Authenticating user and loading AI engine..."):
        time.sleep(2)

    st.success("Cloud verification service is ready.")
    st.session_state.verifying = False
    st.rerun()

# ---------------- SIDEBAR ----------------
st.sidebar.caption(f"👤 Logged in as: {st.session_state.current_user}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.rerun()

st.sidebar.subheader("📜 My Detection History")

with open(HISTORY_FILE, "r") as f:
    history = json.load(f)

user_history = [h for h in history if h["user"] == st.session_state.current_user]

if user_history:
    for item in reversed(user_history[-5:]):
        st.sidebar.write(
            f"{item['timestamp']} | {item['filename']} → "
            f"{item['prediction']} ({item['confidence']*100:.1f}%)"
        )
else:
    st.sidebar.caption("No detections yet")

# -------- CLEAR HISTORY BUTTON --------
if st.sidebar.button("🗑️ Clear My History"):
    new_history = [h for h in history if h["user"] != st.session_state.current_user]
    with open(HISTORY_FILE, "w") as f:
        json.dump(new_history, f, indent=2)

    st.sidebar.success("History cleared")
    st.rerun()

# ---------------- MAIN APP ----------------
st.title("🕵️ Deepfake Fraud Detection System")
st.caption("Cloud-based AI verification platform")

uploaded_file = st.file_uploader(
    "Upload an image to verify",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    if st.button("Verify Media"):
        with st.spinner("Running AI verification..."):
            label, confidence = predict_image(file_path)

        st.subheader("🔍 Detection Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        record = {
            "user": st.session_state.current_user,
            "filename": uploaded_file.name,
            "prediction": label,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(HISTORY_FILE, "r+") as f:
            history = json.load(f)
            history.append(record)
            f.seek(0)
            json.dump(history, f, indent=2)

st.warning(
    "Prediction confidence may vary as this is a prototype system using a lightweight pre-trained AI model."
)