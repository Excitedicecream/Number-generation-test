import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ===============================
# MODEL CREATION & DATA CACHING
# ===============================
@st.cache_resource
def load_model_and_data():
    """Create and train a simple CNN model using the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train_cat, epochs=1, batch_size=128, verbose=0, validation_data=(x_test, y_test_cat))
    return model, x_train, y_train

# ===============================
# HELPER FUNCTION TO STORE FEEDBACK
# ===============================
def store_feedback(image_array, predicted, correct=None):
    """Store prediction feedback in session cache."""
    if "feedback_cache" not in st.session_state:
        st.session_state.feedback_cache = []

    feedback_entry = {
        "predicted": predicted,
        "correct_number": correct if correct is not None else predicted,
        "is_correct": predicted == correct,
        "image_data": image_array
    }
    st.session_state.feedback_cache.append(feedback_entry)

# ===============================
# APP CONFIGURATION
# ===============================
st.set_page_config(page_title="🧠 Keras Number Recognizer", page_icon="🤖", layout="centered")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select a Page", ["Number Recognizer", "Cached Dataset Viewer"])

# Sidebar Author Info
st.sidebar.markdown("---")
st.sidebar.header("👤 About the Creator")
st.sidebar.markdown(
    """
**Jonathan Wong Tze Syuen**  
📚 Data Science  

🔗 [LinkedIn](https://www.linkedin.com/in/jonathan-wong-2b9b39233/)  
🔗 [GitHub](https://github.com/Excitedicecream)
"""
)

# ===============================
# PAGE 1: Number Recognizer
# ===============================
if page == "Number Recognizer":
    st.title("🧠 Deep Learning Number Recognizer")
    st.markdown("""
    Draw a **digit (0–9)** below.  
    This app uses a **Keras CNN model** to recognize handwritten numbers  
    and caches both the model and dataset for faster use.
    """)

    # Drawing Canvas
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("🔍 Recognize Number"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            img = img.convert('L').resize((28, 28))
            img_array = np.array(img).reshape(1, 28, 28, 1).astype("float32") / 255.0

            model, _, _ = load_model_and_data()
            prediction = np.argmax(model.predict(img_array), axis=1)[0]
            st.success(f"✅ Predicted number: **{prediction}**")

            st.markdown("---")
            st.write("Was this prediction correct?")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Yes"):
                    store_feedback(canvas_result.image_data, prediction, correct=prediction)
                    st.success("Thanks! Feedback saved temporarily in cache.")

            with col2:
                # Step 1: When user clicks "No", remember it in session state
                if st.button("❌ No"):
                    st.session_state.show_correction_input = True

                # Step 2: Show correction input only after clicking "No"
                if st.session_state.get("show_correction_input", False):
                    correct_num = st.number_input("Enter the correct number:", 0, 9, key="correct_input")
                    
                    if st.button("Submit Correction"):
                        # Append feedback to cache
                        st.session_state.feedback_cache.append({
                            "image_data": canvas_result.image_data,
                            "predicted": prediction,
                            "correct_number": correct_num
                        })
                        st.info(f"✅ Feedback saved. Correct number: {correct_num}")
                        st.session_state.show_correction_input = False  # Reset for next time
        else:
            st.error("Please draw a number first!")

# ===============================
# PAGE 2: Cached Dataset Viewer
# ===============================
elif page == "Cached Dataset Viewer":
    st.title("📊 Cached MNIST Dataset Viewer")
    st.markdown("This page shows random samples from the **cached MNIST dataset** and any feedback stored in memory.")

    model, x_train, y_train = load_model_and_data()

    st.subheader("📁 Random MNIST Samples")
    num_samples = st.slider("Select number of samples to view:", 1, 10, 5)
    sample_indices = np.random.choice(len(x_train), num_samples, replace=False)

    for idx in sample_indices:
        st.image(x_train[idx].reshape(28, 28), caption=f"Label: {y_train[idx]}", width=100)

    st.markdown("---")
    st.subheader("🧠 User Feedback in Cache")

    if "feedback_cache" in st.session_state and len(st.session_state.feedback_cache) > 0:
        feedback_df = pd.DataFrame(st.session_state.feedback_cache)

        # Split correct and incorrect predictions
        correct_feedback = feedback_df[feedback_df["is_correct"] == True]
        wrong_feedback = feedback_df[feedback_df["is_correct"] == False]

        # --- Correct Predictions ---
        st.markdown("### 🟩 Correct Predictions")
        if not correct_feedback.empty:
            for i, fb in correct_feedback.iterrows():
                if "image_data" in fb:
                    st.image(fb["image_data"], caption=f"✅ Predicted & Correct: {fb['predicted']}", width=120)
                else:
                    st.write(f"✅ Predicted & Correct: {fb['predicted']}")
        else:
            st.info("No correct predictions yet.")

        # --- Wrong Predictions ---
        st.markdown("### 🟥 Wrong Predictions")
        if not wrong_feedback.empty:
            for i, fb in wrong_feedback.iterrows():
                if "image_data" in fb:
                    st.image(
                        fb["image_data"],
                        caption=f"❌ Predicted: {fb['predicted']} | ✅ Actual: {fb['correct_number']}",
                        width=120
                    )
                else:
                    st.write(f"❌ Predicted: {fb['predicted']} | ✅ Actual: {fb['correct_number']}")
        else:
            st.info("No wrong predictions yet.")
    else:
        st.info("No feedback saved yet.")

    if st.button("🧹 Clear Cache & Memory"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.success("✅ Cache and memory cleared.")
