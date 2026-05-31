import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return x_train, y_train, y_train_cat, x_test, y_test, y_test_cat


# ============================================================
# BUILD CNN MODEL
# ============================================================

def build_cnn_model(learning_rate=0.001, dropout_rate=0.3, dense_units=64):
    model = Sequential([
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(28, 28, 1)
        ),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu"
        ),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),

        Dense(10, activation="softmax")
    ])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(epochs, batch_size, learning_rate, dropout_rate, dense_units):
    x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = load_mnist_data()

    model = build_cnn_model(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test_cat,
        verbose=0
    )

    return model, history.history, test_loss, test_accuracy


# ============================================================
# FEEDBACK CACHE
# ============================================================

def store_feedback(image_array, predicted, correct=None):
    if "feedback_cache" not in st.session_state:
        st.session_state.feedback_cache = []

    feedback_entry = {
        "predicted": predicted,
        "correct_number": correct if correct is not None else predicted,
        "is_correct": predicted == correct,
        "image_data": image_array
    }

    st.session_state.feedback_cache.append(feedback_entry)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    [
        "Number Recognizer",
        "Model Performance",
        "Dataset Viewer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.title("Hyperparameters")

epochs = st.sidebar.selectbox(
    "Epochs",
    options=[3, 5, 10],
    index=1
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    options=[32, 64, 128],
    index=1
)

learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    options=[0.001, 0.0005, 0.0001],
    index=0
)

dropout_rate = st.sidebar.selectbox(
    "Dropout Rate",
    options=[0.2, 0.3, 0.5],
    index=1
)

dense_units = st.sidebar.selectbox(
    "Dense Layer Units",
    options=[64, 128, 256],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("Creator")
st.sidebar.markdown(
    """
**Jonathan Wong**  
Data Science  

[LinkedIn](https://www.linkedin.com/in/jonathan-wong-2b9b39233/)  
[GitHub](https://github.com/Excitedicecream)
"""
)

st.sidebar.info(
    """
Changing hyperparameters will retrain the model.

Recommended first setting:
- Epochs: 5
- Batch size: 64
- Learning rate: 0.001
- Dropout: 0.3
- Dense units: 64
"""
)


# ============================================================
# TRAIN MODEL BASED ON SIDEBAR SETTINGS
# ============================================================

with st.spinner("Training/loading CNN model..."):
    model, history, test_loss, test_accuracy = train_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )

x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = load_mnist_data()


# ============================================================
# PAGE 1: NUMBER RECOGNIZER
# ============================================================

if page == "Number Recognizer":

    st.title("Deep Learning Handwritten Digit Recognizer")
    st.write(
        """
        Draw a digit from **0 to 9** below.  
        The app uses a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.
        """
    )

    st.write("### Current Model Settings")

    settings_df = pd.DataFrame({
        "Hyperparameter": [
            "Epochs",
            "Batch Size",
            "Learning Rate",
            "Dropout Rate",
            "Dense Units"
        ],
        "Selected Value": [
            epochs,
            batch_size,
            learning_rate,
            dropout_rate,
            dense_units
        ]
    })

    st.table(settings_df)

    st.write("### Model Test Accuracy")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Test Accuracy",
            value=f"{test_accuracy:.4f}"
        )

    with col2:
        st.metric(
            label="Test Loss",
            value=f"{test_loss:.4f}"
        )

    st.markdown("---")

    st.write("### Draw a Digit")

    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Recognize Number"):

        if canvas_result.image_data is not None:
            img = Image.fromarray(
                canvas_result.image_data.astype("uint8"),
                "RGBA"
            )

            img = img.convert("L").resize((28, 28))
            img_array = np.array(img).reshape(1, 28, 28, 1).astype("float32") / 255.0

            prediction_probs = model.predict(img_array, verbose=0)[0]
            prediction = np.argmax(prediction_probs)
            confidence = prediction_probs[prediction]

            st.success(f"Predicted Number: **{prediction}**")
            st.write(f"Prediction Confidence: **{confidence:.4f}**")

            prob_df = pd.DataFrame({
                "Digit": list(range(10)),
                "Probability": prediction_probs
            })

            st.write("### Prediction Probability for Each Digit")
            st.bar_chart(prob_df.set_index("Digit"))

            st.markdown("---")
            st.write("Was this prediction correct?")

            if "feedback_submitted" not in st.session_state:
                st.session_state.feedback_submitted = False

            col_yes, col_no = st.columns(2)

            with col_yes:
                if st.button("Yes", key="yes_btn") and not st.session_state.feedback_submitted:
                    store_feedback(canvas_result.image_data, prediction, correct=prediction)
                    st.success("Feedback saved temporarily.")
                    st.session_state.feedback_submitted = True

            with col_no:
                if st.button("No", key="no_btn") and not st.session_state.feedback_submitted:
                    st.session_state.show_correction = True

            if st.session_state.get("show_correction", False) and not st.session_state.feedback_submitted:
                correct_num = st.number_input(
                    "Enter the correct number:",
                    min_value=0,
                    max_value=9,
                    step=1,
                    key="correct_input"
                )

                if st.button("Submit Correction", key="submit_correction"):
                    store_feedback(canvas_result.image_data, prediction, correct=correct_num)
                    st.info(f"Feedback saved. Correct number: {correct_num}")
                    st.session_state.feedback_submitted = True


# ============================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("Model Performance and Hyperparameter Results")

    st.write(
        """
        This page shows the performance of the CNN model trained using the selected hyperparameters.
        """
    )

    st.write("### Selected Hyperparameters")

    settings_df = pd.DataFrame({
        "Hyperparameter": [
            "Epochs",
            "Batch Size",
            "Learning Rate",
            "Dropout Rate",
            "Dense Units"
        ],
        "Selected Value": [
            epochs,
            batch_size,
            learning_rate,
            dropout_rate,
            dense_units
        ]
    })

    st.table(settings_df)

    st.write("### Final Test Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Test Accuracy",
            value=f"{test_accuracy:.4f}"
        )

    with col2:
        st.metric(
            label="Test Loss",
            value=f"{test_loss:.4f}"
        )

    st.write("### Accuracy by Epoch")

    accuracy_df = pd.DataFrame({
        "Training Accuracy": history["accuracy"],
        "Validation Accuracy": history["val_accuracy"]
    })

    st.line_chart(accuracy_df)

    st.write("### Loss by Epoch")

    loss_df = pd.DataFrame({
        "Training Loss": history["loss"],
        "Validation Loss": history["val_loss"]
    })

    st.line_chart(loss_df)

    st.write("### Explanation")

    st.info(
        """
        The model uses a CNN architecture because MNIST consists of image data.
        The CNN learns spatial patterns such as edges, curves, and digit shapes.

        Hyperparameters used:
        - Epochs control how many times the model trains over the dataset.
        - Batch size controls how many samples are processed before updating model weights.
        - Learning rate controls how fast the model learns.
        - Dropout helps reduce overfitting.
        - Dense units control the number of neurons in the fully connected layer.

        Early stopping is applied by monitoring validation accuracy.
        If validation accuracy does not improve for 2 epochs, training stops early.
        """
    )


# ============================================================
# PAGE 3: DATASET VIEWER
# ============================================================

elif page == "Dataset Viewer":

    st.title("MNIST Dataset Viewer")

    st.write(
        """
        The MNIST dataset contains handwritten digit images from 0 to 9.
        Each image is grayscale and has a size of 28 by 28 pixels.
        """
    )

    st.write("### Dataset Shape")

    shape_df = pd.DataFrame({
        "Dataset": ["Training Images", "Testing Images"],
        "Shape": [str(x_train.shape), str(x_test.shape)]
    })

    st.table(shape_df)

    st.write("### Random MNIST Samples")

    num_samples = st.slider(
        "Select number of samples to view:",
        min_value=1,
        max_value=20,
        value=10
    )

    sample_indices = np.random.choice(
        len(x_train),
        num_samples,
        replace=False
    )

    cols = st.columns(5)

    for i, idx in enumerate(sample_indices):
        with cols[i % 5]:
            st.image(
                x_train[idx].reshape(28, 28),
                caption=f"Label: {y_train[idx]}",
                width=100
            )

    st.markdown("---")

    st.subheader("Stored Feedback in Memory")

    if "feedback_cache" in st.session_state and len(st.session_state.feedback_cache) > 0:
        feedback_df = pd.DataFrame([
            {
                "Predicted": item["predicted"],
                "Correct Number": item["correct_number"],
                "Is Correct": item["is_correct"]
            }
            for item in st.session_state.feedback_cache
        ])

        st.dataframe(feedback_df)

    else:
        st.write("No feedback stored yet.")

    if st.button("Clear Cache and Memory"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.clear()
        st.success("Cache and memory cleared. Please refresh the app.")