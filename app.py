import os
import json
from datetime import datetime
from textwrap import dedent
import base64


import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image

from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg16_preprocess
)

from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet50_preprocess
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Selection Using Deep Learning",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

JSON_DIR = os.path.join(
    PROJECT_ROOT,
    "json_data"
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "metrics"
)

MODEL_COMPARISON_JSON = os.path.join(
    METRICS_DIR,
    "model_comparison_summary.json"
)

ASSETS_DIR = os.path.join(
    PROJECT_ROOT,
    "assets"
)

PROFILE_IMAGE_PATH = os.path.join(
    ASSETS_DIR,
    "profile.png"
)


# ============================================================
# MODEL PATHS
# ============================================================

CNN_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "food_classification_cnn.keras"
)

VGG16_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "food_classification_vgg16.keras"
)

RESNET50_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "food_classification_resnet50.keras"
)

CLASSES_JSON = os.path.join(
    JSON_DIR,
    "classes.json"
)


# ============================================================
# HTML HELPER
# IMPORTANT:
# st.html() renders HTML correctly.
# This fixes the issue where <div> tags were showing as text.
# ============================================================

def render_html(html):

    st.html(
        dedent(html).strip()
    )

# ============================================================
# LOAD PROFILE IMAGE
# ============================================================

def get_profile_image():

    if not os.path.exists(
        PROFILE_IMAGE_PATH
    ):

        return None


    with open(
        PROFILE_IMAGE_PATH,
        "rb"
    ) as image_file:

        encoded_image = base64.b64encode(
            image_file.read()
        ).decode(
            "utf-8"
        )


    return encoded_image


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>

    .stApp {
        background-color: #f6f8fc;
    }


        .block-container {
        padding-top: 3.5rem;
        padding-bottom: 2rem;
    
        max-width: 1600px;
    }


    /* =======================================================
       HEADER
    ======================================================= */

        .main-title {
        text-align: center;
        font-size: 34px;
        font-weight: 750;
        color: #1f2937;
    
        line-height: 1.4;
    
        padding-top: 12px;
    
        margin-top: 0;
    
        margin-bottom: 5px;
    
        white-space: nowrap;
    }


    .subtitle {
        text-align: center;
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 20px;
    }


    /* =======================================================
       SECTION TITLE
    ======================================================= */

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 12px;
    }


    /* =======================================================
       INFORMATION CARD
    ======================================================= */

    .info-card {

        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;

        box-shadow:
            0px 2px 8px
            rgba(0, 0, 0, 0.03);

    }


    .info-label {

        font-size: 12px;
        font-weight: 500;
        color: #6b7280;
        margin-bottom: 5px;

    }


    .info-value {

        font-size: 15px;
        font-weight: 650;
        color: #1f2937;

    }


    /* =======================================================
       PROFILE CARD
    ======================================================= */

    .profile-card {

        background:
            linear-gradient(
                135deg,
                #ffffff,
                #f8fafc
            );

        border:
            1px solid #e5e7eb;

        border-radius:
            15px;

        padding:
            18px;

        margin-bottom:
            14px;

        text-align:
            center;

        box-shadow:
            0px 3px 12px
            rgba(0, 0, 0, 0.04);

    }


        .profile-avatar {
    
        width: 110px;
    
        height: 110px;
    
        border-radius: 50%;
    
        margin: auto;
    
        overflow: hidden;
    
        background: #eef2ff;
    
        border: 4px solid #eef2ff;
    
        box-shadow:
            0px 4px 15px
            rgba(0, 0, 0, 0.12);
    
        margin-bottom: 14px;
    
    }
    
    
    .profile-avatar img {
    
        width: 100%;
    
        height: 100%;
    
        object-fit: cover;
    
        object-position: center;
    
        display: block;
    
    }


    .profile-name {

        font-size:
            18px;

        font-weight:
            700;

        color:
            #1f2937;

    }


    .profile-role {

        font-size:
            13px;

        color:
            #6b7280;

        margin-top:
            4px;

    }


    /* =======================================================
       ARCHITECTURE TITLE
    ======================================================= */

    .architecture-title {

        font-size:
            14px;

        font-weight:
            600;

        color:
            #374151;

        margin-top:
            12px;

        margin-bottom:
            8px;

    }


    /* =======================================================
       MODEL COMPARISON
    ======================================================= */

    .comparison-label {

        font-size:
            13px;

        font-weight:
            600;

        color:
            #374151;

        margin-bottom:
            4px;

    }


    /* =======================================================
       SELECTED MODEL
    ======================================================= */

    .selected-model-box {

        background:
            #eef4ff;

        border-left:
            4px solid #2563eb;

        border-radius:
            8px;

        padding:
            13px 15px;

        margin-top:
            12px;

        margin-bottom:
            12px;

    }


    .selected-model-title {

        font-size:
            13px;

        color:
            #4b5563;

    }


    .selected-model-value {

        font-size:
            16px;

        font-weight:
            700;

        color:
            #1f2937;

        margin-bottom:
            10px;

    }


    /* =======================================================
       FOOD CLASS BOX
    ======================================================= */

    .class-box {

        background:
            #ffffff;

        border:
            1px solid #e5e7eb;

        border-radius:
            10px;

        padding:
            9px 10px;

        margin-bottom:
            8px;

        min-height:
            60px;

        display:
            flex;

        align-items:
            center;

        gap:
            8px;

        box-shadow:
            0px 2px 5px
            rgba(0, 0, 0, 0.03);

    }


    .class-number {

        min-width:
            28px;

        width:
            28px;

        height:
            28px;

        border-radius:
            50%;

        display:
            flex;

        align-items:
            center;

        justify-content:
            center;

        font-size:
            11px;

        font-weight:
            700;

        background:
            #eef2ff;

        color:
            #4f46e5;

    }


    .class-name {

        font-size:
            13px;

        font-weight:
            600;

        color:
            #374151;

        line-height:
            1.3;

        word-break:
            break-word;

    }


    /* =======================================================
       RESULT CARD
    ======================================================= */

    .result-card {

        background:
            linear-gradient(
                135deg,
                #ffffff,
                #f0fdf4
            );

        border:
            1px solid #bbf7d0;

        border-radius:
            16px;

        padding:
            20px;

        text-align:
            center;

        margin-bottom:
            15px;

    }


    .prediction-food {

        font-size:
            27px;

        font-weight:
            750;

        color:
            #15803d;

        margin-top:
            6px;

    }


    .confidence {

        font-size:
            21px;

        font-weight:
            700;

        color:
            #2563eb;

        margin-top:
            5px;

    }


    /* =======================================================
       TOP PREDICTION CARD
    ======================================================= */

    .top-prediction-card {

        background:
            #ffffff;

        border:
            1px solid #e5e7eb;

        border-radius:
            10px;

        padding:
            10px 12px;

        margin-bottom:
            8px;

    }


    /* =======================================================
       HISTORY CARD
    ======================================================= */

    .history-card {

        background:
            #ffffff;

        border-left:
            4px solid #6366f1;

        border-radius:
            8px;

        padding:
            10px 12px;

        margin-bottom:
            8px;

        box-shadow:
            0px 2px 5px
            rgba(0, 0, 0, 0.03);

    }


    </style>
    """
)


# ============================================================
# LOAD CLASSES
# ============================================================

@st.cache_data
def load_classes():

    if not os.path.exists(
        CLASSES_JSON
    ):

        raise FileNotFoundError(
            f"classes.json not found: {CLASSES_JSON}"
        )


    with open(
        CLASSES_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        class_data = json.load(
            file
        )


    if (
        isinstance(class_data, dict)
        and "classes" in class_data
    ):

        classes_data = class_data["classes"]

    else:

        classes_data = class_data


    normalized_classes = []


    # --------------------------------------------------------
    # LIST FORMAT
    # --------------------------------------------------------

    if isinstance(
        classes_data,
        list
    ):

        for item in classes_data:

            if isinstance(
                item,
                dict
            ):

                class_name = item.get(
                    "class_name"
                )

                if class_name is not None:

                    normalized_classes.append(
                        str(class_name)
                    )

            else:

                normalized_classes.append(
                    str(item)
                )


    # --------------------------------------------------------
    # DICTIONARY FORMAT
    # --------------------------------------------------------

    elif isinstance(
        classes_data,
        dict
    ):

        # Example:
        # {"apple_pie": 0}

        if all(
            isinstance(value, int)
            for value in classes_data.values()
        ):

            sorted_items = sorted(
                classes_data.items(),
                key=lambda item: item[1]
            )


            normalized_classes = [

                class_name

                for class_name, class_index
                in sorted_items

            ]


        # Example:
        # {"0": "apple_pie"}

        else:

            sorted_items = sorted(
                classes_data.items(),
                key=lambda item: int(
                    item[0]
                )
            )


            normalized_classes = [

                class_name

                for class_index, class_name
                in sorted_items

            ]


    else:

        raise ValueError(
            "Unsupported classes.json format"
        )


    return normalized_classes


# ============================================================
# FORMAT FOOD NAME
# ============================================================

def format_food_name(
    food_name
):

    return (

        str(food_name)
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()

    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(
    model_path
):

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )


    return tf.keras.models.load_model(
        model_path
    )


# ============================================================
# LOAD MODEL COMPARISON METRICS
# ============================================================

@st.cache_data
def load_model_comparison_metrics():

    if not os.path.exists(
        MODEL_COMPARISON_JSON
    ):

        raise FileNotFoundError(
            "Model comparison metrics file not found: "
            f"{MODEL_COMPARISON_JSON}"
        )


    with open(
        MODEL_COMPARISON_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        metrics_data = json.load(
            file
        )


    return metrics_data


# ============================================================
# GET MODEL METRICS
# ============================================================

def get_model_metrics(
    architecture
):

    metrics_data = (
        load_model_comparison_metrics()
    )


    architecture_mapping = {

        "CNN":
            "Custom CNN",

        "VGG16":
            "VGG16",

        "ResNet50":
            "ResNet50"

    }


    json_model_name = (
        architecture_mapping[
            architecture
        ]
    )


    model_data = (
        metrics_data["models"][
            json_model_name
        ]
    )


    return {

        "accuracy":
            f'{model_data["accuracy"]:.2f}%',

        "accuracy_value":
            float(
                model_data["accuracy"]
            ),

        "model_name":
            json_model_name

    }


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image,
    architecture
):

    image = image.convert(
        "RGB"
    )


    image = image.resize(
        (
            224,
            224
        )
    )


    image_array = np.array(
        image
    ).astype(
        np.float32
    )


    if architecture == "CNN":

        image_array = (
            image_array / 255.0
        )


    elif architecture == "VGG16":

        image_array = vgg16_preprocess(
            image_array
        )


    elif architecture == "ResNet50":

        image_array = resnet50_preprocess(
            image_array
        )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ============================================================
# GET MODEL PATH
# ============================================================

def get_model_path(
    architecture
):

    model_paths = {

        "CNN":
            CNN_MODEL_PATH,

        "VGG16":
            VGG16_MODEL_PATH,

        "ResNet50":
            RESNET50_MODEL_PATH

    }


    return model_paths[
        architecture
    ]


# ============================================================
# PREDICT FOOD
# ============================================================

def predict_food(
    image,
    architecture,
    classes
):

    model_path = get_model_path(
        architecture
    )


    model = load_model(
        model_path
    )


    processed_image = preprocess_image(
        image,
        architecture
    )


    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if len(
        predictions
    ) != len(
        classes
    ):

        raise ValueError(

            f"Model output classes ({len(predictions)}) "
            f"do not match classes.json ({len(classes)})."

        )


    predicted_index = int(
        np.argmax(
            predictions
        )
    )


    predicted_food = classes[
        predicted_index
    ]


    confidence = float(
        predictions[
            predicted_index
        ] * 100
    )


    top_3_indices = np.argsort(
        predictions
    )[-3:][::-1]


    top_predictions = []


    for index in top_3_indices:


        index = int(
            index
        )


        top_predictions.append(

            {

                "food":
                    classes[index],

                "confidence":
                    float(
                        predictions[index] * 100
                    )

            }

        )


    return (

        predicted_food,

        confidence,

        top_predictions

    )


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:

    st.session_state.prediction_result = None


if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ============================================================
# LOAD CLASSES
# ============================================================

classes = load_classes()

TOTAL_CLASSES = len(
    classes
)
CNN_ACCURACY = (
    get_model_metrics(
        "CNN"
    )["accuracy_value"]
)


VGG16_ACCURACY = (
    get_model_metrics(
        "VGG16"
    )["accuracy_value"]
)


RESNET50_ACCURACY = (
    get_model_metrics(
        "ResNet50"
    )["accuracy_value"]
)

model_comparison_metrics = (
    load_model_comparison_metrics()
)


BEST_MODEL = (
    model_comparison_metrics[
        "best_model"
    ]
)


BEST_ACCURACY = (
    float(
        model_comparison_metrics[
            "best_accuracy"
        ]
    )
)


# ============================================================
# HEADER
# ============================================================

render_html(
    """
    <div class="main-title">
        🍔 Food Selection Using Deep Learning
    </div>
    """
)


render_html(
    """
    <div class="subtitle">
        CNN vs VGG16 vs ResNet50 | AI Food Classification System
    </div>
    """
)


st.divider()


# ============================================================
# THREE COLUMN LAYOUT
# ============================================================

left_column, center_column, right_column = st.columns(

    [

        1.05,

        2.35,

        1.55

    ],

    gap="large"

)


# ============================================================
# LEFT PANEL
# ============================================================

with left_column:


    render_html(
        """
        <div class="section-title">
            👤 User Information
        </div>
        """
    )

    # --------------------------------------------------------
    # PROFILE CARD
    # --------------------------------------------------------

    profile_image = get_profile_image()

    if profile_image is not None:

        render_html(
            f"""
            <div class="profile-card">

                <div class="profile-avatar">

                    <img
                        src="data:image/jpeg;base64,{profile_image}"
                        alt="Syeda Nazneen"
                    >

                </div>

                <div class="profile-name">
                    Syeda Nazneen
                </div>

                <div class="profile-role">
                    Deep Learning Developer
                </div>

            </div>
            """
        )


    else:

        render_html(
            """
            <div class="profile-card">

                <div class="profile-avatar"
                     style="
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:40px;
                     ">

                    👩‍💻

                </div>

                <div class="profile-name">
                    Syeda Nazneen
                </div>

                <div class="profile-role">
                    Deep Learning Developer
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # SKILL
    # --------------------------------------------------------

    render_html(
        """
        <div class="info-card">

            <div class="info-label">
                🧠 Primary Skill
            </div>

            <div class="info-value">
                Deep Learning
            </div>

        </div>
        """
    )


    st.divider()


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    render_html(
        """
        <div class="section-title">
            🤖 Model Information
        </div>
        """
    )


    render_html(
        f"""
        <div class="info-card">

            <div class="info-label">
                Total Food Classes
            </div>

            <div class="info-value">
                {TOTAL_CLASSES} Classes
            </div>

        </div>
        """
    )

    render_html(
        f"""
        <div class="info-card">

            <div class="info-label">
                🏆 Best Model
            </div>

            <div class="info-value">
                {BEST_MODEL}
            </div>

        </div>
        """
    )

    render_html(
        f"""
        <div class="info-card">

            <div class="info-label">
                📈 Best Validation Accuracy
            </div>

            <div class="info-value">
                {BEST_ACCURACY:.2f}%
            </div>

        </div>
        """
    )


    st.divider()


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    render_html(
        """
        <div class="section-title">
            📊 Model Comparison
        </div>
        """
    )

    render_html(
        f"""
        <div class="comparison-label">
            CNN — {CNN_ACCURACY:.2f}%
        </div>
        """
    )

    st.progress(
        64
    )

    render_html(
        f"""
        <div class="comparison-label">
            VGG16 — {VGG16_ACCURACY:.2f}%
        </div>
        """
    )

    st.progress(
        75
    )

    render_html(
        f"""
        <div class="comparison-label">
            ResNet50 — {RESNET50_ACCURACY:.2f}%
        </div>
        """
    )

    st.progress(
        83
    )

    # ========================================================
    # ACCURACY COMPARISON CHART
    # ========================================================

    render_html(
        """
        <div class="section-title">
            📈 Accuracy Chart
        </div>
        """
    )

    accuracy_data = pd.DataFrame(
        {
            "Accuracy": [
                CNN_ACCURACY,
                VGG16_ACCURACY,
                RESNET50_ACCURACY
            ]
        },

        index=[
            "CNN",
            "VGG16",
            "ResNet50"
        ]
    )

    st.bar_chart(
        accuracy_data,
        use_container_width=True,
        height=280
    )

    st.caption(
        "Validation accuracy comparison between all Deep Learning models"
    )


# ============================================================
# CENTER PANEL
# ============================================================

with center_column:


    render_html(
        """
        <div class="section-title">
            🍽️ Food Prediction
        </div>
        """
    )


    uploaded_file = st.file_uploader(

        "Upload Food Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )


    selected_image = None


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    if uploaded_file is not None:


        selected_image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )


        st.image(

            selected_image,

            caption="Uploaded Food Image",

            use_container_width=True

        )


    # ========================================================
    # ARCHITECTURE SELECTION
    # ========================================================

    render_html(
        """
        <div class="architecture-title">
            Select Architecture
        </div>
        """
    )


    architecture = st.radio(

        "Select Model",

        [
            "CNN",
            "VGG16",
            "ResNet50"
        ],

        horizontal=True,

        label_visibility="collapsed"

    )


    selected_metrics = get_model_metrics(
        architecture
    )


    # ========================================================
    # SELECTED MODEL BOX
    # ========================================================

    render_html(
        f"""
        <div class="selected-model-box">

            <div class="selected-model-title">
                Selected Architecture
            </div>

            <div class="selected-model-value">
                {architecture}
            </div>

            <div class="selected-model-title">
                Validation Accuracy
            </div>

            <div class="selected-model-value">
                {selected_metrics["accuracy"]}
            </div>

        </div>
        """
    )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict_button = st.button(

        "🔮 Predict Food",

        use_container_width=True,

        type="primary"

    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:


        if uploaded_file is None:


            st.warning(
                "⚠️ Please upload a food image first."
            )


        else:


            try:


                with st.spinner(
                    f"🤖 Predicting using {architecture}..."
                ):


                    predicted_food, confidence, top_3 = (
                        predict_food(

                            selected_image,

                            architecture,

                            classes

                        )
                    )


                # STORE RESULT

                st.session_state.prediction_result = {

                    "food":
                        predicted_food,

                    "confidence":
                        confidence,

                    "architecture":
                        architecture,

                    "top_3":
                        top_3

                }


                # ADD HISTORY

                st.session_state.prediction_history.insert(

                    0,

                    {

                        "food":
                            predicted_food,

                        "confidence":
                            confidence,

                        "architecture":
                            architecture,

                        "time":
                            datetime.now().strftime(
                                "%H:%M:%S"
                            )

                    }

                )


                # KEEP LAST 10

                st.session_state.prediction_history = (

                    st.session_state.prediction_history[:10]

                )


                st.success(
                    "✅ Prediction Completed Successfully!"
                )


                st.rerun()


            except Exception as error:


                st.error(
                    f"Prediction Error: {error}"
                )


# ============================================================
# RIGHT PANEL
# ============================================================

with right_column:


    # ========================================================
    # BEFORE PREDICTION
    # SHOW ALL FOOD CLASSES
    # ========================================================

    if st.session_state.prediction_result is None:


        render_html(
            f"""
            <div class="section-title">
                🍔 Food Classes ({TOTAL_CLASSES})
            </div>
            """
        )


        st.caption(
            "Available food categories in the dataset"
        )


        # TWO COLUMN GRID

        for i in range(
            0,
            TOTAL_CLASSES,
            2
        ):


            class_col_1, class_col_2 = st.columns(
                2,
                gap="small"
            )


            # ------------------------------------------------
            # CLASS 1
            # ------------------------------------------------

            with class_col_1:


                food_name = format_food_name(
                    classes[i]
                )


                render_html(
                    f"""
                    <div class="class-box">

                        <div class="class-number">
                            {i + 1}
                        </div>

                        <div class="class-name">
                            {food_name}
                        </div>

                    </div>
                    """
                )


            # ------------------------------------------------
            # CLASS 2
            # ------------------------------------------------

            if i + 1 < TOTAL_CLASSES:


                with class_col_2:


                    food_name = format_food_name(
                        classes[i + 1]
                    )


                    render_html(
                        f"""
                        <div class="class-box">

                            <div class="class-number">
                                {i + 2}
                            </div>

                            <div class="class-name">
                                {food_name}
                            </div>

                        </div>
                        """
                    )


    # ========================================================
    # AFTER PREDICTION
    # SHOW RESULT / METRICS / HISTORY
    # ========================================================

    else:


        result = (
            st.session_state.prediction_result
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        render_html(
            """
            <div class="section-title">
                🎯 Prediction Result
            </div>
            """
        )


        render_html(
            f"""
            <div class="result-card">

                <div class="info-label">
                    Predicted Food
                </div>

                <div class="prediction-food">
                    🍽️ {format_food_name(result["food"])}
                </div>

                <br>

                <div class="info-label">
                    Prediction Confidence
                </div>

                <div class="confidence">
                    {result["confidence"]:.2f}%
                </div>

            </div>
            """
        )


        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        render_html(
            """
            <div class="section-title">
                🏆 Top 3 Predictions
            </div>
            """
        )


        for rank, prediction in enumerate(

            result["top_3"],

            start=1

        ):


            food_name = format_food_name(
                prediction["food"]
            )


            render_html(
                f"""
                <div class="top-prediction-card">

                    <b>
                        #{rank} {food_name}
                    </b>

                    <br>

                    <small>
                        Confidence:
                        {prediction["confidence"]:.2f}%
                    </small>

                </div>
                """
            )


            st.progress(

                min(

                    int(
                        prediction["confidence"]
                    ),

                    100

                )

            )


        st.divider()


        # ====================================================
        # MODEL METRICS
        # ====================================================

        render_html(
            """
            <div class="section-title">
                📊 Model Metrics
            </div>
            """
        )


        current_metrics = get_model_metrics(

            result["architecture"]

        )


        metric_col_1, metric_col_2 = st.columns(
            2
        )


        with metric_col_1:


            st.metric(

                "Model",

                result["architecture"]

            )


        with metric_col_2:


            st.metric(

                "Accuracy",

                current_metrics["accuracy"]

            )


        st.metric(

            "Total Classes",

            TOTAL_CLASSES

        )


        st.divider()


        # ====================================================
        # PREDICTION HISTORY
        # ====================================================

        render_html(
            """
            <div class="section-title">
                🕒 Prediction History
            </div>
            """
        )


        if len(
            st.session_state.prediction_history
        ) == 0:


            st.info(
                "No prediction history available."
            )


        else:


            for item in st.session_state.prediction_history:


                food_name = format_food_name(
                    item["food"]
                )


                render_html(
                    f"""
                    <div class="history-card">

                        <b>
                            🍽️ {food_name}
                        </b>

                        <br>

                        <small>
                            Model:
                            {item["architecture"]}
                        </small>

                        <br>

                        <small>
                            Confidence:
                            {item["confidence"]:.2f}%
                        </small>

                        <br>

                        <small>
                            Time:
                            {item["time"]}
                        </small>

                    </div>
                    """
                )


        # ====================================================
        # NEW PREDICTION
        # ====================================================

        if st.button(

            "🔄 New Prediction",

            use_container_width=True

        ):


            st.session_state.prediction_result = None


            st.rerun()