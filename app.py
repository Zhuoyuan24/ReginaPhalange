# ============================================================
# ISOM5240 Individual Assignment
# Storytelling Application using Hugging Face Models
#
# Professor's required structure:
# 1. Import part
# 2. Functions part
# 3. Main part
# ============================================================


# ----------------------------
# 1. IMPORT PART
# ----------------------------
import re
from pathlib import Path
from typing import Dict, Tuple
from collections import Counter

import streamlit as st
from PIL import Image
import torch

from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
    ViltProcessor,
    ViltForQuestionAnswering,
)


# ----------------------------
# 2. MODEL SETTINGS
# ----------------------------

# Professor-provided image captioning model
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

# Extra image-understanding models
# These help the scenario contain more details from the picture.
VQA_MODEL = "dandelin/vilt-b32-finetuned-vqa"
OBJECT_DETECTION_MODEL = "facebook/detr-resnet-50"

# Professor-provided story generation model
STORY_MODEL = "pranavpsv/genre-story-generator-v2"

# Professor-provided text-to-audio model
TTS_MODEL = "Matthijs/mms-tts-eng"


# ----------------------------
# 3. FUNCTIONS PART
# ----------------------------

def get_torch_device() -> torch.device:
    """
    Choose GPU if available, otherwise use CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_pipeline_device() -> int:
    """
    Hugging Face pipeline uses:
    device = 0  for GPU
    device = -1 for CPU
    """
    if torch.cuda.is_available():
        return 0
    return -1


@st.cache_resource(show_spinner="Loading BLIP image captioning model...")
def load_image_captioning_model():
    """
    Load the professor-provided BLIP image captioning model.

    Important:
    We do NOT use pipeline("image-to-text") here because the current
    Transformers environment may not support that task name.
    """
    device = get_torch_device()

    processor = BlipProcessor.from_pretrained(IMAGE_CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(IMAGE_CAPTION_MODEL)

    model.to(device)
    model.eval()

    return processor, model, device


@st.cache_resource(show_spinner="Loading visual question answering model...")
def load_vqa_model():
    """
    Load the visual question answering model.

    This model helps answer questions about the image, such as:
    - who is in the picture
    - where the scene is
    - what is happening
    """
    device = get_torch_device()

    processor = ViltProcessor.from_pretrained(VQA_MODEL)
    model = ViltForQuestionAnswering.from_pretrained(VQA_MODEL)

    model.to(device)
    model.eval()

    return processor, model, device


@st.cache_resource(show_spinner="Loading object detection model...")
def load_object_detection_pipeline():
    """
    Load the object detection pipeline.

    Note:
    facebook/detr-resnet-50 requires the timm package.
    Make sure timm is included in requirements.txt.
    """
    object_detection_model = pipeline(
        task="object-detection",
        model=OBJECT_DETECTION_MODEL,
        device=get_pipeline_device(),
    )

    return object_detection_model


@st.cache_resource(show_spinner="Loading story generation model...")
def load_story_pipeline():
    """
    Load the text-generation pipeline.

    This model expands the image scenario into a short story.
    """
    story_model = pipeline(
        task="text-generation",
        model=STORY_MODEL,
        device=get_pipeline_device(),
    )

    return story_model


@st.cache_resource(show_spinner="Loading text-to-audio model...")
def load_tts_pipeline():
    """
    Load the text-to-audio pipeline.

    The professor's template uses:
    pipeline("text-to-audio", model="Matthijs/mms-tts-eng")

    Some Transformers versions use "text-to-speech", so this function
    tries text-to-audio first and then uses text-to-speech as fallback.
    """
    try:
        audio_model = pipeline(
            task="text-to-audio",
            model=TTS_MODEL,
            device=get_pipeline_device(),
        )
    except Exception:
        audio_model = pipeline(
            task="text-to-speech",
            model=TTS_MODEL,
            device=get_pipeline_device(),
        )

    return audio_model


def generate_basic_caption(image: Image.Image) -> str:
    """
    Generate a basic image caption using the professor-provided BLIP model.

    Returns:
        A short caption, such as:
        "a child playing with a dog in the park"
    """
    processor, model, device = load_image_captioning_model()

    inputs = processor(image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
        )

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def ask_vqa_question(image: Image.Image, question: str) -> str:
    """
    Ask a visual question about the uploaded image.

    Parameters:
        image: PIL image
        question: question about the image

    Returns:
        A short answer from the VQA model.
    """
    try:
        processor, model, device = load_vqa_model()

        inputs = processor(
            image,
            question,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_id = outputs.logits.argmax(-1).item()
        answer = model.config.id2label[predicted_id]

        return str(answer).strip()

    except Exception:
        # The app should not stop if VQA fails.
        return "unknown"


def format_object_count(label: str, count: int) -> str:
    """
    Format detected object counts in simple English.

    Example:
        person, 2 -> 2 people
        dog, 1 -> 1 dog
    """
    label = label.lower().strip()

    if count == 1:
        return f"1 {label}"

    if label == "person":
        return f"{count} people"

    if label.endswith("s"):
        return f"{count} {label}es"

    if label.endswith("y"):
        return f"{count} {label[:-1]}ies"

    return f"{count} {label}s"


def detect_main_objects(image: Image.Image, score_threshold: float = 0.80) -> str:
    """
    Detect visible objects in the uploaded image.

    Parameters:
        image: PIL image
        score_threshold: minimum confidence score for keeping detected objects

    Returns:
        A short text summary of the main objects.
    """
    try:
        object_detector = load_object_detection_pipeline()
        detections = object_detector(image)

        object_labels = []

        for item in detections:
            label = item.get("label", "").lower()
            score = item.get("score", 0)

            if label and score >= score_threshold:
                object_labels.append(label)

        if not object_labels:
            return "no clear objects detected"

        object_counts = Counter(object_labels)

        object_summary = []
        for label, count in object_counts.most_common(6):
            object_summary.append(format_object_count(label, count))

        return ", ".join(object_summary)

    except Exception:
        # The app should still work even if object detection fails.
        return "object detection unavailable"


def img2text(url: str) -> str:
    """
    Convert uploaded image into a richer scenario.

    This follows the professor's required function style:

        def img2text(url):
            ...
            return text

    The function extracts:
    1. Basic caption
    2. Main subject
    3. People or characters
    4. Location or setting
    5. Main activity
    6. Visible objects
    7. Mood or theme

    This gives the story model more picture-specific information.
    """
    image = Image.open(url).convert("RGB")

    # 1. Basic image caption using BLIP
    caption = generate_basic_caption(image)

    # 2. Ask image-specific questions using VQA
    main_subject = ask_vqa_question(image, "What is the main subject of the image?")
    people = ask_vqa_question(image, "Who is in the picture?")
    place = ask_vqa_question(image, "Where is the scene?")
    activity = ask_vqa_question(image, "What is happening in the image?")
    mood = ask_vqa_question(image, "What is the mood of the image?")

    # 3. Detect important visible objects
    objects = detect_main_objects(image)

    # 4. Combine all image information into one scenario
    text = (
        f"Image caption: {caption}. "
        f"Main subject: {main_subject}. "
        f"People or characters: {people}. "
        f"Location or setting: {place}. "
        f"Main activity: {activity}. "
        f"Visible objects: {objects}. "
        f"Mood or theme: {mood}."
    )

    return text


def build_story_prompt(text: str) -> str:
    """
    Build a clear prompt for the story-generation model.

    This version removes user-selected genre, lesson, and age.
    It focuses only on the uploaded picture.
    """
    prompt = (
        "Write a short story for young children based only on the image details below. "
        "The story must clearly use the main subject, people, place, activity, objects, "
        "and mood from the image. "
        "Do not invent unrelated characters, places, or events. "
        "Use simple, warm, child-friendly English. "
        "Keep the story between 50 and 100 words. "
        "Do not include scary, violent, or adult content. "
        f"Image details: {text} "
        "Story:"
    )

    return prompt


def extract_story_only(raw_story: str, prompt: str) -> str:
    """
    Remove prompt text and extra formatting from the generated story.
    """
    story = raw_story.replace(prompt, "").strip()
    story = story.replace("Story:", "").strip()
    story = re.sub(r"\s+", " ", story)
    return story


def limit_story_length(story: str, max_words: int = 100) -> str:
    """
    Limit the story length to match the assignment requirement.
    """
    words = story.split()

    if len(words) <= max_words:
        return story

    shortened_story = " ".join(words[:max_words])

    if not shortened_story.endswith((".", "!", "?")):
        shortened_story += "."

    return shortened_story


def make_simple_backup_story(text: str) -> str:
    """
    Create a simple backup story if the text-generation model output is too short.

    The backup story still uses the image scenario, so it remains related
    to the uploaded image.
    """
    backup_story = (
        f"In the picture, we can see this scene: {text} "
        "A little explorer looks carefully at the people, place, and objects around them. "
        "The moment feels simple, warm, and special. Everyone enjoys what they are doing, "
        "and the small scene becomes a happy story about noticing the world with kindness."
    )

    return limit_story_length(backup_story, max_words=100)


def clean_story_text(raw_story: str, prompt: str, text: str) -> str:
    """
    Clean and check the generated story.

    Steps:
    1. Remove prompt text.
    2. Remove extra spaces.
    3. Cut story to 100 words if needed.
    4. Use a simple image-based backup story if output is too short.
    """
    story = extract_story_only(raw_story, prompt)
    story = limit_story_length(story, max_words=100)

    if len(story.split()) < 40:
        story = make_simple_backup_story(text)

    return story


def text2story(text: str) -> str:
    """
    Convert image scenario text into a short story.

    This follows the professor's required function style:

        def text2story(text):
            story_text = ""
            return story_text

    Parameters:
        text: rich image scenario from img2text()

    Returns:
        story_text: generated story
    """
    story_model = load_story_pipeline()
    prompt = build_story_prompt(text)

    try:
        story_results = story_model(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.50,
            top_p=0.85,
            repetition_penalty=1.20,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            return_full_text=False,
        )
    except TypeError:
        story_results = story_model(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.50,
            top_p=0.85,
            repetition_penalty=1.20,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
        )

    raw_story = story_results[0]["generated_text"]

    story_text = clean_story_text(
        raw_story=raw_story,
        prompt=prompt,
        text=text,
    )

    return story_text


def text2audio(story_text: str) -> Tuple[object, int]:
    """
    Convert generated story text into audio.

    This follows the professor's required function style:

        def text2audio(story_text):
            audio_data = ""
            return audio_data

    Returns:
        audio_array: audio waveform
        sample_rate: audio sample rate for st.audio()
    """
    audio_model = load_tts_pipeline()
    audio_data: Dict = audio_model(story_text)

    audio_array = audio_data["audio"]
    sample_rate = audio_data["sampling_rate"]

    return audio_array, sample_rate


def save_uploaded_image(uploaded_file) -> str:
    """
    Save uploaded image locally, following the professor's example.

    Returns:
        image_path: local file path for the uploaded image
    """
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in [".jpg", ".jpeg", ".png"]:
        suffix = ".png"

    image_path = f"uploaded_image{suffix}"

    bytes_data = uploaded_file.getvalue()

    with open(image_path, "wb") as file:
        file.write(bytes_data)

    return image_path


def show_sidebar():
    """
    Show app explanation in the sidebar.
    """
    st.sidebar.title("How this app works")
    st.sidebar.write(
        "1. Upload an image.\n"
        "2. The app extracts image details.\n"
        "3. The app creates a short story.\n"
        "4. The app converts the story into audio."
    )

    st.sidebar.divider()

    st.sidebar.write("Models used:")
    st.sidebar.caption("Image caption: Salesforce/blip-image-captioning-base")
    st.sidebar.caption("Visual Q&A: dandelin/vilt-b32-finetuned-vqa")
    st.sidebar.caption("Object detection: facebook/detr-resnet-50")
    st.sidebar.caption("Story: pranavpsv/genre-story-generator-v2")
    st.sidebar.caption("Audio: Matthijs/mms-tts-eng")


# ----------------------------
# 4. MAIN PART
# ----------------------------

def main():
    """
    Main Streamlit application.

    Execution flow:
    upload image -> show image -> img2text -> text2story -> text2audio
    """
    st.set_page_config(
        page_title="Your Image to Audio Story",
        page_icon="🦜",
        layout="centered",
    )

    show_sidebar()

    st.title("🦜 Your Image to Audio Story")
    st.write("Upload an image and turn it into a short audio story for kids.")

    st.divider()

    uploaded_file = st.file_uploader(
        "Select an Image...",
        type=["jpg", "jpeg", "png"],
        key="story_image_uploader",
    )

    if uploaded_file is not None:
        image_path = save_uploaded_image(uploaded_file)

        st.subheader("Uploaded Image")
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Story and Audio", type="primary"):
            try:
                # Stage 1: Image to text
                st.text("Processing img2text...")
                with st.spinner("Understanding the image..."):
                    scenario = img2text(image_path)

                st.subheader("1. Generated Scenario")
                st.write(scenario)

                # Stage 2: Text to story
                st.text("Generating a story...")
                with st.spinner("Writing a short child-friendly story..."):
                    story = text2story(scenario)

                st.subheader("2. Generated Story")
                st.write(story)
                st.caption(f"Word count: {len(story.split())} words")

                # Stage 3: Story to audio
                st.text("Generating audio data...")
                with st.spinner("Creating audio..."):
                    audio_array, sample_rate = text2audio(story)

                st.subheader("3. Story Audio")
                st.audio(audio_array, sample_rate=sample_rate)

                st.success("Done! The image has been converted into a story and audio.")

            except Exception as error:
                st.error("Something went wrong while generating the story or audio.")
                st.exception(error)

    else:
        st.info("Please upload an image to start.")


if __name__ == "__main__":
    main()
