# ============================================================
# ISOM5240 Individual Assignment
# Storytelling Application using Hugging Face Pipelines
# Structure follows: Import part -> Functions part -> Main part
# ============================================================

# ----------------------------
# 1. IMPORT PART
# ----------------------------
import re
from typing import Dict, Tuple

import streamlit as st
from PIL import Image
from transformers import pipeline
import torch


# ----------------------------
# 2. CONSTANTS / MODEL SETTINGS
# ----------------------------
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "pranavpsv/genre-story-generator-v2"
TTS_MODEL = "facebook/mms-tts-eng"   # Alternative from professor notes: "Matthijs/mms-tts-eng"


# ----------------------------
# 3. FUNCTIONS PART
# ----------------------------
def get_device() -> int:
    """
    Return GPU device number if CUDA is available; otherwise use CPU.
    Hugging Face pipeline uses device=-1 for CPU and device=0 for first GPU.
    """
    return 0 if torch.cuda.is_available() else -1


@st.cache_resource(show_spinner="Loading image captioning model...")
def load_image_captioning_pipeline():
    """Load and cache the image-to-text pipeline."""
    return pipeline(
        task="image-to-text",
        model=IMAGE_CAPTION_MODEL,
        device=get_device(),
    )


@st.cache_resource(show_spinner="Loading story generation model...")
def load_story_pipeline():
    """Load and cache the text-generation pipeline."""
    return pipeline(
        task="text-generation",
        model=STORY_MODEL,
        device=get_device(),
    )


@st.cache_resource(show_spinner="Loading text-to-speech model...")
def load_tts_pipeline():
    """Load and cache the text-to-speech pipeline."""
    return pipeline(
        task="text-to-speech",
        model=TTS_MODEL,
        device=get_device(),
    )


def img2text(image: Image.Image) -> str:
    """
    Convert an uploaded image into a short scenario/caption.

    Parameters:
        image: PIL image uploaded by user.

    Returns:
        A short text caption describing the image.
    """
    image_to_text_model = load_image_captioning_pipeline()
    result = image_to_text_model(image)

    # The output is normally a list of dictionaries:
    # [{'generated_text': 'a child playing in a park'}]
    scenario = result[0].get("generated_text", "")
    return scenario.strip()


def build_story_prompt(
    scenario: str,
    genre: str,
    lesson: str,
    audience_age: int,
) -> str:
    """
    Create a clear prompt for the story-generation model.
    The prompt is designed to produce a safe 50-100 word story for kids.
    """
    prompt = (
        f"Write a short {genre.lower()} story for a {audience_age}-year-old child. "
        f"The story should be based on this scene: {scenario}. "
        f"The story should teach this lesson: {lesson}. "
        "Use simple, warm, child-friendly English. "
        "Keep the story between 50 and 100 words. "
        "Do not include scary, violent, or adult content. "
        "Story:"
    )
    return prompt


def clean_story_text(raw_story: str, prompt: str) -> str:
    """
    Clean the generated story:
    - Remove the prompt if it appears in the model output.
    - Remove extra spaces.
    - Cut the story to about 100 words to match assignment requirement.
    """
    story = raw_story.replace(prompt, "").strip()
    story = story.replace("Story:", "").strip()
    story = re.sub(r"\s+", " ", story)

    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100])
        # End more naturally if possible.
        if not story.endswith((".", "!", "?")):
            story += "."

    # Very short fallback in case the model returns empty or low-quality text.
    if len(story.split()) < 30:
        story = (
            "Once upon a time, a little explorer saw something wonderful in the picture. "
            "With a brave heart and a kind smile, the explorer learned to help others, "
            "share joy, and keep trying. By the end of the day, everyone felt proud, "
            "because even small acts of kindness can make a big difference."
        )

    return story


def text2story(
    scenario: str,
    genre: str = "Adventure",
    lesson: str = "Be kind and curious",
    audience_age: int = 6,
) -> str:
    """
    Generate a short story from the image scenario.

    Parameters:
        scenario: caption generated from image.
        genre: selected story style.
        lesson: moral/lesson selected by user.
        audience_age: intended child age.

    Returns:
        A 50-100 word child-friendly story.
    """
    story_model = load_story_pipeline()
    prompt = build_story_prompt(scenario, genre, lesson, audience_age)

    try:
        result = story_model(
            prompt,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            return_full_text=False,
        )
        raw_story = result[0]["generated_text"]
    except TypeError:
        # Some older pipelines may not support return_full_text.
        result = story_model(
            prompt,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
        )
        raw_story = result[0]["generated_text"]

    return clean_story_text(raw_story, prompt)


def text2audio(story_text: str) -> Tuple[object, int]:
    """
    Convert story text into audio.

    Returns:
        audio_array: waveform/audio data
        sample_rate: sampling rate needed by st.audio
    """
    audio_model = load_tts_pipeline()
    audio_data: Dict = audio_model(story_text)
    audio_array = audio_data["audio"]
    sample_rate = audio_data["sampling_rate"]
    return audio_array, sample_rate


def show_assignment_sidebar():
    """Show a short app explanation in the sidebar."""
    st.sidebar.title("How this app works")
    st.sidebar.write(
        "1. Upload an image.\n"
        "2. The app describes the image as a scenario.\n"
        "3. The scenario becomes a short kid-friendly story.\n"
        "4. The story is converted into audio."
    )
    st.sidebar.divider()
    st.sidebar.caption("Built for ISOM5240 Individual Assignment")


# ----------------------------
# 4. MAIN PART
# ----------------------------
def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Image to Audio Story",
        page_icon="📚",
        layout="centered",
    )

    show_assignment_sidebar()

    st.title("📚 Turn Your Image into an Audio Story")
    st.write(
        "Upload an image, choose a few story settings, and the app will create "
        "a short story with audio for young children."
    )

    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        key="story_image_uploader",
    )

    genre = st.selectbox(
        "Choose a story style",
        options=["Adventure", "Fantasy", "Friendship", "Animal Story", "Bedtime"],
    )

    lesson = st.selectbox(
        "Choose a simple lesson",
        options=[
            "Be kind and curious",
            "Share with friends",
            "Keep trying",
            "Help others",
            "Believe in yourself",
        ],
    )

    audience_age = st.slider(
        "Target child age",
        min_value=3,
        max_value=10,
        value=6,
    )

    if uploaded_file is None:
        st.info("Please upload an image to start the story generation process.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story and Audio", type="primary"):
        try:
            with st.spinner("Step 1/3: Understanding the image..."):
                scenario = img2text(image)

            st.subheader("1. Generated Scenario")
            st.write(scenario)

            with st.spinner("Step 2/3: Writing a short story..."):
                story = text2story(
                    scenario=scenario,
                    genre=genre,
                    lesson=lesson,
                    audience_age=audience_age,
                )

            st.subheader("2. Generated Story")
            st.write(story)
            st.caption(f"Word count: {len(story.split())} words")

            with st.spinner("Step 3/3: Creating audio..."):
                audio_array, sample_rate = text2audio(story)

            st.subheader("3. Story Audio")
            st.audio(audio_array, sample_rate=sample_rate)
            st.success("Done! The image has been converted into a story and audio.")

        except Exception as error:
            st.error("Something went wrong while generating the story or audio.")
            st.exception(error)


if __name__ == "__main__":
    main()
