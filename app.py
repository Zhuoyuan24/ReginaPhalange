# ============================================================
# ISOM5240 Individual Assignment
# Storytelling Application using Hugging Face Models + gTTS
#
# Professor's required structure:
# 1. Import part
# 2. Functions part
# 3. Main part
# ============================================================


# ----------------------------
# 1. IMPORT PART
# ----------------------------
import io
import re
from pathlib import Path
from typing import Dict

import streamlit as st
from PIL import Image
import torch
from gtts import gTTS

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    ViltProcessor,
    ViltForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ----------------------------
# 2. MODEL SETTINGS
# ----------------------------

# Image captioning model suggested by professor
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

# Extra image understanding model
# Used only to extract character, setting, and activity
VQA_MODEL = "dandelin/vilt-b32-finetuned-vqa"

# Better story generation model
# FLAN-T5 follows instructions better than the earlier story generator model
STORY_MODEL = "google/flan-t5-base"


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


@st.cache_resource(show_spinner="Loading image captioning model...")
def load_image_captioning_model():
    """
    Load BLIP image captioning model.

    This replaces pipeline("image-to-text") because some Streamlit Cloud
    environments no longer recognize the image-to-text pipeline task name.
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
    Load visual question answering model.

    This model helps extract simple image details:
    - character
    - setting
    - activity
    """
    device = get_torch_device()

    processor = ViltProcessor.from_pretrained(VQA_MODEL)
    model = ViltForQuestionAnswering.from_pretrained(VQA_MODEL)

    model.to(device)
    model.eval()

    return processor, model, device


@st.cache_resource(show_spinner="Loading story generation model...")
def load_story_model():
    """
    Load FLAN-T5 model for story generation.
    """
    device = get_torch_device()

    tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(STORY_MODEL)

    model.to(device)
    model.eval()

    return tokenizer, model, device


def generate_basic_caption(image: Image.Image) -> str:
    """
    Generate the original image caption using BLIP.
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


def clean_short_answer(answer: str, fallback: str = "unknown") -> str:
    """
    Clean short VQA model answers.
    """
    if answer is None:
        return fallback

    answer = str(answer).strip().lower()

    if answer in ["", "none", "unknown", "nothing", "n/a"]:
        return fallback

    return answer


def ask_vqa_question(image: Image.Image, question: str) -> str:
    """
    Ask a visual question about the uploaded image.
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

        return clean_short_answer(answer)

    except Exception:
        return "unknown"


def choose_character(caption: str, character_answer: str) -> str:
    """
    Choose a natural character phrase from the caption and VQA answer.
    """
    combined = f"{caption} {character_answer}".lower()

    if "children" in combined or "kids" in combined:
        return "children"

    if "woman" in combined or "girl" in combined:
        return "a woman"

    if "man" in combined or "boy" in combined:
        return "a man"

    if "child" in combined:
        return "a child"

    if "people" in combined or "group" in combined:
        return "a group of people"

    if "person" in combined:
        return "a person"

    return "someone"


def choose_setting(caption: str, setting_answer: str) -> str:
    """
    Convert setting information into natural English.
    """
    combined = f"{caption} {setting_answer}".lower()

    if "gym" in combined:
        return "in a gym"

    if "park" in combined:
        return "in a park"

    if "snow" in combined or "mountain" in combined:
        return "on a snowy mountain"

    if "ocean" in combined or "sea" in combined:
        return "by the ocean"

    if "lake" in combined:
        return "by a lake"

    if "beach" in combined:
        return "at a beach"

    if "street" in combined or "road" in combined:
        return "on a street"

    if "room" in combined or "inside" in combined or "indoor" in combined:
        return "inside a room"

    if "outside" in combined or "outdoor" in combined:
        return "outside"

    if setting_answer not in ["unknown", "none", ""]:
        if setting_answer.startswith(("in ", "on ", "at ", "by ")):
            return setting_answer
        return f"in {setting_answer}"

    return "in a nice place"


def choose_activity(caption: str, activity_answer: str) -> str:
    """
    Convert activity information into natural English.
    """
    combined = f"{caption} {activity_answer}".lower()

    if "basketball" in combined:
        return "playing basketball"

    if "snowboard" in combined:
        return "snowboarding"

    if "ski" in combined:
        return "skiing"

    if "sitting on a rock" in combined and ("ocean" in combined or "sea" in combined):
        return "sitting on a rock and looking at the ocean"

    if "sitting" in combined:
        return "sitting quietly"

    if "walking" in combined or "walk" in combined:
        return "walking"

    if "running" in combined or "run" in combined:
        return "running"

    if "playing" in combined or "play" in combined:
        return "playing"

    if "laying in the snow" in combined or "lying in the snow" in combined:
        return "resting in the snow"

    if activity_answer not in ["unknown", "none", ""]:
        if activity_answer.endswith("ing"):
            return activity_answer
        return f"enjoying {activity_answer}"

    return "enjoying the moment"


def img2text(url: str) -> str:
    """
    Convert uploaded image into a concise scenario.

    This follows the professor's required function style:

        def img2text(url):
            text = ...
            return text

    This function extracts only the useful story information:
    - Character
    - Setting
    - Activity
    - Original image caption
    """
    image = Image.open(url).convert("RGB")

    # Basic image caption
    caption = generate_basic_caption(image)

    # Ask a few visual questions
    character_answer = ask_vqa_question(image, "Who is the main character in the image?")
    setting_answer = ask_vqa_question(image, "Where is the scene?")
    activity_answer = ask_vqa_question(image, "What is the main activity?")

    # Normalize answers into natural phrases
    character = choose_character(caption, character_answer)
    setting = choose_setting(caption, setting_answer)
    activity = choose_activity(caption, activity_answer)

    text = (
        f"Character: {character}. "
        f"Setting: {setting}. "
        f"Activity: {activity}. "
        f"Original image caption: {caption}."
    )

    return text


def parse_scenario(text: str) -> Dict[str, str]:
    """
    Parse scenario text into fields for the story prompt.
    """
    fields = {
        "character": "someone",
        "setting": "in a nice place",
        "activity": "enjoying the moment",
        "caption": "a simple scene",
    }

    patterns = {
        "character": r"Character:\s*(.*?)(?:\. Setting:|$)",
        "setting": r"Setting:\s*(.*?)(?:\. Activity:|$)",
        "activity": r"Activity:\s*(.*?)(?:\. Original image caption:|$)",
        "caption": r"Original image caption:\s*(.*?)(?:\.|$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value:
                fields[key] = value

    return fields


def build_story_prompt(text: str) -> str:
    """
    Build a clear image-grounded story prompt.
    """
    details = parse_scenario(text)

    prompt = f"""
Write one complete short story for children aged 3 to 10.

Use these picture details naturally:
Character: {details["character"]}
Setting: {details["setting"]}
Activity: {details["activity"]}
Original caption: {details["caption"]}

Story requirements:
- 50 to 100 words
- clear beginning, middle, and happy ending
- simple child-friendly English
- directly related to the picture details
- do not list the details
- do not mention the words "image", "caption", or "picture"
- no scary, violent, romantic, or adult content

Write only the story:
"""
    return prompt.strip()


def generate_story_from_prompt(prompt: str) -> str:
    """
    Generate a story using FLAN-T5.
    """
    tokenizer, model, device = load_story_model()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=140,
            num_beams=4,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    story = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return story.strip()


def remove_sensitive_sentences(story: str) -> str:
    """
    Remove sentences containing content unsuitable for children aged 3-10.
    """
    sensitive_words = [
        "wife",
        "husband",
        "fiancé",
        "fiance",
        "romantic",
        "blood",
        "kill",
        "dead",
        "death",
        "gun",
        "war",
        "adult",
        "drunk",
        "wine",
    ]

    sentences = re.findall(r"[^.!?]+[.!?]", story)

    if not sentences:
        sentences = [story]

    safe_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        has_sensitive_word = False

        for word in sensitive_words:
            if word in sentence_lower:
                has_sensitive_word = True
                break

        if not has_sensitive_word:
            safe_sentences.append(sentence.strip())

    cleaned_story = " ".join(safe_sentences)
    cleaned_story = re.sub(r"\s+", " ", cleaned_story).strip()

    return cleaned_story


def keep_complete_sentences(story: str) -> str:
    """
    Keep complete sentences only.
    """
    sentences = re.findall(r"[^.!?]+[.!?]", story)

    if not sentences:
        return story.strip()

    return " ".join(sentence.strip() for sentence in sentences)


def limit_story_length(story: str, max_words: int = 100) -> str:
    """
    Limit story to the assignment word range upper limit.
    """
    words = story.split()

    if len(words) <= max_words:
        return story

    shortened_story = " ".join(words[:max_words])
    shortened_story = keep_complete_sentences(shortened_story)

    if len(shortened_story.split()) < 40:
        shortened_story = " ".join(words[:max_words])
        if not shortened_story.endswith((".", "!", "?")):
            shortened_story += "."

    return shortened_story


def clean_story_text(story: str) -> str:
    """
    Clean the generated story.
    """
    story = story.replace("Story:", "").strip()
    story = story.replace("Write only the story:", "").strip()
    story = re.sub(r"\s+", " ", story).strip()

    story = remove_sensitive_sentences(story)
    story = keep_complete_sentences(story)
    story = limit_story_length(story, max_words=100)

    return story


def text2story(text: str) -> str:
    """
    Convert scenario text into a complete short story.

    This follows the professor's required function style:

        def text2story(text):
            story_text = ...
            return story_text
    """
    prompt = build_story_prompt(text)

    raw_story = generate_story_from_prompt(prompt)
    story_text = clean_story_text(raw_story)

    # If the first story is too short, ask the model one more time with a stronger prompt.
    if len(story_text.split()) < 50:
        retry_prompt = (
            prompt
            + "\n\nThe previous answer was too short. "
            + "Now write a complete 70 to 90 word story with a beginning, middle, and happy ending."
        )

        raw_story = generate_story_from_prompt(retry_prompt)
        story_text = clean_story_text(raw_story)

    return story_text


def prepare_text_for_audio(story_text: str) -> str:
    """
    Prepare text for smoother text-to-speech.
    """
    story_text = re.sub(r"\s+", " ", story_text).strip()

    # Extra spaces after punctuation help the audio pause more naturally.
    story_text = story_text.replace(". ", ".  ")
    story_text = story_text.replace("! ", "!  ")
    story_text = story_text.replace("? ", "?  ")

    return story_text


def text2audio(story_text: str) -> io.BytesIO:
    """
    Convert story text into MP3 audio using gTTS.

    This follows the professor's required function style:

        def text2audio(story_text):
            audio_data = ...
            return audio_data
    """
    audio_text = prepare_text_for_audio(story_text)

    audio_data = io.BytesIO()

    tts = gTTS(
        text=audio_text,
        lang="en",
        slow=False,
    )

    tts.write_to_fp(audio_data)
    audio_data.seek(0)

    return audio_data


def save_uploaded_image(uploaded_file) -> str:
    """
    Save uploaded image locally, following the professor's template.
    """
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in [".jpg", ".jpeg", ".png"]:
        suffix = ".png"

    image_path = f"uploaded_image{suffix}"

    bytes_data = uploaded_file.getvalue()

    with open(image_path, "wb") as file:
        file.write(bytes_data)

    return image_path


# ----------------------------
# 4. MAIN PART
# ----------------------------

def main():
    """
    Main Streamlit application.

    Flow:
    upload image -> show image -> img2text -> text2story -> text2audio
    """
    st.set_page_config(
        page_title="Your Image to Audio Story",
        page_icon="🦜",
        layout="centered",
    )

    st.header("Turn Your Image to Audio Story")
    st.write("Upload an image and the app will generate a short child-friendly story with audio.")

    uploaded_file = st.file_uploader(
        "Select an Image...",
        type=["jpg", "jpeg", "png"],
        key="story_image_uploader",
    )

    if uploaded_file is not None:
        image_path = save_uploaded_image(uploaded_file)

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True,
        )

        if st.button("Generate Story and Audio"):
            try:
                # Stage 1: Image to text
                st.text("Processing img2text...")
                scenario = img2text(image_path)
                st.write(f"**Scenario:** {scenario}")

                # Stage 2: Text to story
                st.text("Generating a story...")
                story = text2story(scenario)
                st.write(f"**Story:** {story}")
                st.caption(f"Word count: {len(story.split())} words")

                # Stage 3: Text to audio
                st.text("Generating audio data...")
                audio_data = text2audio(story)

                st.audio(audio_data, format="audio/mp3")

            except Exception as error:
                st.error("Something went wrong while generating the story or audio.")
                st.exception(error)

    else:
        st.info("Please upload an image to start.")


if __name__ == "__main__":
    main()
