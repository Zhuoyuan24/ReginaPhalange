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
from collections import Counter

import streamlit as st
from PIL import Image
import torch
from gtts import gTTS

from transformers import (
    pipeline,
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

# Professor-provided image captioning model
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

# Extra image-understanding models
VQA_MODEL = "dandelin/vilt-b32-finetuned-vqa"
OBJECT_DETECTION_MODEL = "facebook/detr-resnet-50"

# Improved story model
# FLAN-T5 is better at following instructions than the previous story model.
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
    Load the BLIP image captioning model directly.

    We do not use pipeline("image-to-text") because some current
    Transformers environments no longer recognize that task name.
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
    Load the object detection model.

    Note:
    facebook/detr-resnet-50 requires timm in requirements.txt.
    """
    object_detection_model = pipeline(
        task="object-detection",
        model=OBJECT_DETECTION_MODEL,
        device=get_pipeline_device(),
    )

    return object_detection_model


@st.cache_resource(show_spinner="Loading FLAN-T5 story model...")
def load_story_model():
    """
    Load FLAN-T5 for instruction-following story generation.

    This uses AutoTokenizer and AutoModelForSeq2SeqLM directly instead of
    pipeline("text2text-generation"), because some environments may not list
    text2text-generation as an available pipeline task.
    """
    device = get_torch_device()

    tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(STORY_MODEL)

    model.to(device)
    model.eval()

    return tokenizer, model, device


def clean_short_answer(answer: str, fallback: str = "unknown") -> str:
    """
    Clean short VQA answers.
    """
    if answer is None:
        return fallback

    answer = str(answer).strip().lower()

    bad_answers = ["", "unknown", "none", "no", "nothing", "n/a"]
    if answer in bad_answers:
        return fallback

    return answer


def generate_basic_caption(image: Image.Image) -> str:
    """
    Generate a basic caption using the BLIP model.
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
    Ask a visual question about the image.
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


def add_article(noun: str) -> str:
    """
    Add a simple article to object labels.

    Example:
        ball -> a ball
        umbrella -> an umbrella
    """
    noun = noun.strip().lower()

    if not noun:
        return ""

    if noun.startswith(("a ", "an ", "the ")):
        return noun

    if noun[0] in ["a", "e", "i", "o", "u"]:
        return f"an {noun}"

    return f"a {noun}"


def format_object_list(labels) -> str:
    """
    Format object labels naturally without counts.

    Example:
        ["ball", "skateboard"] -> "a ball and a skateboard"
    """
    if not labels:
        return "none"

    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

    unique_labels = unique_labels[:3]
    article_labels = [add_article(label) for label in unique_labels]

    if len(article_labels) == 1:
        return article_labels[0]

    if len(article_labels) == 2:
        return f"{article_labels[0]} and {article_labels[1]}"

    return f"{article_labels[0]}, {article_labels[1]}, and {article_labels[2]}"


def detect_story_objects(image: Image.Image, score_threshold: float = 0.80) -> str:
    """
    Detect useful story objects.

    This version avoids listing people counts because it sounds unnatural
    in the story.
    """
    try:
        object_detector = load_object_detection_pipeline()
        detections = object_detector(image)

        object_labels = []

        for item in detections:
            label = item.get("label", "").lower().strip()
            score = item.get("score", 0)

            if not label or score < score_threshold:
                continue

            # Avoid unnatural object lines like "1 person"
            if label in ["person", "people"]:
                continue

            # Make COCO label sound more natural
            if label == "sports ball":
                label = "ball"

            object_labels.append(label)

        return format_object_list(object_labels)

    except Exception:
        return "none"


def choose_character(caption: str, people: str, main_subject: str) -> str:
    """
    Choose a natural character phrase from the image information.
    """
    combined = f"{caption} {people} {main_subject}".lower()

    if "children" in combined or "kids" in combined:
        return "children"

    if "woman" in combined or "girl" in combined:
        return "a woman"

    if "man" in combined or "boy" in combined:
        return "a man"

    if "child" in combined:
        return "a child"

    if "group" in combined or "people" in combined:
        return "a group of people"

    if "person" in combined:
        return "a person"

    return "someone"


def choose_setting(caption: str, place: str) -> str:
    """
    Convert location information into a natural setting phrase.
    """
    combined = f"{caption} {place}".lower()

    if "gym" in combined:
        return "in a gym"

    if "park" in combined:
        return "in a park"

    if "snow" in combined or "mountain" in combined:
        return "on a snowy mountain"

    if "street" in combined:
        return "on a street"

    if "beach" in combined:
        return "at a beach"

    if "room" in combined or "inside" in combined or "indoor" in combined:
        return "inside a room"

    if "outside" in combined or "outdoor" in combined:
        return "outside"

    if place not in ["unknown", "none", ""]:
        return f"in {place}"

    return "in a nice place"


def choose_activity(caption: str, activity: str) -> str:
    """
    Convert activity information into a natural action phrase.
    """
    combined = f"{caption} {activity}".lower()

    if "basketball" in combined:
        return "playing basketball"

    if "snowboard" in combined:
        return "snowboarding"

    if "ski" in combined:
        return "skiing"

    if "walk" in combined:
        return "walking together"

    if "run" in combined:
        return "running around"

    if "play" in combined:
        return "playing"

    if "laying in the snow" in combined or "lying in the snow" in combined:
        return "resting in the snow"

    if activity not in ["unknown", "none", ""]:
        if activity.endswith("ing"):
            return activity
        return f"enjoying {activity}"

    return "enjoying the moment"


def img2text(url: str) -> str:
    """
    Convert uploaded image into a concise scenario.

    This follows the professor's required function style:

        def img2text(url):
            ...
            return text

    Improvement:
    The scenario is now natural and story-friendly.
    It no longer lists object counts or unreliable mood labels.
    """
    image = Image.open(url).convert("RGB")

    # 1. Basic caption
    caption = generate_basic_caption(image)

    # 2. Extra visual details
    main_subject = ask_vqa_question(image, "What is the main subject of the image?")
    people = ask_vqa_question(image, "Who is in the picture?")
    place = ask_vqa_question(image, "Where is the scene?")
    activity = ask_vqa_question(image, "What is happening in the image?")

    # 3. Useful objects, without people counts
    objects = detect_story_objects(image)

    # 4. Normalize into natural story facts
    character = choose_character(caption, people, main_subject)
    setting = choose_setting(caption, place)
    action = choose_activity(caption, activity)

    if objects == "none":
        object_sentence = "There are no extra important objects needed for the story."
    else:
        object_sentence = f"An important visual detail is {objects}."

    text = (
        f"Character: {character}. "
        f"Setting: {setting}. "
        f"Activity: {action}. "
        f"{object_sentence} "
        f"Original image caption: {caption}."
    )

    return text


def parse_scenario_details(text: str) -> Dict[str, str]:
    """
    Parse the scenario text into fields.
    """
    fields = {
        "character": "someone",
        "setting": "in a nice place",
        "activity": "enjoying the moment",
        "objects": "none",
        "caption": "a simple scene",
    }

    patterns = {
        "character": r"Character:\s*(.*?)(?:\. Setting:|$)",
        "setting": r"Setting:\s*(.*?)(?:\. Activity:|$)",
        "activity": r"Activity:\s*(.*?)(?:\. An important visual detail is|\. There are no extra|$)",
        "objects": r"An important visual detail is\s*(.*?)(?:\. Original image caption:|$)",
        "caption": r"Original image caption:\s*(.*?)(?:\.|$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value:
                fields[key] = value

    return fields


def get_character_grammar(character: str) -> Dict[str, str]:
    """
    Return grammar settings for the character.

    This prevents errors like:
        "man were"
        "woman were"
    """
    character = character.strip().lower()

    if character in ["children", "people", "a group of people"]:
        return {
            "subject": character,
            "be": "were",
            "pronoun": "they",
            "pronoun_cap": "They",
        }

    if character in ["a man", "man"]:
        return {
            "subject": "a man",
            "be": "was",
            "pronoun": "he",
            "pronoun_cap": "He",
        }

    if character in ["a woman", "woman"]:
        return {
            "subject": "a woman",
            "be": "was",
            "pronoun": "she",
            "pronoun_cap": "She",
        }

    if character in ["a child", "child"]:
        return {
            "subject": "a child",
            "be": "was",
            "pronoun": "the child",
            "pronoun_cap": "The child",
        }

    return {
        "subject": character,
        "be": "was",
        "pronoun": "they",
        "pronoun_cap": "They",
    }


def build_story_prompt(text: str) -> str:
    """
    Build a strict image-grounded prompt for FLAN-T5.
    """
    details = parse_scenario_details(text)

    prompt = f"""
Write one complete short story for children aged 3 to 10.

Use these image facts naturally. Do not list the facts.

Image facts:
Character: {details["character"]}
Setting: {details["setting"]}
Activity: {details["activity"]}
Visual detail: {details["objects"]}
Caption: {details["caption"]}

Story rules:
- 50 to 90 words
- clear beginning, middle, and happy ending
- simple child-friendly English
- directly related to the image facts
- no scary, violent, romantic, or adult content
- do not mention "image", "caption", or "visual detail"

Story:
"""
    return prompt.strip()


def extract_story_only(raw_story: str, prompt: str) -> str:
    """
    Clean raw model output.
    """
    story = raw_story.replace(prompt, "").strip()

    if "Story:" in story:
        story = story.split("Story:")[-1].strip()

    unwanted_labels = [
        "Image facts:",
        "Character:",
        "Setting:",
        "Activity:",
        "Visual detail:",
        "Caption:",
        "Beginning:",
        "Middle:",
        "Ending:",
        "Answer:",
        "Output:",
    ]

    for label in unwanted_labels:
        story = story.replace(label, "")

    story = re.sub(r"\s+", " ", story).strip()

    return story


def keep_complete_sentences(story: str) -> str:
    """
    Keep only complete sentences ending with punctuation.
    """
    sentence_matches = re.findall(r"[^.!?]+[.!?]", story)

    if not sentence_matches:
        return story.strip()

    complete_story = " ".join(sentence.strip() for sentence in sentence_matches)
    return complete_story.strip()


def limit_story_length(story: str, max_words: int = 100) -> str:
    """
    Limit story length while keeping it readable.
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


def story_has_bad_content(story: str) -> bool:
    """
    Check if the story contains unsuitable content.
    """
    bad_words = [
        "wife",
        "fiancé",
        "fiance",
        "husband",
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

    story_lower = story.lower()

    for word in bad_words:
        if word in story_lower:
            return True

    return False


def story_is_too_generic(story: str, text: str) -> bool:
    """
    Check whether the story uses enough image-specific details.
    """
    details = parse_scenario_details(text)
    story_lower = story.lower()

    keywords = []

    for field in ["character", "setting", "activity", "objects"]:
        value = details.get(field, "")
        words = re.findall(r"[a-zA-Z]+", value.lower())

        for word in words:
            if len(word) >= 4 and word not in ["with", "there", "important", "visual", "detail"]:
                keywords.append(word)

    if not keywords:
        return False

    matched = 0
    for keyword in set(keywords):
        if keyword in story_lower:
            matched += 1

    return matched < 2


def make_simple_backup_story(text: str) -> str:
    """
    Create a reliable, image-specific backup story.

    This is only used if the model output is too short, unsafe, or unrelated.
    """
    details = parse_scenario_details(text)
    grammar = get_character_grammar(details["character"])

    subject = grammar["subject"]
    be = grammar["be"]
    pronoun = grammar["pronoun"]
    pronoun_cap = grammar["pronoun_cap"]

    setting = details["setting"]
    activity = details["activity"]
    objects = details["objects"]

    if objects != "none":
        object_sentence = f"{pronoun_cap} noticed {objects} nearby."
    else:
        object_sentence = f"{pronoun_cap} looked around carefully."

    story = (
        f"One day, {subject} {be} {setting}. "
        f"{pronoun_cap} {be} {activity}, and the moment felt like a small adventure. "
        f"{object_sentence} "
        f"{pronoun_cap} tried carefully, smiled, and kept going. "
        f"Soon, the day felt bright and special. "
        f"By the end, {pronoun} felt proud because every simple moment can become a happy story."
    )

    return limit_story_length(story, max_words=100)


def clean_story_text(raw_story: str, prompt: str, text: str) -> str:
    """
    Clean and validate the generated story.
    """
    story = extract_story_only(raw_story, prompt)
    story = keep_complete_sentences(story)
    story = limit_story_length(story, max_words=100)

    word_count = len(story.split())

    if word_count < 50:
        story = make_simple_backup_story(text)

    if story_has_bad_content(story):
        story = make_simple_backup_story(text)

    if story_is_too_generic(story, text):
        story = make_simple_backup_story(text)

    return story


def text2story(text: str) -> str:
    """
    Convert image scenario text into a complete short story.

    This follows the professor's required function style:

        def text2story(text):
            story_text = ""
            return story_text
    """
    tokenizer, model, device = load_story_model()
    prompt = build_story_prompt(text)

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
            max_new_tokens=130,
            num_beams=4,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    raw_story = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    story_text = clean_story_text(
        raw_story=raw_story,
        prompt=prompt,
        text=text,
    )

    return story_text


def prepare_text_for_audio(story_text: str) -> str:
    """
    Prepare text for smoother TTS reading.
    """
    story_text = re.sub(r"\s+", " ", story_text).strip()

    # Add small spacing after punctuation to help speech rhythm.
    story_text = story_text.replace(". ", ".  ")
    story_text = story_text.replace("! ", "!  ")
    story_text = story_text.replace("? ", "?  ")

    return story_text


def text2audio(story_text: str) -> io.BytesIO:
    """
    Convert story text into MP3 audio using gTTS.

    This follows the professor's required function style:

        def text2audio(story_text):
            audio_data = ""
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
    Save uploaded image locally, following the professor's example.
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
    st.sidebar.caption("Story: google/flan-t5-base")
    st.sidebar.caption("Audio: gTTS")


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
                    audio_data = text2audio(story)

                st.subheader("3. Story Audio")
                st.audio(audio_data, format="audio/mp3")

                st.success("Done! The image has been converted into a story and audio.")

            except Exception as error:
                st.error("Something went wrong while generating the story or audio.")
                st.exception(error)

    else:
        st.info("Please upload an image to start.")


if __name__ == "__main__":
    main()
