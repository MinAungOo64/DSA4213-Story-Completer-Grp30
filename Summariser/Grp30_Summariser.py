"""
Summariser Script that takes reads in 'stories_with_outlines_first3000.jsonl' data file,
then summarises the story into a structured json outline.

Ensure these files/folders are placed into the same directory as this script:
1) stories_with_outlines_first3000.jsonl
2) ./Event-summariser-LoRA-v1
3) ./Ending-summariser-LoRA-v2
4) ./QA-LoRA-v2

Follow instructions in README.md, then you can execute this script with:
>   python Grp30_Summariser.py
"""

# ==============================
# SET UP & CONFIGURATIONS
# ==============================
print("Setting up data & required models...")

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from gensim.models import Word2Vec
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import json
import re
import numpy as np
import pandas as pd
import random
import hashlib
import os
import orjson

# Function to check NLTK data packages are available
def ensure_nltk_data_is_ready(packages_to_check, venv_name="venv"):
    # Dynamically construct NLTK data directory path
    # Example path: /path/to/project/venv/nltk_data
    nltk_data_dir = os.path.join(os.getcwd(), venv_name, "nltk_data")
    # Add the custom directory to NLTK's search path so NLTK knows where to look for the data
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    # Create the directory if it doesn't exist (harmless if it does)
    os.makedirs(nltk_data_dir, exist_ok=True)
    print(f"NLTK data directory set to: {nltk_data_dir}\n")
    for package in packages_to_check:
        try:
            # Check if the package is already installed
            # The 'resource' argument for find() needs the full NLTK path format
            nltk.data.find(f"tokenizers/{package}") # Example path for 'punkt'
            # If find() succeeds, the package is installed
            print(f"NLTK package '{package}' already installed. Skipping download.")
        except LookupError:
            # If find() raises a LookupError, the package is NOT installed
            print(f"NLTK package '{package}' not found. Downloading...")
            # Use the download_dir argument to save it to our custom location
            nltk.download(package, download_dir=nltk_data_dir)
            print(f"... Successfully downloaded '{package}'.")       
        except Exception as e:
            # Catch other potential errors during the check/download process
            print(f"An unexpected error occurred while processing '{package}': {e}")

# Function to get optional user input for: story INDEX
def get_valid_storyID(minval=1, maxval=3000):
    valid_range = f"{minval} to {maxval}"
    # input instructions for user
    prompt = f"Please enter an integer within {valid_range}, else random story will be selected: "
    user_input = input(prompt)
    # process the input
    try: 
        story_id = int(user_input)
        if minval <= story_id <= maxval:
            print(f"Story number {story_id} has been chosen.")
            return story_id-1
        else:
            print(f"Number out of valid range: {valid_range}, random story being selected...")
    except ValueError:
        print(f"Empty/invalid input, random story being selected...")
    # assign random story id
    story_id = random.randint(minval, maxval)
    print(f"Story {story_id} has been chosen at random for you.")
    return story_id-1

# Config
DATA_FILE = "stories_with_outlines_first3000.jsonl"
EVENT_MODEL = "./Event-summariser-LoRA-v1"
ENDING_MODEL = "./Ending-summariser-LoRA-v2"
QA_MODEL = "./QA-LoRA-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDEX = None
THRESHOLD = None

# Prepare tokenizers and models
print(f"Loading event summariser model: {EVENT_MODEL}")
event_model = AutoModelForSeq2SeqLM.from_pretrained(EVENT_MODEL).to(DEVICE)
event_tokenizer = AutoTokenizer.from_pretrained(EVENT_MODEL)
print(f"Loading ending summariser model: {ENDING_MODEL}")
ending_model = AutoModelForSeq2SeqLM.from_pretrained(ENDING_MODEL).to(DEVICE)
ending_tokenizer = AutoTokenizer.from_pretrained(ENDING_MODEL)
print(f"Loading QA model: {QA_MODEL}")
qa_model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL).to(DEVICE)
qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)


# ==============================
# DATA PROCESSING
# ==============================
print(f"Loading dataset: {DATA_FILE}")

# Read all lines as JSON
dataset = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        each_line = json.loads(line)
        story = each_line['story']
        outline = each_line['outline']
        dataset.append({"story": story, "outline": outline})


# ==============================
# FUNCTIONS
# ==============================

# helper: function to count sentences
def count_sentences(text):
    sentences = sent_tokenize(text)
    return len(sentences)

# helper: average word vectors for a sentence
def sentence_vector(sentence, model):
    words = [w for w in word_tokenize(sentence.lower()) if w in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[words], axis=0)

# helper: cosine similarity
def cosine_sim(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# MAIN FUNCTION 1: semantic chunking
def split_story_by_similarity(story: str, model: Word2Vec, threshold: float = 0.7) -> list:
    sentences = sent_tokenize(story)
    if len(sentences) <= 1:
        return [story]
    chunks = []
    from_lines = []
    current_chunk = [sentences[0]]
    current_lines = [1]  # Start tracking line numbers from 1
    prev_vec = sentence_vector(sentences[0], model)
    for i in range(1, len(sentences)):
        curr_vec = sentence_vector(sentences[i], model)
        sim = cosine_sim(prev_vec, curr_vec)
        if sim < threshold:
            # new context detected → start new chunk
            chunks.append(" ".join(current_chunk))
            # record line numbers (1-based) of sentences in this chunk
            from_lines.append(", ".join(str(x) for x in current_lines))
            current_chunk = [sentences[i]]
            current_lines = [i + 1]  # Reset with current line number
        else:
            current_chunk.append(sentences[i])
            current_lines.append(i + 1)  # Add line number to current chunk
        prev_vec = curr_vec
    # add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        from_lines.append(", ".join(str(x) for x in current_lines))
    return chunks, from_lines

# helper: generate short event ID
def short_event_id(summary: str) -> str:
    """Deterministic short ID from summary text."""
    h = hashlib.sha1(summary.encode("utf-8")).hexdigest()
    return f"e{h[:1]}_{h[1:3]}"

# helper: convert str into list of str
def text_to_list(text: str):
    # Split by commas
    items = [item.strip() for item in text.split(',')]
    # Remove extra whitespace & capitalize properly
    items = [re.sub(r'\s+', ' ', item).strip().title() for item in items if item.strip()]
    return list(set(items)) # to remove duplicate elements

# helper: generate event summary
def summarise_this_event(event_model, event_tokenizer, chunk: str, device) -> str:
    prompt = f"Summarise this text:\n{chunk}\n"
    inputs = event_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = event_tokenizer.decode(
        event_model.generate(**inputs, max_new_tokens=128)[0], skip_special_tokens=True 
    )
    return outputs

# helper: generate story title
def generate_title(qa_model, qa_tokenizer, story: str, device) -> str:
    title_prompt = f"Question: What is a good title for this story? \nStory: {story}"
    title_input = qa_tokenizer(title_prompt, return_tensors="pt").to(device)
    title_output = qa_tokenizer.decode(
        qa_model.generate(**title_input, max_new_tokens=64)[0], skip_special_tokens=True       
    )
    return title_output

# helper: generate story characters
def generate_char(qa_model, qa_tokenizer, story: str, device) -> list[str]:
    char_prompt = f"Question: Who are the characters in this story? \nStory: {story}"
    char_input = qa_tokenizer(char_prompt, return_tensors="pt").to(device)
    char_output = qa_tokenizer.decode(
        qa_model.generate(**char_input, max_new_tokens=64)[0], skip_special_tokens=True       
    )
    char_output = text_to_list(char_output)
    return char_output

# helper: generate story settings
def generate_setting(qa_model, qa_tokenizer, story: str, device) -> list[str]:
    settings_prompt = f"Question: What are all the settings in this story? \nStory: {story}"
    settings_input = qa_tokenizer(settings_prompt, return_tensors="pt").to(device)
    settings_output = qa_tokenizer.decode(
        qa_model.generate(**settings_input, max_new_tokens=64)[0], skip_special_tokens=True       
    )
    settings_output = text_to_list(settings_output)
    return settings_output

# helper: generate story ending
def generate_ending(ending_model, ending_tokenizer, story: str, device) -> str:
    ending_prompt = f"Extract the ending for this text: \n{story}"
    ending_input = ending_tokenizer(ending_prompt, return_tensors="pt").to(device)
    ending_output = ending_tokenizer.decode(
        ending_model.generate(**ending_input, max_new_tokens=128)[0], skip_special_tokens=True       
    )
    return ending_output

# MAIN FUNCTION 2: returns a structured json schema
def generate_json_outline(story,
                          qa_model=qa_model, qa_tokenizer=qa_tokenizer,
                          ending_model=ending_model, ending_tokenizer=ending_tokenizer,
                          event_model=event_model, event_tokenizer=event_tokenizer,
                          chunk_similarity_threshold=0.2, device=DEVICE):
    # JSON schema to be returned
    summary_json = {
        "title": None,
        "characters": None,
        "settings": None,
        "events": {},
        "sequence": [],
        "ending": None
    }
    # split story into semantically coherent chunks first
    sentences = [word_tokenize(s.lower()) for s in sent_tokenize(story)]
    w2v_model = Word2Vec(sentences, vector_size=150, window=5, min_count=1, workers=2)
    event_chunks, from_lines = split_story_by_similarity(story, w2v_model, chunk_similarity_threshold)
    # populate events and sequence
    for i in range(len(event_chunks)):
        # generate json entry for each event summary
        curr_chunk_summary = summarise_this_event(event_model, event_tokenizer, event_chunks[i], device)
        event_id = short_event_id(curr_chunk_summary)
        summary_json["events"][event_id] = {
            "rev": 1, # revision number will always be 1 for generated outlines
            "summary": curr_chunk_summary,
            # convert from_lines to list of integers
            "from_lines": list(map(int, from_lines[i].split(", ")))
        }
        # add event id to sequence
        summary_json["sequence"].append(event_id)
    # populate title, characters, settings & ending
    title = generate_title(qa_model, qa_tokenizer, story, device)
    char = generate_char(qa_model, qa_tokenizer, story, device) 
    setting = generate_setting(qa_model, qa_tokenizer, story, device)
    ending = generate_ending(ending_model, ending_tokenizer, story, device)
    summary_json["title"] = title
    summary_json["characters"] = char
    summary_json["settings"] = setting
    summary_json["ending"] = {"summary": ending}

    return summary_json

# helper: prints json outline with compacted lists
def print_json_with_compact_lists(data):
    # ORJSON_INDENT_2: Indents the output with 2 spaces.
    # ORJSON_NON_STR_KEYS: Allows non-string keys
    options = orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS
    json_bytes = orjson.dumps(data, option=options)
    json_str = json_bytes.decode('utf-8')
    compacted = re.sub(
        r'\[\s*((?:[^\[\]]|\[[^\[\]]*\])*)\s*\]',
        lambda m: '[' + ' '.join(m.group(1).split()) + ']',
        json_str)
    print(compacted)

# MAIN FUNCTION 3: get optional user input for story chunking similarity threshold
def get_valid_threshold(minval=0, maxval=1):
    valid_range = f"{minval} to {maxval}"
    # input instructions for user
    prompt = f"Please enter a number within {valid_range} for story chunking " \
        "(default/recommended threshold is normally < 0.4), " \
        "else it will be randomly generated: "
    user_input = input(prompt)
    # process the input
    try: 
        threshold = float(user_input)
        if minval <= threshold <= maxval:
            print(f"Threshold: {threshold}, will be used to break down story into chunks for summarising")
            return threshold
        else:
            print(f"Number out of valid range: {valid_range}, random threshold being selected...")
    except ValueError:
        print(f"Empty/invalid input, random threshold being selected...")
    # assign random threshold
    threshold = float(random.randint(minval, 0.4*100)/100)
    print(f"Threshold randomly set to {threshold}.")
    return threshold

# MAIN FUNCTION 4: allows user to continue using the summariser if not instructed to end
def run_interactive_main_loop():
    print("------------------------------------")  
    print("--- Starting Interactive Session ---")  
    print("------------------------------------")   
    while True:
        user_input = input("Press any key to start/continue summarising your story! " \
        "\nElse enter 'close'/'stop'/'exit'/'end' to close this script:")
        # script end
        if user_input in ['stop', 'exit', 'close', 'end']:
            print("\nClosing script... Goodbye!")
            break
        # script summarising loop
        else:
            # get story for summarising
            INDEX = get_valid_storyID()
            story = dataset[INDEX]['story']
            print("------------------------------------------------------------------------") 
            # get threshold for breaking down story into chunks before summary
            THRESHOLD = get_valid_threshold()
            print("------------------------------------------------------------------------") 
            print(f"Extracting JSON summary outline for story {INDEX}...")
            print("------------------------------------------------------------------------\n") 
            json_outline = generate_json_outline(story, chunk_similarity_threshold=THRESHOLD)
            print_json_with_compact_lists(json_outline)


# ==============================
# Main execution
# ==============================

if __name__ == "__main__":
    # define the list of nltk packages needed
    required_packages = ["punkt", "stopwords", "punkt_tab"]
    ensure_nltk_data_is_ready(required_packages)
    # run the main summarising loop
    run_interactive_main_loop()
    


