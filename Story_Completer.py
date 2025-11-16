"""
Main Script that takes reads in 'stories_with_outlines_first3000.jsonl' data file,
1. summarises the story into a structured json outline.
2. regenerates selected events that have been chosen for edit.

Follow instructions in README.md, then you can execute this main script with:
>   python Story_Completer.py
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
from StoryCompleterFunctions import (
    split_story_by_similarity,
    generate_json_outline,
    find_good_masking,
    add_event, 
    delete_event,
    generate_masked_span,
)

# Config
DATA_FILE = "./Summariser/stories_with_outlines_first3000.jsonl"
EVENT_MODEL = "./Summariser/Event-summariser-LoRA-v1"
ENDING_MODEL = "./Summariser/Ending-summariser-LoRA-v2"
QA_MODEL = "./Summariser/QA-LoRA-v2"
GENERATOR_MODEL = "./Generator/full_model_best"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDEX = None            # random story ID to be filled later
THRESHOLD = None        # threshold value to split story by semantic similarity
OUTLINE_SUMMARY = None  # placeholder for summariser output later
BALANCED_CFG = dict(    # Balanced config for generation with some randomness
    max_new_tokens=128,
    min_new_tokens=50,
    num_beams=1,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    length_penalty=1.0,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
)
CREATIVE_CFG = dict(    # Creative config for generation with more randomness
    max_new_tokens=150,
    min_new_tokens=50,
    num_beams=1,
    do_sample=True,
    temperature=1.1,
    top_p=0.95,
    length_penalty=1.0,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
)

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
print(f"Loading generator model: {GENERATOR_MODEL}")
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL).to(DEVICE)
gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)


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

# MAIN FUNCTION: get optional user input for story chunking similarity threshold
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

# MAIN FUNCTION: get valid user input to generate new event
def get_valid_event_mask(data_entry):
    valid_eventIDs = [i[0] for i in find_good_masking(data_entry)]  # returns a list of valid event IDs
    # loop till valid event ID is entered
    while True:
        prompt1 = "Enter the event ID you want to edit:"
        user_input = input(prompt1)
        if user_input in valid_eventIDs:
            break
        else:
            print(f"{user_input} is not a valid event ID. \nValid options are: {valid_eventIDs}")
    print("------------------------------------------------------------------------\n") 
    # loop till valid action is entered
    while True:
        prompt2 = "Enter whether you want to 'modify'/'delete' this event:"
        action = input(prompt2).lower()
        if action == "modify":
            input_text, target_text = add_event(data_entry, user_input)
            break
        elif action == 'delete':
            input_text, target_text = delete_event(data_entry, user_input)
            break
        else: 
            print("Invalid action. Please type either 'modify' or 'delete'.")
    print("------------------------------------------------------------------------\n") 
    output = generate_masked_span(gen_model, gen_tokenizer, input_text, DEVICE)
    print(f"Predicted text to be modified/removed: {output}")
    print(f"Original text: {target_text}")


# MAIN FUNCTION: allows user to continue using the script if not instructed to end
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
            json_outline = generate_json_outline(story,
                                                 qa_model, qa_tokenizer,
                                                 ending_model, ending_tokenizer,
                                                 event_model, event_tokenizer,
                                                 device=DEVICE,
                                                 chunk_similarity_threshold=THRESHOLD)
            print_json_with_compact_lists(json_outline)
            # shift original outline under a new key: "old_outline"
            dataset[INDEX]["old_outline"] = dataset[INDEX]["outline"]
            dataset[INDEX]["outline"] = json_outline
            # replace generated summary outline with
            print("------------------------------------------------------------------------\n") 
            # select the event id to edit, then generate mask
            get_valid_event_mask(dataset[INDEX])
            print("------------------------------------------------------------------------\n") 


# ==============================
# Main execution
# ==============================

if __name__ == "__main__":
    # define the list of nltk packages needed
    required_packages = ["punkt", "stopwords", "punkt_tab"]
    ensure_nltk_data_is_ready(required_packages)
    # run the main summarising loop
    run_interactive_main_loop()
    


