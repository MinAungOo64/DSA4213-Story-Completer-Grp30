"""
DSA4213 GRP30
python file that stores all the functions required for Summariser and Generator Models
"""

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


# ==============================
# SUMMARISER FUNCTIONS
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
                          qa_model, qa_tokenizer,
                          ending_model, ending_tokenizer,
                          event_model, event_tokenizer,
                          device,
                          chunk_similarity_threshold=0.2):
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

    return summary_json, event_chunks

# ==============================
# GENERATOR FUNCTIONS
# ==============================

# Find good masking, continguos spans of text to mask out
def find_good_masking(input):
    events = input["outline"]["events"]
    good_masking = []
    for eid, event in events.items():
        if is_contiguous(event["from_lines"]):
            good_masking.append((eid, event["from_lines"]))
    return good_masking

# helper: Check if line indices are contiguous
def is_contiguous(line_indices):
    sorted_indices = sorted(line_indices)
    return all(b - a == 1 for a, b in zip(sorted_indices, sorted_indices[1:]))

# helper: mask a span of sentences
def mask_span(story_lines, from_lines):
    mask = []
    mask.append("[MASK_START]")
    # 0 based
    start = from_lines[0] - 1
    end = from_lines[-1] - 1
    mask.extend(story_lines[start:end+1])
    mask.append("[MASK_END]")
    return mask

# helper: Extract story into list of sentences
def prepare_sentences(input):
    # Extract story into list of sentences
    # Remove \n 
    story_text = input["story"].replace("\n", " ").strip()
    # Split by ". " , "! " to get sentences
    story_lines = [line.strip() + '.' for line in re.split(r'(?<=[.!?]) +', story_text) if line]
    story_lines = [re.sub(r'([.!?])\1+$', r'\1', line) for line in story_lines]
    return story_lines

# Add event function for insertion and modification masking
def add_event(input, event_id):

    masked_story_lines = []
    # Add conditioning token <ADD>
    masked_story_lines.append("<ADD>")

    # Add the outline summary
    masked_story_lines.append("<START_OUTLINE>")
    for eid in input["outline"]["sequence"]:
        summary = input["outline"]["events"][eid]["summary"].strip()
        masked_story_lines.append("<BOE>")  # Beginning of event
        masked_story_lines.append(summary)
        masked_story_lines.append("<EOE>")  # End of event
    masked_story_lines.append("<END_OUTLINE>")

    # Add the story lines
    masked_story_lines.append("<START_STORY>")
    story_lines = prepare_sentences(input)
    # Copy story_lines until from_lines, then add [MASK], then copy rest of story_lines
    # Mask out the event lines by adding [MASK_START] and [MASK_END] between the ground truth
    from_lines = input["outline"]["events"][event_id]["from_lines"]
    masked_span = mask_span(story_lines, from_lines)
    start = from_lines[0] - 1
    end = from_lines[-1] - 1
    masked_story_lines.extend(story_lines[:start])
    masked_story_lines.extend(masked_span)
    masked_story_lines.extend(story_lines[end+1:])
    masked_story_lines.append("<END_STORY>")

    # Join the masked_story_lines into a single input text
    input_text = " ".join(masked_story_lines)
    
    # ground truth label
    target_text = masked_span[1:-1]  # Exclude [MASK_START] and [MASK_END]
    target_text = " ".join(target_text)

    return input_text, target_text


# Delete event function for deletion masking
def delete_event(input, event_id):

    masked_story_lines = []
    # Add conditioning token <DELETE>
    masked_story_lines.append("<DELETE>")

    # Add the outline summary
    masked_story_lines.append("<START_OUTLINE>")
    for eid in input["outline"]["sequence"]:
        # Skip the event to be deleted
        if eid == event_id:
            continue
        summary = input["outline"]["events"][eid]["summary"].strip()
        masked_story_lines.append("<BOE>")  # Beginning of event
        masked_story_lines.append(summary)
        masked_story_lines.append("<EOE>")  # End of event
    masked_story_lines.append("<END_OUTLINE>")

    # Add the story lines
    masked_story_lines.append("<START_STORY>")
    story_lines = prepare_sentences(input)
    # Copy story_lines until from_lines, then add [MASK], then copy rest of story_lines
    # Mask out the event lines by adding [MASK_START] and [MASK_END] between the ground truth
    from_lines = input["outline"]["events"][event_id]["from_lines"]
    masked_span = mask_span(story_lines, from_lines)
    start = from_lines[0] - 1
    end = from_lines[-1] - 1
    masked_story_lines.extend(story_lines[:start])
    masked_story_lines.extend(masked_span)
    masked_story_lines.extend(story_lines[end+1:])
    masked_story_lines.append("<END_STORY>")

    # Join the masked_story_lines into a single input text
    input_text = " ".join(masked_story_lines)
    
    # ground truth label
    target_text = masked_span[1:-1]  # Exclude [MASK_START] and [MASK_END]
    target_text = " ".join(target_text)

    return input_text, target_text


# MAIN: Generate function (supports overrides) -----
def generate_masked_span(model, tokenizer, text, device, **override):
    # Default precise config for generation, no randomness
    DECODE_CFG = {
        "max_new_tokens": 128,
        "min_new_tokens": 50,
        "num_beams": 5,
        "length_penalty": 2,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "do_sample": False,
        "temperature": 0.7, # ignored if do_sample=False
        "top_p": 0.9, # ignored if do_sample=False
    }
    cfg = {**DECODE_CFG, **override}  # merge defaults + overrides
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    ids = model.generate(**inputs, **cfg)
    outputs = tokenizer.decode(ids[0], skip_special_tokens=True)
    return outputs

