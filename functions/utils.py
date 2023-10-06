"""
This file contains functions shared across notebooks.
"""

import os
import glob
import re
import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import json


def extract_chords(text):
    """Extracts the chords given a text. The text variable is a string representation of the kern files found in the dataset. It also 
    combines the sections that are on top of the fuile (the sequnce element). A sequnce element looks like this:
    *>[A,A2,B,A3]
    it is not a must (there are some files without it). Before each chord, there is a time information, e.g. 2C:min. This function will
    not discard it, but repeat the chord 2 times.
    Since we now consier time inforamtion this is very important:
    In our dataset, the following rythms occur:
    {'4/4': 1099, '3/4': 79, '6/4': 3, '5/4': 3, '3/2': 1, '6/8': 1}
    Since we have very dominant 4/4 percentage, We will "normalize everything to 4/4. When we have a 3/4 beat played, the content of one measure (or bar) may look like this:
    2.Amin7
    This means a half note dotted (half dutation of the note longer). So it would be a third in duration.
    """
    lines = text.strip().split("\n")
    chords_sections = {}
    sequnce_element = ""
    chords_no_element = []
    
    for line in lines:
        # first check for the sequnce element (eg [A, A1, ...]
        if line.startswith("*>["):
            sequnce_element = line[3:-1].split(",")
            # sometimes, the files do have one single sequunce inforamtion but no header (the thing checkt in the elif down below). This avoids this problem
            if len(sequnce_element) == 1:
                
                current_section = sequnce_element[0]
                chords_sections[current_section] = []

        elif line.startswith("*>"):
            current_section = line[2:]
            chords_sections[current_section] = []
            
            
            
        # check if we have digit (aka duration in front) 
        if line[0].isdigit():
            
            if sequnce_element != "":
                chords_sections[current_section] = chords_sections[current_section] + [line.split("(")[0]]
            #     duration = line[0]
            #     # check if we have a dotted note
            #     if line[1] == ".":
            #         # split("(") removes subsitute chord        #we add the time plus the dot if so to the chord information
            #         chords_sections[current_section] = chords_sections[current_section] + [duration + "." + line[2:].split("(")[0]]
            #     else:
            #         # duration added, no dot
            #         chords_sections[current_section] = chords_sections[current_section] + [duration + line[1:].split("(")[0]]
            
            # no sequnce element
            else:
                chords_no_element.append(line.split("(")[0])
                # if line[1] == ".":
                #     # split("(") removes subsitute chord
                #     chords_no_element.append(line[2:].split("(")[0])
                # else:
                #     chords_no_element.append(line[1:].split("(")[0])
            

    # order the chords accordingly
    final_sequnce = []
    for n in sequnce_element:
        final_sequnce += chords_sections[n]

    # prune that ONE outlier 
    if len(final_sequnce) > 300:
        return final_sequnce[:200]
    
    if sequnce_element != "": return final_sequnce
    else: return chords_no_element


def flatten_chords(chords:list):
    """Inputs a list of chords with time infraomtion in front. Takes this time infomation and converts
    it to a seqcune format , eg. 1B:min -> [B:min, B:min, B:min, B:min].
    Important: The final rythm will be 4/4 since our dataset has most datapoints in this rythm.
    """
    # defines how many times a chord is repeated in our 1/4 teimstep. since one step at the rnn is 1/4 bars, a duration of 1 will be played 4 time steps into the rnn
    repeats = {'1': 4, '2': 2, '4': 1, '2.': 3, '4.': 1, '1.': 6, '8': 4}
    arranged_chords = []
    for chord in chords:
        # check if we have a dotted note
        if chord[1] == ".":
            arranged_chords.extend([chord[2:]] * repeats[chord[:2]])

        else:
            arranged_chords.extend([chord[1:]] * repeats[chord[0]])

    return arranged_chords
                


def extract_signature(text):
    # Using regular expression to find the pattern *M followed by a time signature
    match = re.search(r'\*M(\d+/\d+)', text)
    if match:
        return match.group(1)
    else:
        return None



## Chord Simplification Visualization pipeline ----------------------------------------------------------------------------------------------------------------------------

def visualize_chord_simplification(chords, ChordSimplifier):
    """This function takes chords, which is a list of chords with ONE chord of each kind (no repetition), 
    compares the before and after of the simlification, and displays the results in a way that is readable by humans. This is very helpful
    when debugging or verifying that the simplification procedures were successful."""
    
    # Helper function to format complicated chords
    def format_complicated_chords(simple_chord, comp_chords):
        MAX_WIDTH = 80
        base_indent = len("Simplified Chord | ")
        max_indent = 40  # Maximum threshold for indentation
        indent = min(len(simple_chord) + base_indent, max_indent)
        comp_chords_str = ', '.join(comp_chords)
        
        # Split the complicated chords string if it exceeds the max width
        formatted_chords = []
        while comp_chords_str:
            if len(simple_chord + " | " + comp_chords_str) <= MAX_WIDTH:
                formatted_chords.append(comp_chords_str)
                break
            split_index = comp_chords_str.rfind(',', 0, MAX_WIDTH - len(simple_chord + " | "))
            formatted_chords.append(comp_chords_str[:split_index])
            comp_chords_str = comp_chords_str[split_index + 2:]

        return formatted_chords

    # Using the updated ChordSimplifier to simplify chords
    simplifier = ChordSimplifier()
    simplified_chords = {}
    for chord in chords:
        simplified = simplifier.simplify_chord(chord)
        # Convert boolean values to "Invalid/No Chords" string
        if isinstance(simplified, bool):
            simplified = "Invalid/No Chords"
        if simplified not in simplified_chords:
            simplified_chords[simplified] = []
        simplified_chords[simplified].append(chord)

    # Textual visualization
    output_lines = []
    output_lines.append("Simplified Chord   | Complicated Chords")
    output_lines.append("--------------------------------------------")
    for simple_chord, complicated_chords in simplified_chords.items():
        comp_chords_list = format_complicated_chords(simple_chord, complicated_chords)
        output_lines.append(f"{simple_chord:<18} | {comp_chords_list[0]}")
        for comp_chord in comp_chords_list[1:]:
            output_lines.append(f"{' ':<18} | {comp_chord}")
        output_lines.append("--------------------------------------------")

    # Return the visualization
    return '\n'.join(output_lines)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# CHORD ENCODING
def encode_chords(all_chords:list):
    """takes all chords as 2d list (where every list is a list of string chords like "A:min", "B:min, ...)and returns the following: 
    - chord vocab
    - chord_to_idx: given a chord, a index is returned
    - idx_to_chord: given a index, a chord is returned
    this is usefull for encoding and decoding later
    - padded_sequnces: padded sequnces using pytorchs padded seqnces functin
    - vocab size
    """
    # Create a vocabulary of unique chords:
    chord_vocab = sorted(set(chord for chord_sequence in all_chords for chord in chord_sequence))

    # Create a mapping from chords to indices:
    chord_to_idx = {chord: idx+1 for idx, chord in enumerate(chord_vocab)}
    idx_to_chord = {i+1: chord for i, chord in enumerate(chord_vocab)}
    
    # Encode the chord sequences using indices and pad them to the same length:
    max_length = max(len(chord_sequence) for chord_sequence in all_chords)
    # OR: Choose it by myself, long sequnces can be a problem ( a lot of padding)
    # max_length = 40 # 100
    
    # Convert the chord sequences to encoded tensors
    encoded_sequences = []
    for chord_sequence in all_chords:
        encoded_sequence = [chord_to_idx[chord] for chord in chord_sequence[:max_length]]
        encoded_sequences.append(torch.tensor(encoded_sequence))
    
    # Pad the sequences to the same length
    padded_sequences = pad_sequence(encoded_sequences, batch_first=True)
    
    # Compute the lengths of each sequence
    lengths = [len(sequence) for sequence in encoded_sequences]
    
    # Vocab Size 
    vocab_size = padded_sequences.max().item() + 1
    
    # add the pad tokens to both 
    chord_to_idx['pad'] = 0
    idx_to_chord[0] = 'pad'

    return chord_vocab, chord_to_idx, idx_to_chord, padded_sequences, vocab_size


