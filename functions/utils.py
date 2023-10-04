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



## Chord Simplification pipeline ----------------------------------------------------------------------------------------------------------------------------

def check_chord(chord:str=None):
    original_chord = chord # for debugging
    #first check if its a chord at all
    if not is_chord(chord): return False
        
    # then get root for later
    root_note = get_root(chord)
    if root_note == "C-":
        root_note = "B"

    # now, we sometimes have werid symbols in the chords (playstlyes, arpegio or harmonic indicators. we will just remove them)
    # A list of possible "playstyle" symbols
    playstyle_symbols = ["#", "o", "h", "^", "*", ";", "+"]
    
    # Removing playstyle symbols from the chord
    for symbol in playstyle_symbols:
        chord = chord.replace(symbol, "")
        
    # now check again; some
    if not is_chord(chord): return False

    # Search for "min"
    if re.search(r'min', chord):
        #print(f'Found min in {chord}')
        return root_note + ":min"
    
    # Search for "maj"
    if re.search(r'maj', chord):
        #print(f'Found maj in {chord}')
        return root_note + ":maj"
    
    # Search for chords with a number
    if re.search(r'[A-G]-?\d', chord):
        #print(f'Found chord with a number in {chord}')
        return root_note + ":maj"

    # Search for slash chords - gives us a major
    if re.search(r'/', chord):
        #print(f'Found slash chord in {chord}')
        return root_note + ":maj"

    # Search for add9|sus|aug chords
    if re.search(r'(add9|sus|aug)', chord):
        #print(f'Found dim, sus, aug chord in {chord}')
        return root_note + ":maj"

    # search for dim
    if re.search(r'(dim)', chord):
        #print(f'Found dim, sus, aug chord in {chord}')
        return root_note + ":min"

    # this seems like a annotation issue, but here is the case nontheless (D-:7, B-:7)
    if re.search(r'[A-G]-?:', chord):
        #print(f'Found chord with colon after the root note in {chord}')
        return root_note + ":maj"

    # now, some chords are not found, they cant be -> more aggresive chord checking:
    if len(chord) <= 2: return False

    raise ValueError("No case found!")


def get_root(chord):
    # extracts the root - becuase chords can eiter be "normal" like G, C or flat: D-, G-, ....
    if chord[1] == '-':  # checks if the second character is '-' which indicates a flat
        return chord[:2]  # return the first two characters (the flat note)
    elif re.search(r'^[A-G]#', chord):        
        return transpose_chord(chord) 
        
    else:
        return chord[0]   # return the first character (the single note)


def transpose_chord(chord):
    # some chords in the dataset are sharps. Since we want to reduce vocab size, we will aslo prune them.
    white_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

    if '#' not in chord:
        raise ValueError("Function triggered without need")

    note = chord[0]
    if note in white_keys:
        return white_keys[(white_keys.index(note) + 1) % len(white_keys)] + "-"
    else:
        raise ValueError("Note does not exist")


def is_chord(chord):
    # A list of possible symbols that indiate certian playstyles like glissandro, harmonic (taken from dataset docs)
    playstyle_symbols = ["-", "#", "o", "h", "^"]

    # ignore rest chords
    if "r" in chord[:2]:
        return False
        
    # some dingus used C- (C flat) in the dataset sometimes. We just replace it to B
    chord = re.sub(r'C-', 'B', chord)
    
    # Check if chord contains any playstyle symbol with a note (it's a note, not a chord)
    if len(chord) <= 2 and re.match(r'^[A-G](' + '|'.join(map(re.escape, playstyle_symbols)) + ')?$', chord):
        return False

    # Otherwise, it should be a chord
    return True


def batch_simplify_chord(chords:list=None):
    simple_chords = []
    for chord in chords:
        s_chord = check_chord(chord)
        if s_chord: simple_chords.append(s_chord)

    return simple_chords

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


