import re

class ChordSimplifier:
    def __init__(self):
        self.JAZZ5_KINDS = ["maj", "min", "maj7", "min7", "dom", "hdim7", "dim"]
        self.JAZZ5_MIREX_KINDS = [":maj", ":min", ":maj7", ":min7", ":7", ":hdim7", ":dim7"]
        self.playstyle_symbols = ["^", "*", ";", "+"]
        self.not_found = []
        # h end glissando
        # o harmonic
        # ^ accent mark
        # ; pause sign
        # + undefined, user assignable

    def _get_root(self, chord):
        if chord[1] == '-':
            return chord[:2]
        elif chord[1] == '#':
            return chord[:2]
        else:
            return chord[0]
    
    def _is_chord(self, chord):
        playstyle_symbols = ["-", "#", "^"]
        if "r" in chord[:2]:
            return False
        chord = re.sub(r'C-', 'B', chord)
        if len(chord) <= 2 and re.match(r'^[A-G](' + '|'.join(map(re.escape, playstyle_symbols)) + ')?$', chord):
            return False
        return True

    def _chop_chord(self, chord:str = None):
        for symbol in self.playstyle_symbols:
            chord = chord.replace(symbol, "")
        return chord

    def extract_chord_quality(self, chord):
        quality_list = [
            ("maj7", self.JAZZ5_MIREX_KINDS[2]),
            ("min7", self.JAZZ5_MIREX_KINDS[3]),
            ("h", self.JAZZ5_MIREX_KINDS[5]), # h-dim and h-dim 7th to hdim7
            ("o", self.JAZZ5_MIREX_KINDS[6]), # dimished and dimished seventh seventh to dim7
            ("7", self.JAZZ5_MIREX_KINDS[4]),
            ("maj", self.JAZZ5_MIREX_KINDS[0]),
            ("min", self.JAZZ5_MIREX_KINDS[1]),
        ]
        for quality, chord_type in quality_list:
            if quality in chord:
                return chord_type
        return None

    def simplify_chord(self, chord: str = None):
        if not self._is_chord(chord): 
            return "Invalid/No Chord"
        root_note = self._get_root(chord)
        chord = self._chop_chord(chord)
        if not self._is_chord(chord): 
            return "Invalid/No Chord"

        chord_type = self.extract_chord_quality(chord)
        if chord_type is not None:
            return root_note + chord_type

        # Enhanced regex patterns to handle extended chords, altered chords, and slash chords
        # (1) Basic chords with possible alterations and extensions - matches chords like C7, F#9, and Ab13.
        basic_chords = r'[A-G]#?-?\d{1,2}'
        # Chords with specific extensions -  matches chords like Cadd9, D#sus4, Faug, Gdim, A#11, Eb13b9, etc.
        extended_chords = r'[A-G]#?(add|sus|aug|dim|\d{0,2}(#5|b5|#9|b9))'
        #  matches chords like D/F#, G/B, Cm/Eb, etc.
        slash_chords = r'[A-G](#|-)?(/[A-G](#|-)?)?'
        
        if re.search(basic_chords, chord) or \
           re.search(extended_chords, chord) or \
           re.search(slash_chords, chord):
            return root_note + self.JAZZ5_MIREX_KINDS[0]  # Consider them as major type


        if chord not in self.not_found:
            self.not_found.append(chord)
        return root_note + self.JAZZ5_MIREX_KINDS[0]  # Default to major type

    def batch_simplify_chord(self, chords:list=None):
        simple_chords = []
        for chord in chords:
            s_chord = self.simplify_chord(chord)
            if s_chord: 
                simple_chords.append(s_chord)
        return simple_chords

    def batch_chop_chord(self, chords:list=None):
        simple_chords = []
        for chord in chords:
            s_chord = self._chop_chord(chord)
            if s_chord: 
                simple_chords.append(s_chord)
        return simple_chords
