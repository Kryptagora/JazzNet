# Outputs
MIDI outputs from the LSTM can be found in the folders `piano/` and `arranged/`. In `piano`, pure chord sequences are played by a piano, whereas in `arranged/` the chords are arranged to mimic a jazzy style more. Arranging them was done in the following manner:

For each chord in the progression:
- The bass instrument plays the root note of the chord.
- The chord's notes are played by the electric piano.
- The piano instrument plays an arpeggiated version of the chord.
- A basic jazz drum pattern is introduced, with a ride cymbal and side stick sound.

Every output has an ID at the end, corresponding to the sequences that are stored in `../sequences/LSTM`. Only 50 sequences are exported to MIDI (no specific selection) to not overfill the repo. 

