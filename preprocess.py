from music21 import *
import glob
import ipdb
import numpy as np

def get_data_from_dir(dir, sents=True):
    all_lyrics = []
    all_notes = []
    for file in glob.glob(f"{dir}/*.mid"):
        print(f"Working on {file}")
        if sents:
            lyrics, notes = parse_single_sentences(file)
            if not lyrics or not notes:
                print(f"No lyrics in {file}")
            else:
                all_lyrics.extend(lyrics)
                all_notes.extend(notes)
        else:
            lyrics, notes = parse_whole_lyrics(file)
            if not lyrics or not notes:
                print(f"No lyrics in {file}")
            else:
                all_lyrics.append(lyrics)
                all_notes.append(notes)
    return all_lyrics, all_notes

def get_notes_from_stream(s):
    notes = []
    notes_to_parse = s.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def parse_whole_lyrics(file, all_instruments=False):
    m = midi.MidiFile()
    m.open(file)
    m.read()
    for track in m.tracks:
        lyrics = [ev.data for ev in track.events if ev.type=="LYRIC"]
        if not all_instruments:
            temp_stream = midi.translate.midiTrackToStream(track)
            notes = get_notes_from_stream(temp_stream)
        if len(lyrics) > 0:
            break
    if all_instruments:
        song = converter.parse(file)
        notes = []
        notes_to_parse = song.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return lyrics, notes
def clean_word(word):
    rmv = [",","."]
    for c in rmv:
        word = word.replace(c,"")
    return word
def parse_single_sentences(file):
    lyrics = []
    notes = []
    m = midi.MidiFile()
    m.open(file)
    m.read()
    for track in m.tracks:
        if "LYRIC" in [ev.type for ev in track.events]:
            start = midi.translate.getStartEvents()
            end = midi.translate.getEndEvents()
            current_events = []
            current_lyrics = []
            for ev in track.events:
                if ev.type == "LYRIC":
                    if ev.data == b"\r":
                        try:
                            temp_track = midi.MidiTrack(1)
                            temp_track.events = start+current_events+end
                            temp_stream = midi.translate.midiTrackToStream(temp_track)
                            current_notes = get_notes_from_stream(temp_stream)
                            if len(current_notes) <= 100 and len(current_notes) > 0:
                                lyrics.append("".join([clean_word(i.decode("utf-8").lower()) for i in current_lyrics]).strip().split())
                                notes.append(current_notes)
                            else:
                                print(f"Not adding {current_notes} and {current_lyrics}")
                            current_lyrics = []
                            current_notes = []
                        except Exception as e:
                            print(e)
                            print(f"Could not add {current_lyrics}")
                            current_lyrics = []
                            current_notes = []
                    else:
                        current_lyrics.append(ev.data)
                else:
                    current_events.append(ev)

    return lyrics, notes

def notes_to_midi(notes_list, output):
    offset = 0
    output_notes = []
    # create note and chord objects based on the values generated by the model
    for pattern in notes_list:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=f'output/{output}.mid')


def play_midi(midi_file):
    song = converter.parse(midi_file)
    song.show("midi")

def generate_dict(x, start_index=2):
    s = set([item for sublist in x for item in sublist])
    return {e:i+start_index for i,e in enumerate(s)}

def lookup(sent, vocab):
    return [vocab[i] for i in sent if i in vocab]


def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths
"""
from preprocess import *
x, y = get_data_from_dir("test_midi_small/")

vocab_words = generate_dict(x)
vocab_notes = generate_dict(y)

vocab_all = generate_dict(x+y)

x_transformed = [lookup(i,vocab_all) for i in x if i]
y_transformed = [lookup(i,vocab_all) for i in y if i]
"""
