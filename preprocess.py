from music21 import *
import glob
import ipdb
import numpy as np
import spacy
import re

# Initialize a spacy instance for text processing
spacy_nlp = spacy.load('en')



def get_data_from_dir(dir, single_sents=True, max_notes=100):
    """Generates the data in the form of a list of words for the lyrics and a list
    of notes and chords for the music. Files that do not include lyrics are ignored.

    Parameters
    ----------
    dir : string
        Path to the directory that contains the midi files.
    single_sents : boolean
        Decides whether to return single sentences or return whole lyric text as
        one sentence. Set by default to True, because by splitting the lyrics to
        single sentences we get better results, faster computation time and more
        training data.
    max_notes : int
        Maximum number of length for a single sentence to consider. If the length
        exceeds this number, this training sample is ignored.


    Returns
    -------
    Tuple
        A tupe of lists where the first list contains the lists of strings of words in the
        lyrics and the second contains the lists of the string representations of the notes/chords,
        extracted from all midi files.


    """
    all_lyrics = []
    all_notes = []
    # Loop over every file
    for file in glob.glob(f"{dir}/*.mid"):
        print(f"Working on {file}")
        # Extract single sentences
        if single_sents:
            lyrics, notes = parse_single_sentences(file, max_notes)
            if not lyrics or not notes:
                print(f"No lyrics in {file}")
            else:
                all_lyrics.extend(lyrics)
                all_notes.extend(notes)

        # Extract whole lyrics as one sentence
        else:
            lyrics, notes = parse_whole_lyrics(file)
            if not lyrics or not notes:
                print(f"No lyrics in {file}")
            else:
                all_lyrics.append(lyrics)
                all_notes.append(notes)

    return all_lyrics, all_notes

def get_notes_from_stream(s):
    """Generates the list of the string representation of notes/chords from a
    music21.stream object. For notes, the pitch is recorded. As for chords, it generates
    a string of the normal order representation of each note in the chord separated by
    a dot. This form is convenient for reversing the presentation back to music notes
    with music21.

    Parameters
    ----------
    s : music21.strem
        Stream object of the midi file converted with music21

    Returns
    -------
    list
        List of the string representation of the notes

    """
    notes = []
    notes_to_parse = s.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def parse_whole_lyrics(file, all_instruments=False):
    """Helper function to read and parse all the text of a song as a single
    training sample.

    Parameters
    ----------
    file : string
        Path to the midi file.
    all_instruments : boolean
        Whether to include the tracks of all instruments in while extracting the notes/
        chords, or only focus on the notes/chords that are present in the same
        track as the lyrics. Set by default to False, because using all instruments
        resulted in very long target sequences of notes that are much longer
        than the lyrics lists

    Returns
    -------
    Tuple
        A tupe of lists where the first list contains strings of words in the
        lyrics and the second contains the string representations of the notes/chords

    """
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


def parse_single_sentences(file, max_notes):
    """Helper function to read and parse single sentence of lyrics and generate multiple
    training samples from a single midi file. This method cannot include all
    instruments by default because it is segmenting the track that contains the
    lyrics, hence other tracks are ignored.

    Parameters
    ----------
    file : string
        Path to the midi file.
    max_notes : type
        Maximum length of notes/chords allowed in the training data.

    Returns
    -------
    Tuple
        A tupe of lists where the first list contains the lists of strings of words in the
        lyrics and the second contains the lists of the string representations of the notes/chords,
        all extracted from a single file.
    """
    lyrics = []
    notes = []
    m = midi.MidiFile()
    m.open(file)
    m.read()
    for track in m.tracks:
        if "LYRIC" in [ev.type for ev in track.events]:
            # Prepare events for track creation
            start = midi.translate.getStartEvents()
            end = midi.translate.getEndEvents()

            # Initialize accumlation lists for sub-tracks
            current_events = []
            current_lyrics = []
            for ev in track.events:
                if ev.type == "LYRIC":
                    # Track with lyrics found
                    if ev.data == b"\r":
                        # Return character found = previous sentence finished
                        # Create sub-track to convert it to strings
                        try:
                            temp_track = midi.MidiTrack(1)
                            temp_track.events = start+current_events+end
                            temp_stream = midi.translate.midiTrackToStream(temp_track)
                            current_notes = get_notes_from_stream(temp_stream)
                            if len(current_notes) <= max_notes and len(current_notes) > 0:
                                lyrics.append(preprocessing_pipeline(current_lyrics))
                                notes.append(current_notes)
                            else:
                                print(f"Not adding {current_notes} and {current_lyrics}")
                            # Empty buffer for next sentence
                            current_lyrics = []
                            current_events = []
                        except Exception as e:
                            # Exception caused by weird characters in the lyrics
                            print(e)
                            print(f"Could not add {current_lyrics}")
                            current_lyrics = []
                            current_events = []
                    else:
                        # Accumilate lyrics text
                        current_lyrics.append(ev.data)
                else:
                    # Accumilate midi events for conversion
                    current_events.append(ev)

    return lyrics, notes

def produce_batch(inputs, max_sequence_length=None):
    """Produces the batches in the required format by the network. At some stage
    we implemented time-major architectures, which why this function returns the
    time-major format. If batch major is needed, then the transpose of the output
    is used.

    Parameters
    ----------
    inputs : list
        List of lists of the input data in the integer representation.
    max_sequence_length : int
        Maximum sequence length of the lyrics.

    Returns
    -------
    Tuple
        List of list of the input data in the time-major format and the list containing
        the original length of each input.

    """
    sequence_lengths = [sum(i > 2 for i in seq) for seq in inputs]
    print(sequence_lengths)
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

def preprocessing_pipeline(sentence, is_list=True):
    """The main pipeline for cleaning input text. The function lemmatizes the text,
    removes unnecessary characters and uses the lower case as normalization step.
    In addition to that the text is tokenized if it is not already in a list format.

    Parameters
    ----------
    sentence : string or list
        In case of testing, this is the string containing the sentence to be converted.
        While training, this is the list containg the words as extracted from the midi
        file.
    is_list : boolean
        Indicates whether the input is a list or is a string.

    Returns
    -------
    list
        Returns the list of clean tokens.

    """
    if is_list:
        data_string = "".join([decontracted(i.decode("utf-8").lower()) for i in sentence])
    else:
        data_string = decontracted(sentence.lower())
    data_tokens = [token.lemma_ if not (token.lemma_ == "-PRON-" or token.pos_=="PUNCT") else token.text for token in spacy_nlp(data_string)]
    return data_tokens

def decontracted(sentence):
    """Cleans the sentence from observed noisy characters. Needed to be handcrafted
    after observing the data at hand.

    Parameters
    ----------
    sentence : string
        Input lyric sentence.

    Returns
    -------
    string
        Clean output lyric sentence.

    """
    # PUNCT
    sentence = re.sub(r"\{", "", sentence)
    sentence = re.sub(r"\}", "", sentence)
    sentence = re.sub(r"\[", "", sentence)
    sentence = re.sub(r"\]", "", sentence)
    sentence = re.sub(r"\(", "", sentence)
    sentence = re.sub(r"\)", "", sentence)
    sentence = re.sub(r"_", " ", sentence)
    sentence = re.sub(r"\n", "", sentence)
    sentence = re.sub(r",", "", sentence)

    return sentence

def prepare_prediction(lyrics, batch_size, data_dict):
    """Prepares sample inputs to be fed to the network by padding dummy data to fill
    the batch dimension, tokenizing the sentences in the same way the training data
    was tokenized and looking up the words for their integer representation

    Parameters
    ----------
    lyrics : list
        List of strings of the sentences to be tested.
    batch_size : int
        Batch size used while training.
    data_dict : Seq2Seq_Dictionary
        Class defined in the notebook, contains information about the lyrics.

    Returns
    -------
    Tuple
        Returns the list of transformed testing data, and the mask to distinguish
        between actual test data and padded dummy data.

    """
    mask = 0
    while not len(lyrics) % batch_size == 0:
        lyrics.append("<PAD>")
        mask += 1
    predict_input = [data_dict.transform_sentence(preprocessing_pipeline(i, False)) for i in lyrics]
    return predict_input, mask

def notes_to_midi(notes_list, output, randomize_offset=False):
    """Converts a list of nodes in string representation to a music21.stream object, that could
    be saved to a midi file and played. By default an offset of 0.5 seconds is added
    to the notes when generating the stream.

    Parameters
    ----------
    notes_list : list
        List of the string representation of the notes/chords. They have to follow
        the convention followed for this project in order to be converted properly.
    output : string
        String containg the file name to save the midi file to. Will be added in the
        output directory.
    randomize_offset : boolean
        If set to false a constant offset of 0.5 is added between every note/chord,
        else a random offset between 0.4 and 0.6 is added.

    Returns
    -------
    None

    """

    offset = 0
    output_notes = []
    # Create note and chord objects based on the values generated by the model
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
        if not randomize_offset:
            offset += 0.5
        else:
            offset += np.random.uniform(low=0.4, high=0.6)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=f'output/{output}.mid')


def play_midi(midi_file):
    """Produces a player in the ipynb to play a midi file.

    Parameters
    ----------
    midi_file : string
        Path to midi file.

    Returns
    -------
    None

    """
    song = converter.parse(midi_file)
    song.show("midi")
