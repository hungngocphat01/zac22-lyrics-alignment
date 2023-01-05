"""
This module defines function necessary for an end-to-end process on generating alignment
for a given piece of audio and lyrics
"""

from dotenv import load_dotenv
import os
import json
import unicodedata
from dataclasses import dataclass

import soundfile as sf
import torch
import librosa
import numpy as np

SONGS_PATH = os.environ["SONGS_PATH"]
LYRICS_PATH = os.environ["LYRICS_PATH"]

blank_id = 109
sample_rate = 16000

def end_to_end_align(song, processor, model):
    """
    The function that does end-to-end alignment.
    Takes in a song and returns its alignment.

    Parameters
    ----------
    song : dict
        A dictionary representing a song returned by `load_single_example`.
    processor : transformers.Wav2Vec2Processor
    model : transformers.Wav2Vec2ForCTC

    Returns
    -------
    list
        Tokens in the given audio piece, aligned in microseconds
    """
    array = song["array"]
    transcript = song["text"]
    
    # Resample input song
    input_values = processor(
        array, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values
    
    # Run the ASR model, generate the probability matrix
    with torch.no_grad():
        logits = model(input_values).logits[0]
        asr_output = torch.log_softmax(logits, dim=-1)

    alignment = alignment_from_asr(array, asr_output, transcript, processor)
    return alignment

def alignment_from_asr(waveform, asr_output, transcript, processor):
    """
    The function that does the alignment process from ASR output

    Parameters
    ----------
    waveform : torch.Tensor
        The `array` element of the song
    asr_output : torch.Tensor
        The output probability matrix from the ASR model
    transcript : str
        The lyrics of this audio segment 
    processor : transformers.Wav2Vec2Processor

    Returns
    -------
    list 
        List of words aligned in microseconds
    """
    aligned = []

    # Get token numbers from text transcript
    with processor.as_target_processor():
        tokens = processor(transcript).input_ids
        
    trellis = get_trellis(asr_output, tokens)
    path = backtrack(trellis, asr_output, tokens)
    segments = merge_repeats(path, transcript)
    word_segments = merge_words(segments)
    
    for i in range(len(word_segments)):
        s, e = get_word_seconds(i, waveform, trellis, word_segments)
    
        aligned.append({
            "d": word_segments[i].label,
            "s": int(s * 1000),
            "e": int(e * 1000)
        })
    
    return aligned

def load_single_example(file_id):
    """
    Load a song from disk, convert it into audio array and normalize its lyrics string.
    For use in inference only.

    Parameters
    ----------
    file_id : str
        The ID of the training example (its filename with the extension part excluded)
    
    Returns
    -------
    dict
        {filename, array, text}
    """
    audio_path = os.path.join(SONGS_PATH, file_id + ".wav")
    audio_arr = read_audio(audio_path)
    
    lyrics_path = os.path.join(LYRICS_PATH, file_id + ".json")

    with open(lyrics_path) as f:
        lyrics_data = json.load(f)
    lyrics = " ".join([tok["d"].lower() for sent in lyrics_data for tok in sent["l"]])
    
    std_lyrics = "".join(c for c in lyrics if c.isalnum() or c == " ")
    std_lyrics = unicodedata.normalize("NFC", std_lyrics)
        
    return {
        "filename": file_id,
        "array": audio_arr,
        "text": std_lyrics
    }

def read_audio(filename):
    """
    Read, convert to mono channel and resample an audio file

    Parameters
    ----------
    filename : str
        Path to `.wav` file
    
    Returns
    -------
    torch.Tensor
        Audio array
    """
    speech, sampling_rate = sf.read(filename)
    
    # Convert stereo to mono
    if len(speech.shape) == 2:
        speech = np.mean(speech, axis=1)
        
    speech = librosa.resample(y=speech, orig_sr=sampling_rate, target_sr=16000)
    
    return torch.Tensor(speech)

###############################################
# Below code was taken from
# https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html

def get_trellis(emission, tokens, blank_id=blank_id):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=blank_id):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator=" "):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def get_word_seconds(i, waveform, trellis, word_segments):    
    waveform = waveform.unsqueeze(0)
    ratio = waveform.size(1) / (trellis.size(0) - 1)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    
    x0_sec = x0 / sample_rate
    x1_sec = x1 / sample_rate
    
    return x0_sec, x1_sec

