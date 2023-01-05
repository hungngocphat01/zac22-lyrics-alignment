"""
This module defines function necessary for sentence segmentation after aligning
Example: 
    "lần đầu ta gặp nhỏ trong nắng chiều ban mai thẹn thùng..." -> [lần đầu ta gặp nhỏ, trong nắng chiều ban mai, ...]
"""


import json
import os
import pandas as pd
import numpy as np
from copy import deepcopy

SONGS_PATH = os.environ["SONGS_PATH"]
LYRICS_PATH = os.environ["LYRICS_PATH"]


def do_segmentation_transform(pred_labels, annotate_lyrics):
    """
    End-to-end segmentation

    Parameters
    ----------
    pred_labels : list 
        The list of predicted alignment generated by `predict_utils.end_to_end_align`
    annotate_lyrics : dict 
        The raw test set labels returned by `segment_utils.load_annotate_lyrics`
    
    Returns
    -------
    list
        Data in correct format for submission for each song
    """
    noise_songs = _get_noise_songs(annotate_lyrics, pred_labels)
    segmented_labels = do_segmentation(pred_labels, annotate_lyrics, noise_songs)
    submission = _transform_result(segmented_labels, annotate_lyrics)
    
    return submission

def load_annotate_lyrics(file_ids):
    """
    Load the raw label files in the test set

    Parameters
    ----------
    file_ids : list[str]
        File IDs of the examples in the test set
    
    Returns
    -------
    dict
        Mapping of file_id -> json data in that file
    """
    annotate_lyrics = dict()
    for file in file_ids:
        if file == ".ipynb_checkpoints":
            continue

        with open(os.path.join(LYRICS_PATH, file + ".json")) as f:
            data = json.load(f)
            annotate_lyrics[file] = data
    
    return annotate_lyrics

def _count_len_sent_annotate(song):
    """
    Return the number of tokens of each sentence in the song
    Example: [lần đầu ta gặp nhỏ, trong nắng, chiều, ban mai] -> [5, 2, 1, 2]
    """
    sent_toks = []
    for sent in song:
        sent_toks.append(len(sent["l"]))
    return sent_toks

def _get_noise_songs(annotate_lyrics, pred_labels):
    """
    Noise songs are songs that has different number of tokens in the predicted alignment
    and the loaded raw labels
    """
    num_tok_annotate = {}
    num_tok_pred = {}
    
    for song_id, song in annotate_lyrics.items():
        num_tok_annotate[song_id] = sum(_count_len_sent_annotate(song))
    
    for song in pred_labels:
        num_tok_pred[song["song"]] = len(song["alignment"])
        
    assert set(num_tok_annotate.keys()) == set(num_tok_pred.keys())
    
    noise_songs = []
    
    for song_id in num_tok_annotate:
        if num_tok_annotate[song_id] != num_tok_pred[song_id]:
            noise_songs.append(song_id)
            
    print("Noise songs", len(noise_songs))
    
    return set(noise_songs)

def do_segmentation(pred_labels, annotate_lyrics, noise_songs):
    """
    Segment the predicted alignment accordingly to the raw label

    Parameters
    ----------
    pred_labels : list 
        The predicted alignment generated by `predict_utils.end_to_end_align`
    annotate_lyrics : dict 
        The raw test set labels returned by `segment_utils.load_annotate_lyrics`
    noise_songs : list
        List of noisy song IDs, returned by `segment_utils._get_noise_songs`
    """
    new_labels = dict()
    
    for song in pred_labels:
        song_id = song["song"]

        if song_id in noise_songs:
            continue
        alignment = deepcopy(song["alignment"])
        
        for i in range(len(alignment)):
            if i != len(alignment) - 1:
                alignment[i]["e"] = alignment[i + 1]["s"]

        segs = []
        tok_segs = _count_len_sent_annotate(annotate_lyrics[song_id])

        for n in tok_segs:
            new_seg = alignment[:n]
            alignment = alignment[n:]    
            segs.append(new_seg)

        new_labels[song_id] = segs
    
    return new_labels

def _transform_result(segmented_labels, annotate_lyrics):
    """
    Transform the result into the correct format required by ZaloAI 
    """
    result = dict()
    for file_id, song in segmented_labels.items():
        new_song = []
        annotated_song = annotate_lyrics[file_id]
        
        try:
            for sent, anno_sent in zip(song, annotated_song):
                new_arr = []
                
                for tok, anno_tok in zip(sent, anno_sent["l"]):
                    tok["d"] = anno_tok["d"]
                    new_arr.append(tok)
                
                new_sent = {
                    "s": sent[0]["s"],
                    "e": sent[-1]["e"],
                    "l": new_arr
                }
                new_song.append(new_sent)
        except Exception:
            print(file_id)
            continue
        
        result[file_id] = new_song
    
    return result
