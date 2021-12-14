import os
import glob
import json
import subprocess
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt

def midi_to_note(midi):
    notes = np.array(['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'])
    midi_range =np.array(['0','1','2','3','4','5','6','7'])
    note_name = notes[midi%12]
    note_range = midi_range[int(np.floor(midi/12 - 1))]
    note = note_name+note_range
    return note

def make_graph():
    with open('notes/test.npy', 'rb') as f1:
        a = np.load(f1)
    a = a.astype('int')
    midi = []
    print(a)
    for i in a:
        midi.append(midi_to_note(i))
    notes = np.asarray(midi)
    print(notes)
    swaras, counts=np.unique(notes, return_counts=True)

    trans = []
    trans2 = []
    for i,j in enumerate(notes):
        if (i+1)>=(np.size(notes)):
            print(i)
            transition = notes[np.size(notes)-2]+notes[np.size(notes)-1]
            transition2 = [notes[np.size(notes)-2],notes[np.size(notes)-1]]
            break
        else:
            transition = j+notes[i+1]
            transition2 = [j,notes[i+1]]
        trans.append(transition)
        trans2.append(transition2)

    trans_dic = {}
    for i in trans:
        trans_dic[i] = 0
    for i in trans:
        trans_dic[i] += 1  

    dic = {}
    for i in notes:
        dic[i] = counts[np.where(swaras==i)[0][0]]
    _, idx = np.unique(notes, return_index=True)
    swaras = notes[np.sort(idx)]

    data = {"nodes": [], "edges": []}

    for i in range(len(swaras)):
            
            if i == 0:
                data["nodes"].append({ 'data': { 'id': str(swaras[i]), 'name': str(swaras[i]), 'type':'rectangle', 'weight': int(counts[i]*10), 'color' : 'red' }})
            elif i == len(swaras)-1:
                data["nodes"].append({ 'data': { 'id': str(swaras[i]), 'name': str(swaras[i]), 'type':'rectangle', 'weight': int(counts[i]*10), 'color' : 'blue' }})
            else:
                data["nodes"].append({ 'data': { 'id': swaras[i], 'name': swaras[i], 'weight': counts[i]*10, 'color': '#a38344' }})

    for j in range(len(trans2)):
        if trans2[j][0] in swaras and trans2[j][1] in swaras:
            weight = trans2[j][0]+trans2[j][1]
            data["edges"].append({'data': { 'source': trans2[j][0], 'target': trans2[j][1], 'weight': trans_dic[weight] }})

    with open("data.json", "w") as f3:
            json.dump(data, f3)