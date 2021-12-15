import os
import glob
import json
import subprocess
import numpy as np

import pandas as pd
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
    path = 'notes/test.npy'
    with open(path, 'rb') as f1:
        a = np.load(f1)
    a = a.astype('int')
    pitches =[] 
    for i in a:
        pitches.append(midi_to_note(i))

    with open("temp.txt", "w") as f2:
        for ele in pitches:
            f2.write(ele+"\n")
    os.system("context -n 2 temp.txt | sed 's/ /, /g' | infot -n > temp2.txt")

    path = "temp2.txt"

    df=pd.read_csv(path, sep='\n', header=None)
    df.columns=["transitions", "weights"]
    tr_str=np.array(df["transitions"])
    tr=[]
    for j in range(len(tr_str)):
        tr.append((tr_str[j].split(", ")))

    print("weights:", type(df["weights"]))
    df["weights"][(df["weights"])<3]=3
    df["weights"][(df["weights"])>10]=10
    withcomma=[]
    for ind in range(len(tr)):
        withcomma.append((tr[ind][0].split(" ")))
    swaras, counts=np.unique(np.array(withcomma).flatten(), return_counts=True)
    counts[(counts*10)<30]=3
    counts=counts.tolist()

    data = {"nodes": [], "edges": []}

    for i in range(len(swaras)):
        if i == 0:
            data["nodes"].append({ 'data': { 'id': swaras[i], 'name': swaras[i], 'type':'rectangle', 'weight': counts[i]*10, 'color' : 'red' }})
        elif i == len(swaras)-1:
            data["nodes"].append({ 'data': { 'id': swaras[i], 'name': swaras[i], 'type':'rectangle', 'weight': counts[i]*10, 'color' : 'blue' }})
        else:
            data["nodes"].append({ 'data': { 'id': swaras[i], 'name': swaras[i], 'weight': counts[i]*10, 'color': '#a38344' }})
        
    # print(len(tr))
    for j in range(len(tr)):
        if tr[j][0] in swaras and tr[j][1] in swaras:
            data["edges"].append({'data': { 'source': tr[j][0], 'target': tr[j][1], 'weight': int(df["weights"][j]) }})
    print(len(data["edges"]))
    with open("data.json", "w") as f3:
        json.dump(data, f3)

    f1.close()
    f2.close()
    f3.close()

make_graph()
