import numpy as np
import pandas as pd

data = pd.read_csv('enjoysport.csv')
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in specific_h] for _ in range(len(specific_h))]
    for i, h in enumerate(concepts):
        if target[i].lower() == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        else:
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
    general_h = [gh for gh in general_h if gh != ['?', '?', '?', '?', '?', '?']]
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("Final Specific Hypothesis:\n", s_final)
print("Final General Hypotheses:\n", g_final)
