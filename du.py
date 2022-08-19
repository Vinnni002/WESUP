import pickle 

with open('History/history_6',  'rb') as f:
    l = pickle.load(f)

print(l)

maX = 0
for i in l:
    if i[0] > maX:
        maX = i[0]

print(maX)