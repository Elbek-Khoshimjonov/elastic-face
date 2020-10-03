from utils import *
import os
from os.path import isfile, isdir, join
from tqdm import tqdm
from sklearn.metrics.pairwise import manhattan_distances, cosine_distances
import sys
import base64
import matplotlib.pyplot as plt


ids = []
embs = []


upl = "./uploads"
for id in tqdm(os.listdir(upl), "Generating embeddings"):

    if isdir(join(upl, id)):
        
        folder = join(upl, id)
        for f in os.listdir(folder):
            
            
            img = cv2.imread(join(folder, f))
            emb = run(img)
            ids.append(id)
            embs.append(emb)
            

    


# # Save to elasticsearch

# for id, text, emb in zip(ids, texts, embs):

#     vector = {"embedding": emb, "id":id, "text": text} 
#     res = es.index(index="face_recognition", body=vector, doc_type="_doc")    

#     print(res)

embs = np.asarray(embs)
ids = np.asarray(ids)



euc_dis = manhattan_distances(embs)
cos_dis = cosine_distances(embs)

# Preparing for euclidean distances
yes_euc_dis = []
no_euc_dis = []

for i in tqdm(range(len(ids)), desc="Manhattan distance"):

    yes_euc_dis.extend(euc_dis[i][np.where(ids==ids[i])])

    no_euc_dis.extend(euc_dis[i][np.where(ids!=ids[i])])





# Preparing for cosine distances
yes_cos_dis = []
no_cos_dis = []

for i in tqdm(range(len(ids)), desc="Cosine distance"):

    yes_cos_dis.extend(cos_dis[i][np.where(ids==ids[i])])

    no_cos_dis.extend(cos_dis[i][np.where(ids!=ids[i])])


# Histogram
plt.subplot(211)
plt.hist(yes_cos_dis, label="yes")
plt.hist(no_cos_dis, label="no")

plt.subplot(212)
plt.hist(yes_euc_dis, label="yes")
plt.hist(no_euc_dis, label="no")
plt.show()
