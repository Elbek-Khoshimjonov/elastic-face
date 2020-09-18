from utils import *
import os
from os.path import isfile, isdir, join
from tqdm import tqdm
from elasticsearch import Elasticsearch as DB
import sys
import base64


es = DB([{'host': "localhost", 'port': '9200'}], timeout=30)

def exists(id):
  query = {"query":{"bool":{"must": [{"match":{"id": id}}] }}}
  res = es.search(index="face_recognition", body=query)
  return res["hits"]["total"]["value"]>0




upl = "./uploads"
for id in tqdm(os.listdir(upl), "Generating embeddings"):

    if isdir(join(upl, id)):
        
        folder = join(upl, id)
        for f in os.listdir(folder):
            
            if not exists(id):
              img = cv2.imread(join(folder, f))
              emb = run(img)
            

              vector = {"embedding": emb, "id":id, "image": str(base64.b64encode(cv2.imencode(".jpg", img)[1])) } 
              res = es.index(index="face_recognition", body=vector, doc_type="_doc")    
            # print("%s, %s, %d"%(id, text, exists(id, text)))
          

    


# # Save to elasticsearch

# for id, text, emb in zip(ids, texts, embs):

#     vector = {"embedding": emb, "id":id, "text": text} 
#     res = es.index(index="face_recognition", body=vector, doc_type="_doc")    

#     print(res)




# euc_dis = euclidean_distances(embs)
# cos_dis = cosine_distances(embs)

# # Preparing for euclidean distances
# yes_euc_dis = []
# no_euc_dis = []

# for i in tqdm(range(len(ids)), desc="Euclidean distance"):

#     yes_euc_dis.extend(euc_dis[i][np.where(ids==ids[i])])

#     no_euc_dis.extend(euc_dis[i][np.where(ids!=ids[i])])





# # Preparing for cosine distances
# yes_cos_dis = []
# no_cos_dis = []

# for i in tqdm(range(len(ids)), desc="Cosine distance"):

#     yes_cos_dis.extend(cos_dis[i][np.where(ids==ids[i])])

#     no_cos_dis.extend(cos_dis[i][np.where(ids!=ids[i])])


# # Histogram
# plt.subplot(211)
# plt.hist(yes_cos_dis, label="yes")
# plt.hist(no_cos_dis, label="no")

# plt.subplot(212)
# plt.hist(yes_euc_dis, label="yes")
# plt.hist(no_euc_dis, label="no")
# plt.show()
