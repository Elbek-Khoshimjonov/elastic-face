from utils import *
import os
from os.path import isfile, isdir, join
from tqdm import tqdm
from elasticsearch import Elasticsearch as DB
import time
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

def l2_dis(emb1, emb2):
  return euclidean_distances([emb1, emb2])[0][1]

def l1_dis(emb1, emb2):
  return manhattan_distances([emb1, emb2])[0][1]


es = DB([{'host': "localhost", 'port': '9200'}])

# img = cv2.imread("uploads/86296/4f7c72db5c474911bf299a1629503be0.png.thumb.jpeg")
img = cv2.imread("sample_full.jpg")

emb = run(img)

k = 10

# Build query
query = {
    "size": k,
    "_source": {"excludes": ["image"]},
    "query": {
      "knn": {
      "embedding": {
          "vector": emb,
          "k": k
        }
      }
    }
    
}



start_time = time.time()

res = es.search(index="face_recognition", body=query)

print("--- %s seconds ---" % (time.time() - start_time))

# print(res)

for row in res["hits"]["hits"]:

    print("Possible Match: id : %s, cosine: %f, l2: %f, l1: %f" % (row["_source"]["id"], row["_score"], l2_dis(emb, row["_source"]["embedding"]), l1_dis(emb, row["_source"]["embedding"]) ) )




