from utils import *
import os
from os.path import isfile, isdir, join
from tqdm import tqdm
from elasticsearch import Elasticsearch as DB
import time

es = DB([{'host': "localhost", 'port': '9200'}])

img = cv2.imread("uploads/86296/4f7c72db5c474911bf299a1629503be0.png.norm.png")
# img = cv2.imread("sample_full.jpg")

emb = run(img)

# Build query
query = {
    "size":10,
    "query":{
        "script_score":{
            "query":{
                "match_all": {}               
            },
            "script":{
                "source": "1 - 0.5 * l1norm(params.embedding, 'embedding') ",
                "params":{
                    "embedding": emb,                
                }
            }
        }  
    } 
}

start_time = time.time()

res = es.search(index="face_recognition", body=query)

print("--- %s seconds ---" % (time.time() - start_time))


for row in res["hits"]["hits"]:

    print("Possible Match: id : %s, text: %s, score: %f" % (row["_source"]["id"], row["_source"]["text"], row["_score"]) )    




