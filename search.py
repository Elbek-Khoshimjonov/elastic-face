from utils import *
import os
from os.path import isfile, isdir, join
from tqdm import tqdm
from elasticsearch import Elasticsearch as DB
import time

es = DB([{'host': "localhost", 'port': '9200'}])

img = cv2.imread("test.jpg")



# Build query
query = {
    "size":128,
    "query":{
        "script_score":{
            "query":{
                "match_all": {}               
            },
            "script":{
                "source": "cosineSimilarity(params.embedding, 'embedding')",
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




