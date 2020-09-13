# Face - ElasticSearch

## Database (debian installs):
1. [ElasticSearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/deb.html) 
2. [Kibana](https://www.elastic.co/guide/en/kibana/current/deb.html)

## Demo database (Kibana):
`PUT face_recognition
{
  "mappings": {
    "properties": {
      
      "embedding":{
        "type": "dense_vector",
        "dims": 2048
      },
      "id":{
        "type": "keyword"
      },
      
      "text":{
        "type": "keyword"
      }
      
    }
  }
}`


## Requirements:

1. elasticsearch
1. tensorflow
1. opencv-python
1. tqdm

## Usage

1. `python load.py` - to load images to database
1. `python search.py` - to search images