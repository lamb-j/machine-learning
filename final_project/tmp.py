import json 
from pprint import pprint

with open("data/matches1.json", "rb") as infile:
  data = json.load(infile)

pprint ( data["matches"][0]["participants"][0] )
