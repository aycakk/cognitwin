from rdflib import Graph
from pathlib import Path

class TTLStore:
    def __init__(self):
        self.g = Graph()

    def load(self, *paths: str):
        for p in paths:
            path = Path(p).resolve()
            self.g.parse(path.as_uri(), format="turtle")