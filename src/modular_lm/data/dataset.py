
import json
from typing import Optional

from torch.utils.data import IterableDataset
    

class EvalsDataset(IterableDataset):

    def __init__(self, dataset_path : str, max_size : Optional[int] = None, **kwargs):
        self.dataset_path = dataset_path
        self.dataset_file = open(dataset_path, "r")
        self.length = len(open(dataset_path, "r").readlines())
        self.max_size = max_size
        if max_size:
            self.length = min(self.length, max_size)
        self.current_line = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        line = self.dataset_file.readline()
        self.current_line += 1
        if line and self.current_line <= self.length:
            jline = json.loads(line)
            return {
                "text" : "\n".join([s["content"] for s in jline["input"]]),
                "labels" : jline["ideal"]
            }
        else:
            raise StopIteration("End of dataset reached.")
        
    def __len__(self):
        return self.length
    
    def generator(self):
        for line in iter(self):
            yield line