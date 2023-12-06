
import json
from typing import Optional, Tuple, Union

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
                "labels" : str(jline["ideal"])
            }
        else:
            raise StopIteration("End of dataset reached.")
        
    def __len__(self):
        return self.length
    
    def generator(self):
        for line in iter(self):
            yield line


class JointEvalsDataset(IterableDataset):

    def __init__(self, dataset_path : Tuple[str], max_size : Optional[int] = None, **kwargs):
        self.dataset_paths = dataset_path
        self.datasets = [EvalsDataset(dataset_path, max_size=max_size, **kwargs) for dataset_path in self.dataset_paths]
        self.current_dataset = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.datasets[self.current_dataset])
        except StopIteration:
            self.current_dataset += 1
            if self.current_dataset < len(self.datasets):
                return next(self)
            else:
                raise StopIteration("End of dataset reached.")
            
    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])
    
    def generator(self):
        for line in iter(self):
            yield line


class ProxyDataset(IterableDataset):

    def __init__(self, dataset_path : Union[str, Tuple[str]], max_size : Optional[int] = None, **kwargs):
        if isinstance(dataset_path, str):
            self.dataset = EvalsDataset(dataset_path, max_size=max_size, **kwargs)
        else:
            self.dataset = JointEvalsDataset(dataset_path, max_size=max_size, **kwargs)

    def __iter__(self):
        return self
    
    def __next__(self):
        return next(self.dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def generator(self):
        for line in iter(self):
            yield line

        
