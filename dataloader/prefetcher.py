import torch


class PreFetcher:
    def __init__(self, data_loader, cuda_data_key, data_key):
        self.loader = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.cuda_data_key = cuda_data_key
        self.data_key = data_key
        self._init()
        self.preload()
        return

    def _init(self):
        for k in self.cuda_data_key:
            setattr(self, k, None)
        for k in self.data_key:
            setattr(self, k, None)

    def _upload(self, next_data):
        for k, v in next_data.items():
            assert k in self.data_key or k in self.cuda_data_key
            setattr(self, k, next_data[k])
        return

    def _cuda(self):
        for k in self.cuda_data_key:
            v = getattr(self, k)
            setattr(self, k, v.cuda(non_blocking=True))
        return

    def _load(self):
        next_data = dict()
        for k in self.cuda_data_key:
            next_data[k] = getattr(self, k)
            if next_data[k] is not None:
                next_data[k].record_stream(torch.cuda.current_stream())
        for k in self.data_key:
            next_data[k] = getattr(self, k)
        return next_data

    def preload(self):
        try:
            next_data = next(self.loader)
            self._upload(next_data)
        except StopIteration:
            self._init()
            return

        with torch.cuda.stream(self.stream):
            self._cuda()
        return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_data = self._load()
        self.preload()
        return next_data
