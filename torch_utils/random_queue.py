# https://github.com/taufikxu/GAN_PID/blob/master/gan_training/random_queue.py
import numpy as np

class Random_queue(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.length = 0
        self.data = []
        self.label = []

    def set_data(self, samples, y=None):
        if len(self.data) == 0:
            shape = samples.shape[1:]
            self.data = np.zeros([self.capacity] + list(shape))
            if y is not None:
                self.label = np.zeros([self.capacity])

        if self.length < self.capacity:
            for i in range(samples.shape[0]):
                self.data[self.length] = samples[i:i + 1]
                if y is not None:
                    self.label[self.length] = y[i:i + 1]
                self.length += 1
        else:
            permutation = np.random.permutation(self.length)
            for i in range(samples.shape[0]):
                self.data[permutation[i]] = samples[i]
                if y is not None:
                    self.label[permutation[i]] = y[i]

    def get_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size > self.length:
            if len(self.label) == 0:
                return self.data[:self.length]
            else:
                return self.data[:self.length], self.label[:self.length]

        results, results_l = [], []
        permutation = np.random.permutation(self.length)
        for i in range(batch_size):
            results.append(self.data[permutation[i]])
            if len(self.label) > 0:
                results_l.append(self.label[permutation[i]])
        if len(self.label) == 0:
            return np.stack(results, 0).astype(np.float32)
        else:
            img = np.stack(results, 0).astype(np.float32)
            lab = np.array(results_l).astype(np.int64)
            return img, lab
    def is_empty(self):
        return self.length == 0
    
import torch
class Random_queue_torch(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.length = 0
        self.data = None
        self.label = None
        self.init([3,256,256])
        self.FLAG = False
    def init(self,dim):
        self.bank = torch.randn( [self.capacity ,*dim])
        self.bank_ptr = 0
    
    def set_data(self, batch: torch.Tensor):
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        batch_size = batch.shape[0]
        ptr = self.bank_ptr

        if ptr + batch_size >= self.capacity:
            self.bank[ptr:] = batch[:self.capacity - ptr].detach()
            self.bank_ptr = 0
            self.FLAG = True
            
        else:
            self.bank[ptr:ptr + batch_size] = batch.detach()
            self.bank_ptr = ptr + batch_size
    
    
    def get_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if not self.FLAG and batch_size > self.bank_ptr:
            return self.bank[:self.bank_ptr]
        # results = []
        if self.FLAG:
            permutation = np.random.permutation(self.capacity)
        else:
            permutation = np.random.permutation(self.bank_ptr)
        
        result = self.bank[permutation[:batch_size]]
        return result
        
    def is_empty(self):
        return self.length == 0
    
if __name__ == '__main__':
    i_real_queue = Random_queue_torch(10,2)
    xtmp = torch.randn(4,3,32,32)
    i_real_queue._dequeue_and_enqueue(xtmp)
    i_xreal = i_real_queue.get_data()
    i_real_queue._dequeue_and_enqueue(xtmp)
    i_xreal = i_real_queue.get_data()
    i_real_queue._dequeue_and_enqueue(xtmp)
    i_xreal = i_real_queue.get_data()
    i_real_queue._dequeue_and_enqueue(xtmp)
    i_xreal = i_real_queue.get_data()
    print(i_xreal.shape)
