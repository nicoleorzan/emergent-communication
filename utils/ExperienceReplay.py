import random
import numpy as np

class ExperienceReplay:
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len(self):
        return len(self.memory)    



class RecurrentExperienceReplay:

    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.memory = []
        self.seq_length = sequence_length

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        finish = random.sample(range(0, len(self.memory)), batch_size)
        begin = [x-self.seq_length for x in finish]
        samp = []
        for start, end in zip(begin, finish):
            #correct for sampling near beginning
            final = self.memory[max(start+1,0):end+1]
            
            #correct for sampling across episodes
            for i in range(len(final)-2, -1, -1):
                if final[i][3] is None:
                    final = final[i+1:]
                    break
                    
            #pad beginning to account for corrections
            while(len(final)<self.seq_length):
                final = [(np.zeros_like(self.memory[0][0]), 0, 0, np.zeros_like(self.memory[0][3]))] + final
                            
            samp+=final

        #returns flattened version
        return samp

    def __len__(self):
        return len(self.memory)
