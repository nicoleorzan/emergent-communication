import random

class ExperienceReplay:
    
    def __init__(self, capacity=10000, seq_length = 10):
        self.capacity = capacity
        self.memory = []
        self.seq_length = seq_length
        
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len(self):
        return len(self.memory)    