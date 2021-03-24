import random

class AgentMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, element):
        self.memory.append(element)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len(self):
        return len(self.memory)    