class BufferQueue:


    def __init__(self):
        self.queue = []
        self.occupancy = 0

    def add(self, job):
        if job.size !=0 :   
            self.queue.append(job)
            self.occupancy += job.size

    def remove(self, removal_time):
        self.occupancy -= self.queue[0].size
        self.queue[0].waiting_time = removal_time - self.queue[0].arrival_time

        return self.queue[0].waiting_time, self.queue.pop(0)
    
    def is_empty(self):
        return len(self.queue) == 0
    
    
