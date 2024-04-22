class Antenna:


    def __init__(self, mu, job):
        self.mu = mu
        self.job = job
        self.time_to_finish = self.job.size / self.mu