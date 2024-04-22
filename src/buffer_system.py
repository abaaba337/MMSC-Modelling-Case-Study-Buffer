import numpy as np
import math
from src.job import Job
from src.rejection_modes import RejectionModes


class BufferSystem:


    def __init__(self, buffer_size, buffer_queue, antennas, job_arrival_rate, job_size_generator, rejection_mode=RejectionModes.PARTIAL):
        self.buffer_size = buffer_size
        self.buffer_queue = buffer_queue
        self.antennas = sorted(antennas, key=lambda antenna: antenna.mu, reverse=True)
        self.job_arrival_rate = job_arrival_rate
        self.job_size_generator = job_size_generator
        self.rejection_mode = rejection_mode

        self.curr_time = 0.0  # curr_time is the time of the last job arrival
        self.num_rejected_jobs = 0 # number of rejected jobs (ie. a job with either fullly or partially lost data)
        self.amt_rejected_data = 0.0 # amount of data rejected (partial or full rejection)
        self.accumulated_waiting_time = 0.0

    def add_job(self, next_job, next_job_arrival):

        if self.rejection_mode == RejectionModes.FULL: ##
            if self.compute_occupancy() + next_job.size > self.buffer_size:
                self.num_rejected_jobs += 1 
                self.amt_rejected_data += next_job.size
                next_job.size = 0
                
            self.buffer_queue.add(next_job)    
            self.assign_antenna_new_jobs(next_job_arrival)
            event_time = next_job_arrival
            event_buff_occupancy = self.compute_occupancy()
            event_antenna_job_sizes = [antenna.job.size for antenna in self.antennas]
                
        elif self.rejection_mode == RejectionModes.PARTIAL:
            if self.compute_occupancy() + next_job.size > self.buffer_size:
                self.num_rejected_jobs += 1
                self.amt_rejected_data += next_job.size - (self.buffer_size - self.compute_occupancy())
                next_job.size = self.buffer_size - self.compute_occupancy()

            self.buffer_queue.add(next_job)
            self.assign_antenna_new_jobs(next_job_arrival)
            event_time = next_job_arrival
            event_buff_occupancy = self.compute_occupancy()
            event_antenna_job_sizes = [antenna.job.size for antenna in self.antennas]
            
        else:
            raise NotImplementedError
        
        return event_time, event_buff_occupancy, event_antenna_job_sizes
    
    def sample_next_job_arrival(self):
        return np.random.exponential(1/self.job_arrival_rate)
    
    def simulate_to_next_job_arrival(self):
        next_job_arrival = self.curr_time + self.sample_next_job_arrival() # time of next job arriving in buffer (T_(n+1))

        event_times = [] # times at which "events" happen (antenna(s) run out of data to transmit)
        event_buff_occupancies  = [] # TOTAL buffer occupancy at these times (antenna + queue)
        event_antenna_job_sizes = [] # size of job left for antenna to transmit at these times 

        # time at which first buffer runs out of data to transmit 
        time_to_next_event = self.find_next_event_time()  # need to keep this outside the loop to have it in the right scope

        # update event time to time of next "event"
        curr_event_time = self.curr_time + time_to_next_event

        while curr_event_time <= next_job_arrival and time_to_next_event > 0:     
            self.update_antenna_job_sizes(time_to_next_event)
            event_times.append(curr_event_time)
            event_buff_occupancies.append(self.compute_occupancy())
            event_antenna_job_sizes.append([antenna.job.size for antenna in self.antennas])
            self.assign_antenna_new_jobs(curr_event_time)
            event_times.append(curr_event_time)
            event_buff_occupancies.append(self.compute_occupancy())
            event_antenna_job_sizes.append([antenna.job.size for antenna in self.antennas])

            time_to_next_event = self.find_next_event_time()
            curr_event_time += time_to_next_event
        
        self.update_antenna_job_sizes(next_job_arrival - curr_event_time + time_to_next_event)  # needed to "unwind" the last event TODO maybe refactor this at some point

        # add the last time point right before the jump
        event_times.append(next_job_arrival)
        event_buff_occupancies.append(self.compute_occupancy())
        event_antenna_job_sizes.append([antenna.job.size for antenna in self.antennas])

        next_job = Job(self.job_size_generator.f_random_size(), next_job_arrival)
        next_job_size = next_job.size
        self.curr_time = next_job_arrival 
        job_times, job_occupancies, job_antenna_job_sizes = self.add_job(next_job, next_job_arrival)
        event_times.append(job_times)
        event_buff_occupancies.append(job_occupancies)
        event_antenna_job_sizes.append(job_antenna_job_sizes)

        return event_times, event_buff_occupancies, event_antenna_job_sizes, next_job_size  

    def find_next_event_time(self):
        return min(antenna.time_to_finish for antenna in self.antennas)
    
    def update_antenna_job_sizes(self, time_to_next_event: float):
        for antenna in self.antennas:
            antenna.job.size = max(antenna.job.size - antenna.mu * time_to_next_event, 0)
            antenna.time_to_finish = antenna.job.size / antenna.mu

    def assign_antenna_new_jobs(self, t_curr: float):
        for antenna in self.antennas:
            if math.isclose(antenna.job.size, 0):
                if not self.buffer_queue.is_empty():
                    single_waiting_time, antenna.job = self.buffer_queue.remove(t_curr)
                    antenna.time_to_finish = antenna.job.size / antenna.mu
                    self.accumulated_waiting_time += single_waiting_time

    def compute_occupancy(self):
        return self.buffer_queue.occupancy + np.sum(antenna.job.size for antenna in self.antennas)





            



    