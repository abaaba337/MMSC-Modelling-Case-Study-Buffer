import math
import numpy as np


class QoS:

    def __init__(
        self,
        event_times,
        event_buff_occupancies,
        event_antenna_job_sizes,
        num_rejected_jobs,
        amt_rejected_data,
        accumulated_waiting_time,
        job_sizes,
        mus,
        end_time,
    ):

        self.event_times = event_times
        self.event_buff_occupancies = event_buff_occupancies

        self.num_rejected_jobs = num_rejected_jobs
        self.amt_rejected_data = amt_rejected_data
        self.accumulated_waiting_time = accumulated_waiting_time
        self.job_sizes = job_sizes
        self.event_antenna_job_sizes = event_antenna_job_sizes
        self.mus = mus
        self.end_time = end_time

    def qos_delta(self):  # proportion of jobs that need (some form of) re-transmission
        return self.num_rejected_jobs / (len(self.job_sizes))

    def qos_epsilon(self):  # proportion of data that needs to be re-transmitted
        return self.amt_rejected_data / (sum(self.job_sizes))

    def qos_t_bar(self):  # average time for a job to start sending
        return self.accumulated_waiting_time / len(self.job_sizes)

    def qos_phi(self):
        idle = False
        start_index = 0  # To remember the start index of a sequence
        time_idle = np.zeros(len(self.mus))

        for i in range(len(self.mus)):

            for j, occupancy in enumerate(np.array(self.event_antenna_job_sizes)[:, i]):
                if math.isclose(occupancy, 0) and not idle:
                    # Found the start of a sequence of zeros
                    idle = True
                    start_index = j
                elif not math.isclose(occupancy, 0) and idle:
                    # Found the end of a sequence of zeros
                    time_idle[i] += (
                        self.event_times[j - 1] - self.event_times[start_index]
                    )
                    idle = False

            if idle == True:
                time_idle[i] += self.event_times[j] - self.event_times[start_index]
                idle = False

        return time_idle / (len(self.mus) * self.end_time)

    def qos_B_bar(self):  # compute average buffer occupancy using trapezium rule
        x = self.event_times
        y = self.event_buff_occupancies

        integral = 0.0
        n = len(x)

        for i in range(1, n):
            width = x[i] - x[i - 1]
            area = (y[i] + y[i - 1]) * width / 2
            integral += area
        return integral / (self.event_times[-1])
