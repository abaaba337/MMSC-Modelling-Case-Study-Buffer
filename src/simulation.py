import typing
import numpy as np
from src.buffer_system import BufferSystem
from src.rejection_modes import RejectionModes
from src.job_size_generator import JobSizeGenerator
from src.buffer_queue import BufferQueue
from src.antenna import Antenna
from src.job import Job
from src.qos import QoS


class Simulation:

    def __init__(
        self,
        end_time: float,
        job_arrival_rate: float,
        p_values: typing.List,
        a_values: typing.List,
        buffer_size: float,
        mus: typing.List[float],
        rejection_mode: RejectionModes,
    ):
        self.end_time = end_time
        antennas = [Antenna(mu, Job(0, 0)) for mu in mus]
        self.buffer_system = BufferSystem(
            buffer_size=buffer_size,
            buffer_queue=BufferQueue(),
            antennas=antennas,
            job_arrival_rate=job_arrival_rate,
            job_size_generator=JobSizeGenerator(p_values, a_values),
            rejection_mode=rejection_mode,
        )

    def run(self):
        self.event_times = [0]
        self.event_buff_occupancies = [0]
        self.event_antenna_job_sizes = [
            [0 for _ in range(len(self.buffer_system.antennas))]
        ]
        self.job_sizes = []  # initialise as empty

        while self.buffer_system.curr_time < self.end_time:
            (   new_event_times,
                new_event_buff_occupancies,
                new_event_antenna_job_sizes,
                new_job_size,
            ) = self.buffer_system.simulate_to_next_job_arrival()
            
            self.event_times.extend(new_event_times)
            self.event_buff_occupancies.extend(new_event_buff_occupancies)
            self.event_antenna_job_sizes.extend(new_event_antenna_job_sizes)
            self.job_sizes.append(new_job_size)

        self.num_rejected_jobs = self.buffer_system.num_rejected_jobs
        self.amt_rejected_data = self.buffer_system.amt_rejected_data
        self.accumulated_waiting_time = self.buffer_system.accumulated_waiting_time
        

    def compute_p0(self, p0_sample_range=1.0):
        # p0_sample_range : float between 0.0 to 1.0, sample the last sample_range * 100% percent of the data to compute p0
        n = len(self.event_times)
        id_ = round(n * p0_sample_range)
        record_time = np.array(self.event_times)[-id_:]
        buffer_states = np.array(self.event_buff_occupancies)[-id_:]

        delta_0 = (
            np.diff(np.isclose(buffer_states, 0) * record_time)
            * (np.isclose(buffer_states, 0) * record_time > 0)[:-1]
        )

        delta_0 = delta_0[delta_0 > 0]
        self.p0 = delta_0.sum() / (record_time[-1] - record_time[0])
    
    
    def compute_qos(self):
        qos = QoS(
            event_times=self.event_times,
            event_buff_occupancies=self.event_buff_occupancies,
            event_antenna_job_sizes=self.event_antenna_job_sizes,
            num_rejected_jobs=self.num_rejected_jobs,
            amt_rejected_data=self.amt_rejected_data,
            accumulated_waiting_time=self.accumulated_waiting_time,
            job_sizes=self.job_sizes,
            mus=[antenna.mu for antenna in self.buffer_system.antennas],
            end_time=self.end_time,
        )
        
        self.qos_delta = qos.qos_delta()
        self.qos_epsilon = qos.qos_epsilon()
        self.qos_t_bar = qos.qos_t_bar()
        # self.qos_phi = qos.qos_phi()
        
        self.qos_B_bar = qos.qos_B_bar()
        self.qos_B_bar_normalized = qos.qos_B_bar() / self.buffer_system.buffer_size
    
    
    def display_result(self):
        print(self.buffer_system.rejection_mode)
        print('1) average occupency : %.3f / %.3f bytes, occupency rate %.3f%%'%(self.qos_B_bar,self.buffer_system.buffer_size,
                                                                                     self.qos_B_bar_normalized*100))
        print('2) empty probability : %.3f'%(self.p0))
        print('3) delta_QoS  , job retransmission proportion  : %.3f%%'%(self.qos_delta*100))
        print('4) epsilon_QoS, data retransmission proportion : %.3f%%'%(self.qos_epsilon*100))
        print('5) t_QoS, average wait time : %.3f sec'%(self.qos_t_bar))
    

    def run_compute(self, p0_sample_range=1.0):
        self.run()
        self.compute_qos()
        self.compute_p0(p0_sample_range)