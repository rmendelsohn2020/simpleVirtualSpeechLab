import numpy as np

class AnalysisMixin:
    def rmse(self):
        sim_dif = self.x - self.r
        sim_dif_sq = sum(sim_dif**2)
        sim_mse = sim_dif_sq / self.time_length
        sim_rmse = np.sqrt(sim_mse)
        return sim_rmse 
    
    def mse(self, system_response, target_response):
        sim_dif = system_response - target_response
        sim_dif_sq = np.sum(sim_dif**2)
        sim_mse = sim_dif_sq / self.time_length
        return float(sim_mse)
    
    def get_state_response(self):
        return self.x