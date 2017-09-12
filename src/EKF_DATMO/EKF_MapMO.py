import numpy as np
from EKF_Element import EKF_Element

class EKF_MapMO(EKF_Element) :
	def __init__(self, freq=200, initx=None) :
		EKF_Element.__init__(freq)
		
		#Process/State noise
		vel_noise_std = 1e-2
		pos_noise_std = 1e-3
		self.Q = np.array([
				[pos_noise_std*pos_noise_std,0,0,0],
				[0,pos_noise_std*pos_noise_std,0,0],
				[0,0,vel_noise_std*vel_noise_std,0],
				[0,0,0,vel_noise_std*vel_noise_std]
		]) 

		#Sensor/Measurement noise
		measurement_noise_std = 5e-1
		self.R = measurement_noise_std * measurement_noise_std * np.identity(3)
		
		if initx is not None :
			self.x = initx
