import numpy as np
from EKF_Element import EKF_Element

class EKF_MapMO(EKF_Element) :
	def __init__(self, freq=200, initx=None) :
		EKF_Element.__init__(self,freq=freq)
		
		#Process/State noise
		vel_noise_std = 1e-12
		pos_noise_std = 1e-12
		self.Q = np.matrix([
				[pos_noise_std*pos_noise_std,0,0,0],
				[0,pos_noise_std*pos_noise_std,0,0],
				[0,0,vel_noise_std*vel_noise_std,0],
				[0,0,0,vel_noise_std*vel_noise_std]
		]) 

		#Sensor/Measurement noise
		measurement_noise_std = 1e-2
		self.R = measurement_noise_std * measurement_noise_std * np.identity(2)
		
		if initx is not None :
			self.x = initx
