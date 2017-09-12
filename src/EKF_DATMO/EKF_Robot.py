import numpy as np
from EKF_Element import EKF_Element

class EKF_Robot(EKF_Element) :
	def __init__(self, freq=200) :
		EKF_Element.__init__(freq)
		
		self.x = np.zeros((6,1))
		self.sigma = np.identity(6)
		
		#State-transition model
		self.A = np.array([
				[1,0,0,self.dt,0,0],
				[0,1,0,0,self.dt,0],
				[0,0,1,0,0,self.dt],
				[0,0,0,1,0,0],
				[0,0,0,0,1,0],
				[0,0,0,0,0,1]
		]) 
		#Observation model
		self.H = np.array([
				[0,0,0,1,0,0],
				[0,0,0,0,1,0],
				[0,0,0,0,0,1]
		]) 

		#Process/State noise
		vel_noise_std = 0.005
		pos_noise_std = 0.005
		self.Q = np.array([
				[pos_noise_std*pos_noise_std,0,0,0,0,0],
				[0,pos_noise_std*pos_noise_std,0,0,0,0],
				[0,0,pos_noise_std*pos_noise_std,0,0,0],
				[0,0,0,vel_noise_std*vel_noise_std,0,0],
				[0,0,0,0,0,vel_noise_std*vel_noise_std,0],
				[0,0,0,0,0,0,vel_noise_std*vel_noise_std]
		]) 

		#Sensor/Measurement noise
		measurement_noise_std = 0.5
		self.R = measurement_noise_std * measurement_noise_std * np.identity(3)
		
		
