import numpy as np
from threading import Lock
from EKF_Robot import EKF_Robot
from EKF_MapMO import EKF_MapMO
import rospy

class EKF_DATMO :
	def __init__(self, freq=100, thresh_assoc_dist=10.0) :
		self.freq = freq
		assert(self.freq != 0 )
		self.dt = 1.0/freq
		self.continuer = True
		
		self.thresh_assoc_dist = thresh_assoc_dist
		
		self.robot = EKF_Robot(freq=self.freq)
		self.mapMO = []
		self.mapMO_types = []
		
		'''
  	Accumulator for observations :
  	'''
  	#Observation descriptions :
  	# robot : [dot x dot y dot yaw ]
  	# map : list of tuple ( type, [ r theta ] ) in local framework...
  	self.acc_obs_robot_vel = []
  	self.acc_obs_map_pos = []
  	
  	self.rMutex = Lock()
  		
  def observationRobotVelocity(self, obs) :
		self.rMutex.acquire()
		self.acc_obs_robot_vel.append(obs)
		self.rMutex.release()
		
	def observationMapPosition(self, obs) :
		self.rMutex.acquire()
		self.acc_obs_map_pos.append(obs)
		self.rMutex.release()
		
	def fromLocalRTToGlobal(self, vectors ) :
		robotstate = self.robot.getState()
		origin = robotstate[0:2,:]
		origin_yaw = robotstate[3,0]
		gvecs = []
		
		for i in range( len(vectors) ) :
			r = vectors[i][0]
			theta = vectors[i][1]
			x = r*np.cos(theta+origin_yaw)
			y = r*np.sin(theta+origin_yaw)
			gvecs.append( origin+[ x, y] )
			
		return gvecs
		
	
	def computeDistMatrix(self, mapMO, obs_xy ) :
		nbrMO = len(mapMO)
		nbrObs = len(obs_xy)
		distm = np.zeros( (nbrMO, nbrObs ) )
		
		for i in range(nbrMO) :
			moxy = mapMO[i].getState()
			for j in range(nbrObs) :
				diff = moxy - obs_xy[j]
				dist = np.sqrt( diff.T*diff )
				distm[i][j] = dist
				
		return distm
		
		
	def dataAssociation( self, thresh_assoc, mapMO, observationsLocal) :
		'''
		return : dict 
		{ 'to_init': list of tuples (type, [x y] in global framework) for new MO attemps, 
			'to_associate': list of tuples (index mapMO, obs_type, obs_globalxy)
			} 
		'''
		obs_types = [ el[0] for el in observationsLocal ]
		obs_rthetas = [ el[1] for el in observationsLocal ]
		
		obs_global = self.fromLocalRTToGlobal(obs_rthetas)
		
		if len(mapMO) == 0 :
			out = []
			for i in range( len(obs_type) ):
				out.append( obs_types[i], obs_global[i] )
			return { 'to_init': out, 'to_associate':[] }
		
		#compute distance matrix :
		dist_mat = self.computeDistMatrix( mapMO, obs_global )
		index_min = np.argmin( dist_mat, axis=0)
		
		#differentiate between to_init and to_assoc :
		out_init = []
		out_assoc = []
		for i in range(index_min.shape[1] ):
			if dist_mat[ index_min[0,i], i] > thresh_assoc :
				out_init.append( (obs_types[i], obs_global[i]) )
			else :
				out_assoc.append( (index_min[0,i], obs_types[i], obs_global[i] ) )
				
		return { 'to_init':out_init, 'to_assoc':out_assoc }
		
	
	
	def loop(self) :
		rate = rospy.Rate(self.freq)
		
		while self.continuer :
			#predict/update robot :
			self.robot.state_callback()
			#measurement : robot :
			if len( self.acc_obs_robot_vel ) :
				self.rMutex.acquire()
				measurement = self.acc_obs_robot_vel.pop(0)
				self.rMutex.release()
				#correction : robot :
				self.robot.measurement_callback(measurement)
				
			#predict/update mapMO :
			for i in range( len(self.mapMO) ) :
				self.mapMO.state_callback()
			#measurement : mapMO :
			if len(self.acc_obs_map_pos) :
				self.rMutex.acquire()
				measurements = self.acc_obs_map_pos.pop(0)
				self.rMutex.release()
				
				todo = self.dataAssociation( thresh_assoc=self.thresh_assoc_dist, mapMO=self.mapMO, observationsLocal=measurements)
				
				#initialization :
				for el in todo['to_init'] :
					self.mapMO_types.append( dict() )
					self.mapMO_types[-1][ el[0] ] = 1
					self.mapMO.append( EKF_MapMO(freq=self.freq, initx=el[1] ) )
					
				#association :
				for el in todo['to_assoc'] :
					index = el[0]
					map_type = el[1]
					measurement = el[2]
					#type :
					if map_type is in self.mapMO_types[index] :
						self.mapMO_types[index][map_type] += 1
					else :
						self.mapMO_types[index][map_type] = 1
					#correction : mapMO :
					self.mapMO[index].measurement_callback(measurement=measurement)
					
			
			rate.sleep()


	def setContinuer(self, continuer=True)
		self.continuer = continuer
	
			
					
		
