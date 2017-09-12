import numpy as np
from threading import Lock
from EKF_Robot import EKF_Robot
from EKF_MapMO import EKF_MapMO
import time
import threading


class EKF_DATMO :
	def __init__(self, freq=100, assoc_dist=1e0) :
		self.freq = freq
		assert( self.freq != 0 )
		self.dt = 1.0/freq
		self.continuer = True
		
		self.thresh_assoc_dist = assoc_dist
		
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
		self.loopThread = threading.Thread( target=EKF_DATMO.loop, args=(self,) )
		self.loopThread.start()
  
 	def getRobot(self) :
		return self.robot
	
	def getMapMO(self) :
		return self.mapMO
  
  
	def observationRobotVelocity(self, obs) :
		self.rMutex.acquire()
		self.acc_obs_robot_vel.append(obs)
		self.rMutex.release()

	def observationMapPosition(self, obs) :
		self.rMutex.acquire()
		self.acc_obs_map_pos.append(obs)
		self.rMutex.release()
		
	def fromLocalRTToGlobal(self, vectors ) :
		nbr = len(vectors)
		robotstate = np.copy(self.robot.getState() )
		origin = robotstate[0:2,:]
		origin_yaw = robotstate[3,0]
		gvecs = []
		
		for i in range( nbr ) :
			r = vectors[i][0]
			theta = vectors[i][1]
			x = r*np.cos(theta+origin_yaw)
			y = r*np.sin(theta+origin_yaw)
			gvec = origin
			gvec[0,0] += x
			gvec[1,0] += y
			gvecs.append( gvec )
			
		return gvecs
		
	def fromGlobalToLocal(self, vectors ) :
		robotstate = np.copy( self.robot.getState() )
		origin = robotstate[0:2,:]
		origin_yaw = float(robotstate[2,0])
		gvecs = []
		for i in range( len(vectors) ) :
			xl = float(vectors[i][0,0])
			yl = float(vectors[i][1,0])
			dxl = (xl-float(origin[0,0]))
			dyl = (yl-float(origin[1,0]))
			r = float(dxl**2 + dyl**2)
			theta = float(np.arctan2(dyl,dxl) - origin_yaw)
			x = r*np.cos(theta)
			y = r*np.sin(theta)
			gvecs.append( np.array([ x, y]).reshape((2,1)) )
		return gvecs
		
	def getMOLocal(self) :
		out = []
		for el in self.mapMO :
			x = el.getState()
			out.append( x )
		return self.fromGlobalToLocal( out )
		
	def computeDistMatrix(self, mapMO, obs_xy ) :
		nbrMO = len(mapMO)
		nbrObs = len(obs_xy)
		distm = np.zeros( (nbrMO, nbrObs ) )
		
		for i in range(nbrMO) :
			moxy = mapMO[i].getState()[0:2,:]
			for j in range(nbrObs) :
				diff = moxy - obs_xy[j]
				dist = np.sqrt( float( diff[0,0]**2+diff[1,0]**2) )
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
			for i in range( len(obs_types) ):
				out.append( (obs_types[i], obs_global[i]) )
			return { 'to_init': out, 'to_assoc':[] }
		
		#compute distance matrix :
		dist_mat = self.computeDistMatrix( mapMO, obs_global )
		index_min = np.argmin( dist_mat, axis=0)
		#discriminate between to_init and to_assoc :
		out_init = []
		out_assoc = []
		for i in range(index_min.shape[0] ):
			if dist_mat[ index_min[i], i] > thresh_assoc :
				out_init.append( (obs_types[i], obs_global[i]) )
			else :
				out_assoc.append( (index_min[i], obs_types[i], obs_global[i] ) )
				
		return { 'to_init':out_init, 'to_assoc':out_assoc }
		
	
	
	def loop(self) :
		
		while self.continuer :
			start = time.time()
			
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
				self.mapMO[i].state_callback()
			#measurement : mapMO :
			if len(self.acc_obs_map_pos) :
				self.rMutex.acquire()
				measurements = self.acc_obs_map_pos.pop(0)
				self.rMutex.release()
				
				todo = self.dataAssociation( thresh_assoc=self.thresh_assoc_dist, mapMO=self.mapMO, observationsLocal=measurements)
				
				#association :
				for el in todo['to_assoc'] :
					index = el[0]
					map_type = el[1]
					measurement = el[2]
					#type :
					if map_type in self.mapMO_types[index] :
						self.mapMO_types[index][map_type] += 1
					else :
						self.mapMO_types[index][map_type] = 1
					#correction : mapMO :
					self.mapMO[index].measurement_callback(measurement=measurement)
					
				#initialization :
				for el in todo['to_init'] :
					self.mapMO_types.append( dict() )
					self.mapMO_types[-1][ el[0] ] = 1
					#initial state with no velocity...
					initstate = np.array( [ el[1][0], el[1][1], 0.0, 0.0 ] ).reshape((4,1))
					self.mapMO.append( EKF_MapMO(freq=self.freq, initx=initstate  ) )	
					print(' OBS : init MO : {}'.format(el[1]) )
				
			end = time.time()
			elt = end-start
			sleep = self.dt-elt
			if sleep > 0 :
				time.sleep(sleep)
			#print('TIME LOOP : {} seconds.'.format(time.time()-start) )
			
	def setContinuer(self, continuer=True) :
		self.continuer = continuer
	
			
					
		
