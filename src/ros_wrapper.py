import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Point, Quaternion, Twist

from EKF_DATMO import EKF_DATMO
import time
import cv2


def drawFrameLocal(datmo) :
	windowSize = 800
	zoomArea = 100.0
	scale = windowSize*1.1/(2.0*zoomArea)
	frame = 125.0*np.ones( (windowSize,windowSize,3))
	elements = datmo.getMOLocal()
	color = (255,0,0)
	for i in range(len(elements) ):
		el = elements[i]
		center=( int( el[0,0]*scale+windowSize/2.0), int( el[1,0]*scale+windowSize/2.0 ) )
		cv2.circle(frame, center=center, radius=5, color=color, thickness=5, lineType=8, shift=0)
		text = 'MO{}'.format( i )
		cv2.putText(frame,text=text, org=center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)	
	
	cv2.circle(frame, center=(windowSize/2,windowSize/2), radius=100, color=color, thickness=2)
	text = '{}'.format(datmo.robot.getState().transpose() )
	cv2.putText(frame,text=text, org=(0,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)	
	cv2.imshow('mapLocal',frame)
	
def drawFrameGlobal(datmo) :
	windowSize = 800
	zoomArea = 50.0
	scale = windowSize*1.1/(2.0*zoomArea)
	frame = 125.0*np.ones( (windowSize,windowSize,3))
	elements = datmo.mapMO
	color = (255,0,0)
	colorred = (0,0,255)
	for i in range(len(elements) ):
		el = elements[i].getState()
		sigma = elements[i].getCovariance()
		radius = int(scale*np.sqrt(float( sigma[0,0])/2.0 + float(sigma[1,1])/2.0 ) )
		center=( int( el[0,0]*scale+windowSize/2.0), int( el[1,0]*scale+windowSize/2.0 ) )
		cv2.circle(frame, center=center, radius=radius, color=color, thickness=radius, lineType=8, shift=0)
		cv2.circle(frame, center=center, radius=2, color=colorred, thickness=2, lineType=8, shift=0)
		text = 'MO{}'.format( i )
		cv2.putText(frame,text=text, org=center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=4)	
	
	#cv2.circle(frame, center=(windowSize/2,windowSize/2), radius=100, color=color, thickness=2)
	
	#draw robot :
	color = (0,255,0)
	robot = datmo.getRobot().getState()
	center=( int( robot[0,0]*scale+windowSize/2.0), int( robot[1,0]*scale+windowSize/2.0 ) )
	cv2.circle(frame, center=center, radius=10, color=color, thickness=10, lineType=8, shift=0)
	text = 'ROBOT'
	cv2.putText(frame,text=text, org=center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=4)	
	
	cv2.imshow('mapGlobal',frame)
	
	
	
	
class EKF_DATMO_ROS :
	def __init__(self,listen_topic,sub_topic, number=0,freq=100,dist=5.0) :
		self.listen_topic = listen_topic
		self.sub_topic = sub_topic
		self.number = number
		
		self.freq = freq
		self.assoc_dist = dist
		
		self.datmo = EKF_DATMO(freq=self.freq, assoc_dist=self.assoc_dist)
		
		rospy.init_node('EKF_DATMO_ROS_'+number, anonymous=False)
		rospy.on_shutdown(self.shutdown)

		#subscribers :
		self.sub_odom = rospy.Subscriber('robot_model_teleop_{}/cmd_vel'.format(str(self.number)), self.callbackODOM)
		self.sub_obs = rospy.Subscriber('robot_model_teleop_{}/YOLO'.format(str(self.number)), self.callbackOBS )


		#publishers :
		self.publisher = rospy.Publisher('/robot_model_teleop_{}/DATMO'.format(str(self.number)), ModelStates, 10)

	def callbackODOM(self, odom_twist) :
		obs = np.array( [ odom_twist.linear.x, odom_twist.linear.y, odom_twist.angular.z]).reshape((3,1))
		self.datmo.observationRobotVelocity( obs )

	def callbackOBS(self, modelstates) :
		measurements = []
		for name, position in zip( modelstates.name, modelstates.pose.position) :
			landmark = None
			if 'target' in name :
				landmark = 'target'
			if 'robot' in name:
				landmark = 'robot'

			if landmark is not None :
				r = np.sqrt( position.x**2 + position.y**2)
				theta = np.arctan2( position.y, position.x )
				obs = np.array( [r, theta])
				measurements.append( (landmark, obs) )

		if len(measurements) :
			self.datmo.observationMapPosition( measurements )

	
	def publish(self) :
		modelstates = ModelStates()

		robot = self.datmo.getRobot()
		mapMO = self.datmo.getMOLocal()#self.datmo.getMapMO()
		mapMOtypes = self.datmo.getMapMOTypes()
		indexMO = dict()

		for elem,i in enumerate(mapMO) :
			types = mapMOtypes[i]
			maintype = None
			nbr = 0
			for key in types.keys() :
				if types[key] >= nbr :
					maintype = key
			if maintype in indexMO.keys() :
				indexMO[maintype] += 1
			else :
				indexMO[maintype] = 0
			name = maintype+str(indexMO[maintype])

			state = elem#elem.getState()

			pose = Pose()
			twist = Twist()

			pose.position.x = state[0,0]
			pose.position.y = state[1,0]

			twist.linear.x = state[2,0]
			twist.linear.y = state[3,0]

			modelstates.name.append(name)
			modelstate.pose.append(pose)
			modelstate.twist.append(twist)

		self.publisher.publish(modelstates)





	def shutdown(self):
		rospy.loginfo("Stop")
		self.datmo.setContinuer( False)
		


if __name__ == '__main__':
	freq = 100
	dist = 5.0
	continuer = True
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--debug", action='store_true', help="print obstacles and laserScan visible cells.")
	parser.add_argument("-theta", action='store', dest='theta', type=float, default=-90.0, help="rotation from the odometry to the grid, in degrees, around z.")
	
	args = parser.parse_args()
	
	datmo_ros = EKF_DATMO_ROS(freq=freq,dist=dist)
	datmo = datmo_ros.datmo

	try:
		while continuer :
			drawFrameLocal(datmo)
			drawFrameGlobal(datmo)
		
			key = cv2.waitKey(30) & 0xFF
			if key == ord('q') :
				continuer = False
			
			if key == ord('a') :
				r = np.random.random()*40.0
				theta = np.random.random()*np.pi*2-np.pi
				obs = np.array( [ r, theta] )
				print('OBS : {}'.format(obs) )
				measurements = [ ('landmark', obs) ]
				datmo.observationMapPosition( measurements )
			
			if key == ord('y') :
				dottheta = (np.random.random()*np.pi*2-np.pi)/10.0
				obs = np.array( [ 0.0, 0.0, dottheta] ).reshape((3,1))
				print('OBS : ROBOT :: {}'.format(obs) )
				measurements = obs
				datmo.observationRobotVelocity( measurements )
			
			time.sleep(0.1)
			
		datmo.setContinuer(False)
		cv2.destroyAllWindows()
	except rospy.ROSInterruptException:
		rospy.loginfo("Exception thrown")

