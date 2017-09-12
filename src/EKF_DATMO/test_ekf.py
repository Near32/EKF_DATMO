import numpy as np
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
	for i in range(len(elements) ):
		el = elements[i].getState()
		sigma = elements[i].getCovariance()
		radius = int(scale*np.sqrt(float( sigma[0,0])/2.0 + float(sigma[1,1])/2.0 ) )
		center=( int( el[0,0]*scale+windowSize/2.0), int( el[1,0]*scale+windowSize/2.0 ) )
		cv2.circle(frame, center=center, radius=radius, color=color, thickness=5, lineType=8, shift=0)
		text = 'MO{}'.format( i )
		cv2.putText(frame,text=text, org=center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)	
	cv2.circle(frame, center=(windowSize/2,windowSize/2), radius=100, color=color, thickness=2)
	
	cv2.imshow('mapGlobal',frame)
	
def test_ekf() :
	freq = 100
	dist = 5.0
	continuer = True
	
	datmo = EKF_DATMO(freq=freq, assoc_dist=dist)
	
	while continuer :
		drawFrameLocal(datmo)
		drawFrameGlobal(datmo)
		
		key = cv2.waitKey(30) & 0xFF
		if key == ord('q') :
			continuer = False
			
		if key == ord('a') :
			r = np.random.random()*10.0
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
	
test_ekf()
