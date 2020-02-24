import cv2
import numpy as np

def visualize_prototypes(x, img):
	screen_width = 600
	screen_height = 400
	x_threshold = 2.4

	world_width = x_threshold*2
	scale = screen_width/world_width

	cartx = int(round(x[0]*scale+screen_width/2.0)) # MIDDLE OF CART
	carty = 300 # TOP OF CART

	length = 0.5 # actually half the pole's length
	polelen = scale * (2 * length)
	polex = int(round(x[0]*scale+screen_width/2.0+polelen*np.sin(x[2])))
	poley = 180
	cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	#cart velocity
	color = (255, 0, 0)  
	thickness = 4
	if x[1]<0:
	    arrow_x = cartx-screen_width//5
	else:
	    arrow_x = cartx+screen_width//5
	cv2_img = cv2.arrowedLine(cv2_img, (cartx,carty), (arrow_x,carty),color,thickness)  
	org = (arrow_x, carty+20) 
	font = cv2.FONT_HERSHEY_SIMPLEX 
	fontScale = .4
	thickness = 1
	cv2_img = cv2.putText(cv2_img, round_str(x[1]), org, font, fontScale, color, thickness, cv2.LINE_AA)

	#cart position
	color = (255,0,0)
	org = (cartx-10, carty+35) 
	cv2_img = cv2.putText(cv2_img, round_str(x[0]), org, font, fontScale, color, thickness, cv2.LINE_AA)
	cv2.imshow("f",cv2_img)

	#pole velocity
	color = (0, 0, 255)  
	thickness = 4
	if x[3]<0:
	    arrow_x = polex-screen_width//5
	else:
	    arrow_x = polex+screen_width//5
	cv2_img = cv2.arrowedLine(cv2_img, (polex,poley), (arrow_x,poley),color,thickness)  
	org = (arrow_x, poley-20) 
	font = cv2.FONT_HERSHEY_SIMPLEX 
	fontScale = .4
	thickness = 1
	cv2_img = cv2.putText(cv2_img, round_str(x[3]), org, font, fontScale, color, thickness, cv2.LINE_AA)

	#pole angle
	color = (0,0,255)
	org = (polex-10, poley-20) 
	cv2_img = cv2.putText(cv2_img, round_str(x[2])+" rad", org, font, fontScale, color, thickness, cv2.LINE_AA)
	return cv2_img

def round_str(x):
	return "{0:.2f}".format(round(x,2))