


#live streaming on Cam
#client network streaming
from imutils.video import VideoStream
import cv2
import time
import argparse
import imagezmq
import socket


# In[8]:


#parsing the argument
ap =argparse.ArgumentParser()
ap.add_argument("-s", "--server" ,required =True,  help ="IP address of the client through which sever will be connected")
args =vars(ap.parse_args())


sender =imagezmq.ImageSender(connect_to="adb connect 100.91.55.183".format(args["server"]))



rpiName = socket.gethostname()
vs =VideoStream(src =0).start()
time.sleep(2.0)

while True:
    frame =vs.read()
    frame =imutils.resize(frame , width =600)
    sender.send_image(rpiName ,frame)

vs =VideoStream(src=0, resolution =(320,280)).start()






