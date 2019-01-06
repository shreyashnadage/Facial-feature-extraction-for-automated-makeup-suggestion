# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
print("success")

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# =============================================================================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/HP/newanaconda/Anaconda3/envs/dlibenv/shape predictor/shape_predictor_68_face_landmarks.dat")
# 
# # load the input image, resize it, and convert it to grayscale
image = cv2.imread("C:/Users/HP/newanaconda/Anaconda3/envs/dlibenv/shreyash.png")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 
# # detect faces in the grayscale image
rects = detector(gray, 1)
print(type(rects)) 
print("faces detected {}".format(len(rects)))
#for (i, rect) in enumerate(rects):
#     # determine the facial landmarks for the face region, then
#     # convert the landmark (x, y)-coordinates to a NumPy array
shape = predictor(gray, rects[0])
shape = face_utils.shape_to_np(shape)
    #face_utils.FACIAL_LANDMARKS_IDX.items()
#     # loop over the face parts individually
    #for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
#         # clone the original image so we can draw on it, then
#         # display the name of the face part on the image
        #if name == "right_eyebrow":
        #    (x_r, y_r, w_r, h_r) = cv2.boundingRect(np.array([shape[i:j]]))
            
            
clone = image.copy()
(x, y, w, h) = cv2.boundingRect(np.concatenate([shape[35:36],shape[14:16]]))
        
        #print("part:{} i:{} j:{} shape_i:{} shape_j:{}".format(name,i,j,shape[i],shape[j-1]))
        #roi = image[y - h//3 : y + h + h//3, x - w//3 : x + w + w//3]
roi = image[y  : y + h , x  : x + w ]
roi = imutils.resize(roi, width=250,height=250, inter=cv2.INTER_CUBIC)
        #print(face_utils.FACIAL_LANDMARKS_IDXS["mouth"].values())
#         # show the particular face part
cv2.imshow("ROI", roi)
cv2.imshow("Image", clone)
cv2.waitKey(0)

