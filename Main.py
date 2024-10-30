import cv2
import numpy as np
import json



CAMERA_INDEX = 2

cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 600, 300)

params_list = ["hue_lower", "saturation_lower", "value_lower", "hue_upper", "saturation_upper", "value_upper"]

for param in params_list:
    cv2.createTrackbar(param.capitalize().replace("_", " "), "Trackbars", 0 if param[-5:] == "lower" else 255, 255, lambda x: None)
    
cv2.createTrackbar("Ready", "Trackbars", 0, 1, lambda x: None)


cap = cv2.VideoCapture(CAMERA_INDEX)

params = {}

Ready = 0
while Ready == 0:
    ret, frame = cap.read()
    
    if ret:
        resized_frame = cv2.resize(frame, (int(frame.shape[1]*0.3), int(frame.shape[0]*0.3)))
        
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        
        for param in params_list:
            param_name = param.capitalize().replace("_", " ")
            params[param] = cv2.getTrackbarPos(param_name, "Trackbars")

        lower_boundrie = np.array([params["hue_lower"], params["saturation_lower"], params["value_lower"]])
        upper_boundrie = np.array([params["hue_upper"], params["saturation_upper"], params["value_upper"]])
        
        mask = cv2.inRange(hsv, lower_boundrie, upper_boundrie)
        
        cv2.imshow('Mask', mask)
        
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cv2.drawContours(frame, contours, -1, [0, 255, 0], 1)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest_contour], 0, [255, 0, 0], 2)

        cv2.imshow("Frame", frame)

        Ready = cv2.getTrackbarPos("Ready", "Trackbars")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif cv2.waitKey(1) & 0xFF == ord("s"):
            with open("params.json", "w") as arq:
                arq.write(json.dumps(params))
        elif cv2.waitKey(1) & 0xFF == ord("l"):
            with open("params.json", "r") as arq:
                params = json.loads(arq.read())
                for param in params_list:
                    param_name = param.capitalize().replace("_", " ")
                    cv2.setTrackbarPos(param_name, "Trackbars", params[param])

    else:
        print("Error: Unable to capture frame")
        break

cap.release()
cv2.destroyAllWindows()

cv2.namedWindow("Trackbars2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars2", 600, 300)


cv2.createTrackbar("treshold", "Trackbars2", 1, 100, lambda x: None)
cv2.createTrackbar("contour", "Trackbars2", 200, 600, lambda x: None)

def find_largest_contour(hsv_image: np.ndarray) -> np.ndarray:
    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv_image, np.array([params["hue_lower"], params["saturation_lower"], params["value_lower"]]),
                                  np.array([params["hue_upper"], params["saturation_upper"], params["value_upper"]]))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return max(contours, key=cv2.contourArea)

def contour_is_object(contour: np.ndarray) -> bool:
    MINIMUM_CONTOUR_AREA = cv2.getTrackbarPos("contour", "Trackbars2")
    # Makes sure the contour isn't some random small spec of noise
    if cv2.contourArea(contour) < MINIMUM_CONTOUR_AREA:
        return False

    # Gets the smallest convex polygon that can fit around the contour
    contour_hull = cv2.convexHull(contour)
    # Fits an ellipse to the hull, and gets its area
    ellipse = cv2.fitEllipse(contour_hull)
    best_fit_ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
    # Returns True if the hull is almost as big as the ellipse
    return cv2.contourArea(contour_hull) / best_fit_ellipse_area > (cv2.getTrackbarPos("treshold", "Trackbars2")/100)

proportion = 0.3

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2
def main():
    # Open the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    samples = []
    samples_size = 3
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        # Converts from BGR to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        contour = find_largest_contour(cv2.resize(frame_hsv, (int(frame.shape[1]*proportion), int(frame.shape[0]*proportion))))
        if contour is not None and contour_is_object(contour):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (int(x/proportion), int(y/proportion)), (int(x/proportion)+int(w/proportion), int(y/proportion)+int(h/proportion)), (0,255,0), 2)
            
            midPoint = (int(x/proportion + w/(2*proportion)), int(y/proportion  + h/(2*proportion)))
            samples.append(midPoint)
            
            if len(samples) == samples_size+1:
                samples.pop(0)
                
            cv2.circle(frame, midPoint, 2, (255,0,0), 2)

            samples_avg = 0
            sampleX = []
            sampleY = []
            for sample in samples:
                sampleX.append(sample[0])
                sampleY.append(sample[1])
                 
            midPoint_avg = (int(sum(sampleX)/len(sampleX)), int(sum(sampleY)/len(sampleY)))
            cv2.circle(frame, midPoint_avg, 2, (0,0,255), 2)

            cv2.putText(frame, f"MidPoint: X: {midPoint[0]} Y: {midPoint[1]}", (10, frame.shape[0]-10), font, fontScale,fontColor,thickness,lineType)
            
            cv2.putText(frame, f"MidPoint Average: X: {midPoint_avg[0]} Y: {midPoint_avg[1]}", (10, frame.shape[0]-40), font, fontScale,(0,0,255),thickness,lineType)
            
            cv2.putText(frame, f"Area: {cv2.contourArea(contour)}", (10, frame.shape[0]-70), font, fontScale,(0,255,0),thickness,lineType)
        else:
            samples = []

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


main()