import cv2
import numpy as np

inputv = "final/static/uploads/maskd/video/"
net = cv2.dnn.readNet('final/mskdata/yolov3_mask_last.weights', 'final/mskdata/yolov3_mask.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
classes = []
with open("final/mskdata/obj.names", "r") as f:
    classes = f.read().splitlines()

def maskgen(filename,source):
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(inputv+filename)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

    def colorsx(label):
        if label == "Proper Mask":
            return (54, 250, 54)
        elif label == "Improper Mask":
            return (5, 246, 250)
        else:
            return (5, 5, 250)


    while True:
        _, img = cap.read()
        height, width, _ = img.shape
        img = cv2.flip(img,1)
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width) + 20
                    h = int(detection[3]*height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colorsx(label)
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                contor = np.array([[x,y],[x,y+20],[x+w,y+20],[x+w,y]])
                cv2.fillPoly(img, pts = [contor], color =colorsx(label))
                cv2.putText(img, f"{label} {float(confidence)*100}%", (x, y+20), font, 2, (255,255,255), 2)
        vframe = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + vframe + b'\r\n\r\n')
    cap.release()
    #     cv2.imshow('Image', img)
    #     key = cv2.waitKey(1)
    #     if key==27:
    #         break
    #cap.release()
    # cv2.destroyAllWindows()