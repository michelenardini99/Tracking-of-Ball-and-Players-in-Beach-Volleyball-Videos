import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
import timeit



coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" , 
"frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
"baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
"plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
"banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
"pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
"mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
"laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
"oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
"clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]
font = cv2.FONT_HERSHEY_SIMPLEX

model = torchvision.models.detection.ssd300_vgg16(weights='SSD300_VGG16_Weights.DEFAULT')
model.eval()

ig = Image.open(r'public\frame_1237.jpg')
transform = T.ToTensor()
img = transform(ig)

with torch.no_grad():
    start = timeit.default_timer()
    pred = model([img])
    end = timeit.default_timer()
    bboxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
    igg = cv2.imread(r'public\frame_1237.jpg')
    num = torch.argwhere(scores > 0.5).shape[0]
    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
        class_name = coco_names[labels.numpy()[i]  - 1]
        if(class_name == "person"):
            igg = cv2.rectangle(igg, (x1, y1), (x2, y2), (0, 255, 0), 1)
            igg = cv2.putText(igg, class_name, (x1, y1 - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            igg = cv2.putText(igg, str(round(scores.numpy()[i], 3)), (x1, y1 - 20), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(r'public\frame_ssd_3.jpg', igg)

end = timeit.default_timer()
print("Tempo di elaborazione:", round(end-start,3), "secondi")