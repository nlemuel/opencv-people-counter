import cv2
import pandas as pd
import numpy as np

from datetime import datetime
from ultralytics import YOLO

from tracker import *


model=YOLO('yolov8s.pt')


area1=[(312,388),(289,390),(474,469),(497,462)]
area2=[(279,392),(250,397),(423,477),(454,469)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('peoplecount1.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 


count=0
tracker = Tracker()
lista_id = []
bbox_id = tracker.update(lista_id)
entering = {}
ppl_entering = set()

exiting = {}
ppl_exiting = set()

timestamps = {
    1: {'enter': datetime.now(), 'exit': datetime.now()},
    2: {'enter': datetime.now(), 'exit': datetime.now()}
}

data_entrada = {}
hora_entrada = {}
data_saida = {}
hora_saida = {}

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    lista_id.clear()
             
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        
        if 'person' in c:
           lista_id.append([x1, y1, x2, y2])
    bbox_id = tracker.update(lista_id)
    for i, bbox in enumerate(bbox_id):
        x3, y3, x4, y4, id = bbox
        results = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
        if results >= 0:
            entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        if id in entering:
            results_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
            if results_area1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                ppl_entering.add(id)
                timestamps[id] = {'enter': datetime.now()} #registra a entrada
      
        
        results_exiting = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
        if results_exiting >= 0:
            exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)


        if id in exiting:
            results_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
            if results_area2 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                ppl_exiting.add(id)
                timestamps[id] = {'exit': datetime.now()} #registra a saída

    
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('1'),(504,471),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('2'),(466,485),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    entrou = (len(ppl_entering))
    saiu = (len(ppl_exiting))
    dentro = (len(ppl_entering)) - (len(ppl_exiting))
    limite = 3

    cv2.putText(frame, str(entrou), (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, str(saiu), (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, str(dentro), (60, 200), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    if dentro > limite:
        cv2.putText(frame, str("LIMITE ATINGIDO"), (40, 260), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

ids = []
datas_entrada = []
horas_entrada = []
datas_saida = []
horas_saida = []

# Iterando sobre os timestamps para extrair a data e hora de entrada e saída
for id, times in timestamps.items():
    ids.append(id)
    if 'enter' in times:
        datas_entrada.append(times['enter'].date())
        horas_entrada.append(times['enter'].time())
    else:
        datas_entrada.append(None)
        horas_entrada.append(None)
    if 'exit' in times:
        datas_saida.append(times['exit'].date())
        horas_saida.append(times['exit'].time())
    else:
        datas_saida.append(None)
        horas_saida.append(None)

# Criando o DataFrame
df = pd.DataFrame({
    'ID': ids,
    'Data de entrada': datas_entrada,
    'Hora de entrada': horas_entrada,
    'Data de saída': datas_saida,
    'Hora de saída': horas_saida
})
# Exibindo o DataFrame
print(df)