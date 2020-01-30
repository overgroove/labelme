import json
import os

root_path = os.getcwd() + '\\data\\labels'
os.chdir(root_path)
file_list = os.listdir(root_path)

for file in file_list:
    with open(file, 'r') as f1:
        json_data = json.load(f1)
    f1.close()
    
    image_path = json_data['imagePath']
    w = json_data['imageWidth']
    h = json_data['imageHeight']
    names = '0'
    
    with open(file[:-4] + 'txt', 'w') as f2:
        for i in json_data['shapes']:

            x1 = i['points'][0][0]
            x2 = i['points'][1][0]
            y1 = i['points'][0][1]
            y2 = i['points'][1][1]

            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            x_w = (x2 - x1) / w
            y_h = (y2 - y1) / h
            
            f2.write(f'{names} {x_center} {y_center} {x_w} {y_h}\n')

    f2.close()
