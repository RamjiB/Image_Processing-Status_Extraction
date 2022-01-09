import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import keras_ocr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import os

def pick_top_priority_region(image, edged, output_dir_path):
    ## pick the right rectangle region
    mask = image.copy()
    cnts= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #sort the contour based on are and take only top 10
    cnts = sorted(cnts, key= cv2.contourArea, reverse=True)
    picked_regions = []
    picked_regions_edges = []
    individual_identity = []
    count = 0
    
    for c in cnts:
        #find the arc length for each detected contour
        peri = cv2.arcLength(c,True)
        #approximate the contours 
        apprx = cv2.approxPolyDP(c, 0.1*peri, True)
        #convert the approximate contours as rectangle
        x,y,w,h = cv2.boundingRect(apprx)
        
        roi = mask[y:y+h, x:x+w,:]
        roi_edges = edged[y:y+h, x:x+w]
            
        aspect_ratio = h/w
        area = w*h
        
        if area>1000 and aspect_ratio>1:
            picked_regions.append(roi)
            picked_regions_edges.append(roi_edges)
#             cv2.rectangle(mask, (x,y), (x+w, y+h),(255,0,0), 3)
            cv2.imwrite(os.path.join(output_dir_path, 'priority_regions_'+str(count+1)+'.png'), roi)
            count+=1
        else:
            aspect_ratio = w/h
            area = w*h
            if aspect_ratio > 1 and area > 500:
                individual_identity.append(roi[y:y+h,x:x+w,:])
    return picked_regions, picked_regions_edges, individual_identity

def pick_individual_regions(picked_regions, picked_regions_edges, output_dir_path, individual_identity):
    # going one step deeper to pick each region
    count = 1
    for region,edge in zip(picked_regions, picked_regions_edges):
        roi_cnts= cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        roi_cnts = imutils.grab_contours(roi_cnts)
        #sort the contour based on are and take only top 10
        roi_cnts = sorted(roi_cnts, key= cv2.contourArea, reverse=True)

        for c_1 in roi_cnts:
            #find the arc length for each detected contour
            peri = cv2.arcLength(c_1,True)
            #approximate the contours 
            apprx = cv2.approxPolyDP(c_1, 0.1*peri, True)
            #convert the approximate contours as rectangle
            x,y,w,h = cv2.boundingRect(apprx)

            aspect_ratio = w/h
            area = w*h
            if aspect_ratio > 1 and area > 500:
                individual_identity.append(region[y:y+h,x:x+w,:])
#                 cv2.imwrite(os.path.join(output_dir_path, 'individual_regions_'+str(count+1)+'.png'), roi)
                count+=1
    return individual_identity

def remove_white_space(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25,25), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    # Find enclosing boundingbox and crop ROI\n”,
    coords = cv2.findNonZero(close)
    x,y,w,h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]
    
def main(input_image_path, output_dir):
    check_mark = cv2.imread('gren_check_mark.png')

    # keras ocr initialization
    pipeline = keras_ocr.pipeline.Pipeline()

#     input_image_path = 'input/1.png'
    output_dir_path = output_dir+input_image_path.split('/')[-1].split('.')[0]

    individual_identity = []
    try:
        os.makedirs(output_dir_path)
    except Exception as e:
        print('Exception: ', e)

    ori_image = cv2.imread(input_image_path)
    image = ori_image[:, 2*ori_image.shape[1]//4:,:]

    #preprocessing steps
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    edged = cv2.Canny(blur, 10,10)

    picked_regions, picked_regions_edges,individual_identity = pick_top_priority_region(image, edged, output_dir_path)

    individual_identity = pick_individual_regions(picked_regions, picked_regions_edges, output_dir_path, individual_identity)

    data = {'region':[], 'is_green':[]}
    for identity in individual_identity:

        try:

            if len(identity) != 0 and identity.shape[1] !=0 :

                mark = remove_white_space(identity[:,2*identity.shape[1]//4:,:])
                text = remove_white_space(identity[:,:2*identity.shape[1]//4,:])
                mark = cv2.resize(mark, (check_mark.shape[1], check_mark.shape[0]))

                #text extraction
                prediction_groups = pipeline.recognize([text])
                predictions = sorted(prediction_groups[0], key=lambda p: p[1][0][0].min())
                if len(predictions) > 0:
                    #is_green comparison
                    score = ssim(check_mark, mark, multichannel=True)
                    is_green = 0
                    if score >= 0.7:
                        is_green = 1

                    data['region'].append(predictions[0][0])
                    data['is_green'].append(int(is_green))
        except Exception as e:
            print(e)

    data = pd.DataFrame(data)
    data.to_csv(output_dir_path + '/output.csv', index=False)
    print('output Saved in ', output_dir_path+ '/output.csv')
    
if __name__ == '__main__':
    main('input/1.png', 'output/')