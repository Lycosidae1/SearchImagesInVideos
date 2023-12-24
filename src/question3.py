import pandas as pd
import numpy as np
import cv2
import os
import time
import math
import csv

def calculate_metric_for_Q3(extension:str):
    test_gt = pd.read_csv(get_repertoire_courant() + '\\data\\test\\test_gt.csv')
    result = pd.read_csv(get_repertoire_courant() + '\\results\\result' + extension.upper() + 'Q3.csv')
    test_gt_videos = test_gt["video"].tolist()
    test_gt_minutage = test_gt["minutage"].tolist()
    result_videos = result["Video"].tolist()
    result_minutage = result["Minutage"].tolist()
    result_moyenne = result["Temps ecoule par image"].tolist()

    correctVideoQty = 0
    ecart_moyen = []
    temps_moyen = []
    for i in range(len(test_gt_videos)):
        if result_videos[i] == test_gt_videos[i]:
            correctVideoQty += 1
        if not math.isnan(float(test_gt_minutage[i])) and not math.isnan(float(result_minutage[i])):
            ecart_moyen.append(abs(test_gt_minutage[i] - result_minutage[i]))
            temps_moyen.append(result_moyenne[i])

    print("Pourcentage de videos bien identifiees: ", correctVideoQty/len(test_gt_videos)*100)
    print("Ecart moyen: ", sum(ecart_moyen)/len(ecart_moyen))
    print("Temps moyen: ", sum(temps_moyen)/len(temps_moyen))

def write_results_CSV(extension, results):
    with open(get_repertoire_courant() + '\\results\\result' + extension.upper() + 'Q3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["Image", "Video", "Minutage", "Temps ecoule par image"]
        writer.writerow(field)
        for result in results:
            writer.writerow(result)

def get_repertoire_courant():
    return os.getcwd()

def path_video():
    dir_path = get_repertoire_courant()
    dir_path += "\\data\\mp4\\"
    arrayOfVideoPath = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            arrayOfVideoPath.append(dir_path + path)
    return arrayOfVideoPath

def path_image(extension):
    dir_path = get_repertoire_courant()
    dir_path += "\\data\\test\\" + extension + "\\"
    arrayOfImagePath = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            arrayOfImagePath.append(dir_path + path)
    return arrayOfImagePath

def get_colorHistogram(image, histSize=8):

    b_hist = cv2.calcHist([image], channels=[0], mask=None, histSize=[histSize], ranges=[0, 256])
    g_hist = cv2.calcHist([image], channels=[1], mask=None, histSize=[histSize], ranges=[0, 256])
    r_hist = cv2.calcHist([image], channels=[2], mask=None, histSize=[histSize], ranges=[0, 256])

    vect = np.concatenate((b_hist, g_hist, r_hist), axis=0).flatten()
    vect = vect / np.linalg.norm(vect)

    return vect

def calculate_videos_histograms_complete(histSize):
    histogramsOfVideos = {}
    for video_path in path_video():
        video_capture = cv2.VideoCapture(video_path)
        timer = 0
        FPS = int(video_capture.get(cv2.CAP_PROP_FPS))
        name = video_path.split('\\')[-1]
        histogramsOfVideos[name] = {}
        while True:
            success, image = video_capture.read()     
            if not success:
                video_capture.release()
                break
            histogramsOfVideos[name][timer / FPS] = get_colorHistogram(image, histSize)
            timer += 1
    return histogramsOfVideos

def compute(hist2, extension:str, histSize, threshold):
    results = []
    for currentImage in path_image(extension):
        lowest_distance = 1
        start_time = time.perf_counter()
        img1 = cv2.imread(currentImage)
        hist1 = get_colorHistogram(img1, histSize)

        correctVideo = "out"
        correctTime = None
        for currentVideo in hist2:
            for timer in hist2[currentVideo]:
                dist = cv2.compareHist(hist1, hist2[currentVideo][timer], cv2.HISTCMP_BHATTACHARYYA)
                if dist < lowest_distance:
                    lowest_distance = dist
                    correctVideo = currentVideo
                    correctTime = timer

        end_time = time.perf_counter()
        if lowest_distance < threshold:
            results.append([currentImage.split('\\')[-1][:4], correctVideo[:4], correctTime, end_time - start_time])
        else:
            results.append([currentImage.split('\\')[-1][:4], "out", None, None])
    return results

def main():
    histSize = 8
    video_hist = calculate_videos_histograms_complete(histSize)
    extension = "png"
    threshold = 0.09

    print("---------------------------------PNG---------------------------------")
    result_PNG = compute(video_hist, extension, histSize, threshold)
    write_results_CSV(extension, result_PNG)
    calculate_metric_for_Q3(extension)
    print("---------------------------------------------------------------------")

    extension = "jpeg"
    print("---------------------------------JPEG--------------------------------")
    result_JPEG = compute(video_hist, extension, histSize, threshold)
    write_results_CSV(extension, result_JPEG)
    calculate_metric_for_Q3(extension)
    print("---------------------------------------------------------------------")

if __name__ == '__main__':
    main()