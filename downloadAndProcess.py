import cv2
import os
import random
from subprocess import call
import pickle

outputResultPath = './transcode/'
basePath = './RealEstate10K/'

# set of videos that were not found
notFoundVideosPath = "./downloaded/notFound.pkl"

def loadNotFoundVideos():
    if os.path.exists(notFoundVideosPath):
        with open(notFoundVideosPath, 'rb') as f:
            return pickle.load(f)
    else:
        return set()

notFoundVideos = loadNotFoundVideos()

processedTxtFilesPath = "./processedTxtFiles.pkl"

def loadProcessedTxtFiles():
    if os.path.exists(processedTxtFilesPath):
        with open(processedTxtFilesPath, 'rb') as f:
            return pickle.load(f)
    else:
        return set()

# These are the txt files that were SUCCESSFULLY processed
# This is useful to create a "golden" record of the dataset
processedTxtFiles = loadProcessedTxtFiles()

def downloadVideo(videoPathURL, notFoundVideos):
    youtubeIDOffset = videoPathURL.find("/watch?v=") + len('/watch?v=')

    youtubeID = videoPathURL[youtubeIDOffset:]
    targetPath = "./downloaded/{}".format(youtubeID)
    
    if youtubeID in notFoundVideos:
        return targetPath, "DOWNLOAD_ERROR", notFoundVideos, youtubeID
    
    if os.path.exists(targetPath):
        print('Skipped {}, warning EXISTS'.format(targetPath))
        return targetPath, False, notFoundVideos, youtubeID

    return_code = call(["youtube-dl", "-f", "bestvideo[height<=480]", videoPathURL, "-o", targetPath, "--cookies", "./cookies.txt" ])
    error = False if return_code == 0 else "DOWNLOAD_ERROR"
    
    if "DOWNLOAD_ERROR" == error:
        notFoundVideos.add(youtubeID)
        with open(notFoundVideosPath, 'wb') as f:
            pickle.dump(notFoundVideos, f)

    return targetPath, error, notFoundVideos, youtubeID

def getBestMatchingFrames(frameTimeStamp, case, maxFrameMatchingDistanceInNS=8000):
    matches = []
    for caseIdx, c in enumerate(case):
        distance = abs(c['timeStamp'] - frameTimeStamp)
        if distance < maxFrameMatchingDistanceInNS:
            #print(c['timeStamp'], frameTimeStamp)
            #print('case index', caseIdx, 'distance',distance)
            matches.append({
                'caseIdx': caseIdx,
                'distance': distance,
            })

    matches.sort(key=lambda x: x['distance'])
    return matches

for rootPath in os.listdir(basePath):
    if 'download' in rootPath:
        continue

    subRootPath = os.path.join(basePath, rootPath)
    for subPath in os.listdir(subRootPath):
        dataFilePath = os.path.join(subRootPath, subPath)

        case = []

        with open(dataFilePath) as f:
            videoPathURL = f.readline().rstrip()
            # process all the rest of the lines 	
            for l in f.readlines():
                line = l.split(' ')

                timeStamp = int(line[0])
                intrinsics = [float(i) for i in line[1:7]]
                pose = [float(i) for i in line[7:19]]
                case.append({
                    'timeStamp': timeStamp, 
                    'intrinsics': intrinsics,
                    'pose': pose})

        downloadedVideoPath, error, notFoundVideos, youtubeID = downloadVideo(videoPathURL, notFoundVideos)
        
        if error != False:
            print('Skipped {}, error {}'.format(downloadedVideoPath, error))
            continue

        # build out the specific frames for the case
        video = cv2.VideoCapture(downloadedVideoPath) 
        video.set(cv2.CAP_PROP_POS_MSEC, 0)
        #import pdb; pdb.set_trace()

        while video.isOpened(): 
            frameOK, imgFrame = video.read() 
            if frameOK == False:
                print('video processing complete')
                break

            frameTimeStamp = (int)(round(video.get(cv2.CAP_PROP_POS_MSEC)*1000))

            matches = getBestMatchingFrames(frameTimeStamp, case, 1e9 / (2* video.get(cv2.CAP_PROP_FPS)))
            for match in matches:
                caseOffset = match['caseIdx']
                distance = match['distance']
                # match was successful, write frame
                imageOutputDir = os.path.join(outputResultPath, youtubeID)
                
                if not os.path.exists(imageOutputDir):
                    os.makedirs(imageOutputDir)
                imageOutputPath = os.path.join(imageOutputDir, '{}.jpg'.format(case[caseOffset]['timeStamp']) )
                
                if not os.path.exists(imageOutputPath):
                    print("Writing {} for frame {}, distance {}".format(imageOutputPath, case[caseOffset]['timeStamp'], distance))
                    cv2.imwrite(imageOutputPath, imgFrame)

                case[caseOffset]['imgPath'] = imageOutputPath
        
        # write the case file to disk
        processedTxtFiles.add(subPath)
        with open(processedTxtFilesPath, 'wb') as f:
            pickle.dump(processedTxtFiles, f)
        #caseFileOutputPath = os.path.join(imageOutputDir, 'case.pkl')
        #with open(caseFileOutputPath, 'wb') as f:
        #    pickle.dump(case, f)
