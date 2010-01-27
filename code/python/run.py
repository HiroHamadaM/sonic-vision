#!/usr/bin/env python
"""
Gijs Molenaar
gijs@pythonic.nl
http://gijs.pythonic.nl

requires opencv 2.0 + new python api
"""

# CHANGE ME
CAMERAID=-1 # -1 for auto, -2 for video
HAARCASCADE="/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
#HAARCASCADE="/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml" # where to find haar cascade file for face detection
MOVIE="/home/gijs/Work/uva/afstuderen/data/movie/heiligenacht.mp4" # what movie to read
STORE=False # write output video?
OUTPUT="/home/gijs/testje.mp4" # where to write output video
OSC_PORT = 6666 # where to send the osc data

# Internal parameters
THRESH = 80 # starting treshhold
FPS = 25 # target FPS
HUEBINS = 30 # how many bins for hue histogram
SATBINS = 32 # how many bins for saturation histogram
XWINDOWS = 3 # how many windows on x axe
WORKING_HEIGHT = 200 # size of image to work with, 300 is okay
FACE_BORDER = 0.2 # border of face to cut of. 0.2 is 60% of face remaining


import cv
import time
import sys
import math
import osc



def hue_histogram_as_image(hist):
    """ Returns a nice representation of a hue histogram """
    histimg_hsv = cv.CreateImage( (320,200), 8, 3)

    mybins = cv.CloneMatND(hist.bins)
    cv.Log(mybins, mybins)
    (_, hi, _, _) = cv.MinMaxLoc(mybins)
    cv.ConvertScale(mybins, mybins, 255. / hi)

    w,h = cv.GetSize(histimg_hsv)
    hdims = int(cv.GetDims(mybins)[0])
    for x in range(w):
        xh = (180 * x) / (w - 1)  # hue sweeps from 0-180 across the image
        val = int(mybins[int(hdims * x / w)] * h / 255)
        cv.Rectangle( histimg_hsv, (x, 0), (x, h-val), (xh,255,64), -1)
        cv.Rectangle( histimg_hsv, (x, h-val), (x, h), (xh,255,255), -1)

    histimg = cv.CreateImage( (320,200), 8, 3)
    cv.CvtColor(histimg_hsv, histimg, cv.CV_HSV2BGR)
    return histimg


class Source:
    def __init__(self, id, flip=True):
        self.flip = flip
        if id == -2:
            self.capture = cv.CaptureFromFile(MOVIE)
        else:
            self.capture = cv.CaptureFromCAM(id)

    def print_info(self):
        for prop in [ cv.CV_CAP_PROP_POS_MSEC, cv.CV_CAP_PROP_POS_FRAMES,
                cv.CV_CAP_PROP_POS_AVI_RATIO, cv.CV_CAP_PROP_FRAME_WIDTH,
                cv.CV_CAP_PROP_FRAME_HEIGHT, cv.CV_CAP_PROP_FPS,
                cv.CV_CAP_PROP_FOURCC, cv.CV_CAP_PROP_BRIGHTNESS,
                cv.CV_CAP_PROP_CONTRAST, cv.CV_CAP_PROP_SATURATION,
                cv.CV_CAP_PROP_HUE]:
            print cv.GetCaptureProperty(self.capture, prop)

    def grab_frame(self):
        self.frame = cv.QueryFrame(self.capture)
        if not self.frame:
            print "can't grap frame, or end of movie. Bye bye."
            sys.exit(2)
        if self.flip:
            cv.Flip(self.frame, None, 1)
        return self.frame
 

class GetHands:
    def __init__(self):
        osc.init()
        self.source = Source(CAMERAID)
        self.source.print_info()
        self.threshold_value = THRESH
        self.hc = cv.Load(HAARCASCADE)
        self.ms = cv.CreateMemStorage()

        self.orig = self.source.grab_frame()

        self.width = self.orig.width
        self.height = self.orig.height
        self.size = (self.width, self.height)
        self.smallheight = WORKING_HEIGHT
        self.smallwidth = int(self.width * self.smallheight/self.height * 1.0)
        self.smallsize = (self.smallwidth, self.smallheight)

        # alloc mem for images
        self.small = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 3)
        self.visualize = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 3)
        self.bw = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.hsv = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 3)
        self.hue = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.sat = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.bp = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.scaled = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.th = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.morphed = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.temp = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.temp3 = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 3)
        self.result = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 3)
        self.hist_image = cv.CreateImage((320,200), cv.IPL_DEPTH_8U, 1)

     
        # make matrix for erode/dilate
        MORPH_SIZE = 3
        center = (MORPH_SIZE / 2) + 1
        self.morpher_small = cv.CreateStructuringElementEx(cols=MORPH_SIZE,
            rows=MORPH_SIZE, anchor_x=center, anchor_y=center,
            shape=cv.CV_SHAPE_ELLIPSE)
        MORPH_SIZE = 11
        center = (MORPH_SIZE / 2) + 1
        self.morpher = cv.CreateStructuringElementEx(cols=MORPH_SIZE,
            rows=MORPH_SIZE, anchor_x=center, anchor_y=center,
            shape=cv.CV_SHAPE_ELLIPSE)

        # alloc mem for histogram
        self.hist = cv.CreateHist([HUEBINS, SATBINS], cv.CV_HIST_ARRAY,
            [[0, 180], [0, 255]], 1)

        # video writer
        if STORE:
            self.writer = cv.CreateVideoWriter(OUTPUT,
                cv.CV_FOURCC('M','J','P','G'), 15, cv.GetSize(self.combined),
                is_color=1)

        # make window
        cv.NamedWindow('Skin Detection') 
        cv.CreateTrackbar('Threshold', 'Skin Detection', self.threshold_value,
            255, self.change_threshold)


    def change_threshold(self, position):
        self.threshold_value = position


    def find_face(self, image):
        """ detect faces in image using haar object detector """
        faces = cv.HaarDetectObjects(self.bw, self.hc, self.ms, scale_factor=1.2,
            min_neighbors=2, flags=cv.CV_HAAR_DO_CANNY_PRUNING)
        if len(faces) > 0:
            (x, y, w, h), n = faces[0]
            if n > 5:
                return (x, y, w, h)
        return False


    def face_region(self, face, border):
        (x, y, w, h) = face
        x2 = int(x+w*border)
        y2 = int(y+h*border)
        w2 = int(w-w*border*2)
        h2 = int(h-h*border*2)
        return (x2, y2, w2, h2)


    def draw_face(self, image, face, sub_face):
        (x, y, w, h) = face
        cv.Rectangle(image, (x, y), (x+w, y+h), (255, 0, 0))
        (x, y, w, h) = sub_face
        cv.Rectangle(image, (x, y), (x+w, y+h), (128, 150, 0))
 

    def update_histogram(self, region):
        cv.SetImageROI(self.hue, region)
        cv.SetImageROI(self.sat, region)
        cv.CalcArrHist([self.hue, self.sat], self.hist, 1)
        cv.NormalizeHist(self.hist, 255)
        cv.ResetImageROI(self.hue)
        cv.ResetImageROI(self.sat)


    def backproject(self):
        """ do a backprojection of face histogram on full image """
        cv.CalcArrBackProject([self.hue, self.sat], self.bp, self.hist)
        return self.bp


    def scale(self, image):
        """ scale backprojection to max of 255 """
        minVal,maxVal,minLoc,maxLoc = cv.MinMaxLoc(image)
        if maxVal > 0:
            scaler = 255/maxVal
        else:
            scaler = 1
        cv.ConvertScale(image, self.scaled, scale=scaler, shift=0.0) 
        return self.scaled


    def threshold(self, image):
        """ binary threshold image """
        cv.Threshold(image, self.th, self.threshold_value, 255, cv.CV_THRESH_BINARY)
        #cv.AdaptiveThreshold(self.scaled, self.th, 255, adaptive_method=cv.CV_ADAPTIVE_THRESH_MEAN_C, threshold_type=cv.CV_THRESH_BINARY_INV, block_size=7, param1=5)
        return self.th


    def morphology(self, image):
        """ remove noisy pixels by doing erode and dilate """
        cv.MorphologyEx(self.th, self.temp, None, self.morpher_small,
            cv.CV_MOP_OPEN, iterations=1) 
        cv.MorphologyEx(self.temp, self.morphed, None, self.morpher,
            cv.CV_MOP_CLOSE, iterations=1) 
        cv.Dilate(self.morphed, self.temp, self.morpher) 
        cv.Copy(self.temp, self.morphed)
        return self.morphed


    def find_contours(self, image):
        # hack to fix segmentation faultin cvseq when no contours are found
        if cv.CountNonZero( image ) < 100:
            return None

        return cv.FindContours(image, self.ms, mode=cv.CV_RETR_EXTERNAL,
            method=cv.CV_CHAIN_APPROX_SIMPLE, offset=(0, 0))


    def draw_contours(self, image, contours):
        if contours:
            cv.DrawContours(img=self.result, contour=contours,
                external_color=(50, 200, 50), hole_color=(50, 200, 50), max_level=1,
                thickness=1, line_type=1, offset=(0, 0))


    def find_limbs(self, contours):
        """ return the 3 biggest contours """
        blobs = []
        while contours:
            (i, center, radius) = cv.MinEnclosingCircle(contours)
            blobs.append((radius, center))
            contours = contours.h_next()

        blobs.sort()
        blobs.reverse()
        return blobs[:3]


    def draw_limbs(self, image, limbs):
        """ draw limb positions with circles into a image """        
        [left_hand, head, right_hand] = limbs

        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1) 

        if left_hand:
            cv.PutText(image, 'L', left_hand[1], font, (255, 255, 255))

        if head:
            cv.PutText(image, 'H', head[1], font, (255, 255, 255))

        if right_hand:
            cv.PutText(image, 'R', right_hand[1], font, (255, 255, 255))

        for (r, c) in [x for x in limbs if x]:
            cv.Circle(self.result, c, r, (0, 0, 255))


    def sort_limbs(self, limbs):
        """ guess the limb type, and sort them lefthand, head, righthand """
        left_hand = None
        right_hand = None
        head = None

        # sort by x position of center
        sorted_limbs = sorted(limbs, key=lambda x: (x[1][0], x[0]))
        if len(limbs) == 3:
            [left_hand, head, right_hand] = sorted_limbs

        elif len(limbs) == 2:
            [left_hand, right_hand] = limbs
        elif len(limbs) == 1:
            [head] = limbs
            left_hand = right_hand = head

        return [left_hand, head, right_hand]


    def make_sound(self, limbs):
        """ translate limb positions to osc signals """
        [left_hand, head, right_hand] = limbs
        if left_hand:
            left = 100 - int(left_hand[1][1] / self.smallheight * 100)
            osc.sendMsg("/left", [left], "127.0.0.1", OSC_PORT)
        if right_hand:
            right = 100 - int(right_hand[1][1] / self.smallheight * 100)
            osc.sendMsg("/right", [right], "127.0.0.1", OSC_PORT)
        

    def combine_images(self, images):
        """ render a list of images into one opencv frame """
        comb_width = self.smallwidth * XWINDOWS
        comb_height = self.smallheight * int(math.ceil(len(images) / float(XWINDOWS)))
        self.combined = cv.CreateImage((comb_width, comb_height), cv.IPL_DEPTH_8U, 3)

        for i,image in enumerate(images):
            if image.nChannels == 1:
                cv.Merge(image, image, image, None, self.temp3)
            else:
                cv.Copy(image, self.temp3)
            xoffset = (i % XWINDOWS) * self.smallsize[0]
            yoffset = (i / XWINDOWS) * self.smallsize[1]
            cv.SetImageROI(self.combined, (xoffset, yoffset, self.smallsize[0],
                self.smallsize[1]))
            cv.Copy(self.temp3, self.combined)
            cv.ResetImageROI(self.combined)
        return self.combined


    def pipline(self):
        presentation = []
        self.orig = self.source.grab_frame()
        cv.Resize(self.orig, self.small)
        cv.CvtColor(self.small, self.bw, cv.CV_BGR2GRAY)
        cv.CvtColor(self.small, self.hsv, cv.CV_BGR2HSV)
        cv.Split(self.hsv, self.hue, self.sat, None, None)
        cv.Copy(self.small, self.visualize)
        presentation.append(self.visualize)
        presentation.append(self.hue)
        presentation.append(self.sat)

        face = self.find_face(self.small)

        if face:
            sub_face = self.face_region(face, FACE_BORDER)
            self.update_histogram(sub_face)
            self.draw_face(self.visualize, face, sub_face)

        bp = self.backproject()

        scaled = self.scale(bp)
        presentation.append(scaled)

        th = self.threshold(scaled)
        presentation.append(th)

        morphed = self.morphology(th)

        # make dark copy of original
        cv.Copy(self.small, self.result)
        cv.ConvertScale(self.result, self.result, 0.2)

        cv.Copy(self.small, self.result, morphed)

        contours = self.find_contours(morphed)

        self.draw_contours(self.result, contours)

        limbs = self.find_limbs(contours)
        limbs = self.sort_limbs(limbs)

        self.draw_limbs(self.result, limbs)
        presentation.append(self.result)
        
        self.make_sound(limbs)
        
        # combine and show the results
        combined = self.combine_images(presentation)
        cv.ShowImage('Skin Detection', combined )

        if STORE:
            cv.WriteFrame(self.writer, self.combined)

        #hist_img = hue_histogram_as_image(self.hist)
        #cv.ShowImage( "histogram", self.hist_image )


    def run(self):
        while True:
            t = time.time()
            self.pipline()
            print int((time.time()-t)*1000)
            wait = max(40, (1000/FPS)-int((time.time()-t)*1000))
            cv.WaitKey(wait)


if __name__ == '__main__':
    g = GetHands()
    g.run()




