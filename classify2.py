########################################################################
#
# File:   classify2.py
# Author: Julie Harris, Rachel Diamond
# Date:   May, 2016
#
# This program is heavily based on Matt Zucker's code
# in the programs faceparts.py and demo.py (for eigenfaces)
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
"""
Questions/Notes:
 - why does re-running result in worse detection rate?
 - deal with not finding a face
 - number of eigenvectors
"""
########################################################################

import cv2
import numpy
import sys
import glob
#import sklearn
from sklearn import svm

######################################################################
if len(sys.argv) != 3:
    print 'usage: python', sys.argv[0], 'train_images', 'test_images'
    print ' e.g.: python', sys.argv[0], 'images/all/train/', 0, '(for camera)'
    print '   or: python', sys.argv[0], 'images/all/train/', 'images/all/test/'
    print
    sys.exit(0)

# For test images you can specify a video device (e.g. 0) on the command line,
# or a folder of static image files

cap = None
test_img_folder = None
train_img_folder = sys.argv[1]

try:
    device_num = int(sys.argv[2])
    cap = cv2.VideoCapture(device_num)
except:
    test_img_folder = sys.argv[2]

if cap is None and test_img_folder is None:
    print 'Failed to load test images'
    sys.exit(0)

######################################################################

zoom = 1 #DEBUG was 4
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def upscale(img):
    return cv2.resize(img, (int(img.shape[1]*zoom), int(img.shape[0]*zoom)),
                      interpolation=cv2.INTER_NEAREST)

def resize_square(img):
    return cv2.resize(img, (256,256),interpolation=cv2.INTER_NEAREST)

def label_emotion(image, text):
    if text == 0:
        label_image(image, 'angry')
    elif text == 1:
        label_image(image, 'suprised')
    elif text == 2:
        label_image(image, 'sad')
    else:
        label_image(image, 'unspecified')

def label_image(image, text):

    h = image.shape[0]

    cv2.putText(image, text, (8, h-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,0), 3, cv2.LINE_AA)

    cv2.putText(image, text, (8, h-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 1, cv2.LINE_AA)

#CODE TO CROP TO FACES
########################################################################
def find_face(img,showBox):
    # img should be grayscale
    # returns a square subimage that is just the detected face
    # if no face is found, returns None

    face_rect = None

    #only want to load this once, so make it global variable instead of defining here
    #face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


    # Downsample for faster runtime -- this is important!
    img_small = downsample(img, 320)

    # Figure out what scale factor we downsampled by - we will need it
    # to rescale the rectangles returned by the cascade.
    scl = img.shape[0] / img_small.shape[0]

    # detector returns an n-by-4 array where each row is [x,y,w,h]
    # (note that w = h for the haarcascade_frontalface_alt.xml classifier)
    face_rects = numpy.array(face_detector.detectMultiScale(img_small))

    # if we found anything
    if len(face_rects):
        if face_rect is None:
            # if first detection, just choose largest area
            areas = face_rects[:,2] * face_rects[:,3]
            face_rect = face_rects[areas.argmax()]
        else:
            # otherwise, choose one closest to prev detection
            face_center = rect_center(face_rect)
            centers = face_rects[:,:2] + 0.5*face_rects[:,2:]
            diffs = centers - face_center
            face_rect = face_rects[(diffs**2).sum(axis=1).argmin()]

    # if we have a face
    if face_rect is not None:

        # upscale it to get ROI in big image
        x,y,w,h = rect_scale(face_rect, scl)

        if showBox:
            # draw rectangle in big image
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)

        # get subimage in ROI
        ret = img[y:y+h, x:x+h]
    else:
        ret =  None

    if showBox:
        cv2.namedWindow('win')
        cv2.imshow('win', img)
        k = cv2.waitKey(5)
        if k == 27:
            sys.exit(0)

    return ret

# Downsample an image to have no more than the specified maximum height
def downsample(src, hmax):
    h, w = src.shape[:2]
    while h > hmax:
        h /= 2
        w /= 2
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

def rect_center(rect):
    x,y,w,h = rect
    return numpy.array( (x+0.5*w, y+0.5*h) )

def rect_scale(rect,scl):
    return numpy.array(rect)*scl
########################################################################

class EigenFacesDemo:

    def __init__(self):
        self.image_shape = None

        train_AN = self.load_all(train_img_folder+'*ANS.JPG')
        train_SU = self.load_all(train_img_folder+'*SUS.JPG')
        train_SA = self.load_all(train_img_folder+'*SAS.JPG')

        self.train_datasets = train_AN + train_SU + train_SA
        self.num_train = len(self.train_datasets)

        self.train_labels = numpy.zeros(self.num_train).reshape(self.num_train,1)
        # 0 means angry, 1 means surprised, 2 means sad
        self.train_labels[len(train_AN):len(train_AN)+len(train_SU)] = 1
        self.train_labels[len(train_AN)+len(train_SU):] = 2
        self.train_strlabels = len(train_AN)*'N'+len(train_SU)*'U'+len(train_SA)*'A'

        print 'loaded {} train_images'.format(self.num_train)

        #compute means for each emotion
        self.mean_imgs = [sum(train_AN)/float(len(train_AN)),
                          sum(train_SU)/float(len(train_SU)),
                          sum(train_SA)/float(len(train_SA))]
        for i in range(len(self.mean_imgs)):
            label_emotion(self.mean_imgs[i],i)

        #init variables to be filled by demo_mean()
        self.mean = [0,0,0]
        self.evecs = [0,0,0]
        self.train_proj = [0,0,0]

        self.image_w = self.image_shape[1]
        self.image_h = self.image_shape[0]
        self.image_pixels = self.image_w*self.image_h
        self.row_shape = (1, self.image_pixels)

    def load_test(self):
        # AN or N = angry, SU or U = suprised, SA or A = sad
        test_AN = self.load_all(test_img_folder+'*ANS.JPG')
        test_SU = self.load_all(test_img_folder+'*SUS.JPG')
        test_SA = self.load_all(test_img_folder+'*SAS.JPG')

        #self.test_datasets = [test_AN,test_SU,test_SA]
        self.test_datasets = test_AN + test_SU + test_SA
        self.num_test = len(self.test_datasets)
        self.test_labels = numpy.zeros(self.num_test)
        # 0 means angry, 1 means surprised, 2 means sad
        self.test_labels[len(test_AN):len(test_AN)+len(test_SU)] = 1
        self.test_labels[len(test_AN)+len(test_SU):] = 2
        self.test_strlabels = len(test_AN)*'N'+len(test_SU)*'U'+len(test_SA)*'A'

        #self.test_proj = self.project_all(self.test_images)
        print 'loaded {} test_images'.format(self.num_test)

    def spacer(self):
        return numpy.zeros( (self.image_h, 8), dtype='float32' )

    def should_quit(self):
        while 1:
            k = cv2.waitKey(5)
            if k == 27:
                return True
            elif k > 0:
                return False

    def make_window(self, win, x=20, y=20):
        cv2.namedWindow(win)
        cv2.moveWindow(win, x, y)

    def demo_menu(self):
        self.load_test()
        win = 'Menu'
        img = 255*numpy.ones((150, 250, 3), dtype='uint8')
        strings = [
            'Keys:',
            '',
            's - run & SHOW mean+evecs',
            'h - run & HIDE mean+evecs',
            'l - run w/ linearSVC'
        ]
        for i in range(len(strings)):
            cv2.putText(img, strings[i],
                        (10, 20 + 20*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),
                        1, cv2.LINE_AA)

        self.make_window(win)
        cv2.imshow(win, img)
        while 1:
            k = cv2.waitKey(5)
            func = None
            if k == ord('s'):
                self.show = True
                self.demo_mean()
                func = self.demo_classify
            elif k == ord('h'):
                self.show = False
                self.demo_mean()
                func = self.demo_classify
            elif k == ord('l'):
                self.show = False
                self.demo_mean()
                func = self.demo_linearSVC()
            elif k == 27:
                break
            if func is not None:
                cv2.destroyWindow(win)
                func()
                #self.display_matches()
                self.make_window(win)
                cv2.imshow(win, img)

    def demo_mean(self):
        self.train_datasets = numpy.array( [ x.flatten() for x in self.train_datasets ] )
        self.mean, self.evecs = cv2.PCACompute(self.train_datasets, mean=None, maxComponents=20)
        self.num_vecs = self.evecs.shape[0]
        print 'got {} eigenvectors'.format(self.num_vecs)

        if self.show:
            win = 'Mean and vectors'
            self.make_window(win)

            print 'showing mean'
            img = self.mean.reshape(self.image_shape)
            cv2.imshow(win, upscale(img))
            if self.should_quit(): return

            for i in range(self.num_vecs):
                print 'showing evec', i+1, 'of', self.num_vecs
                img = self.visualize_evec(self.evecs[i])
                cv2.imshow(win, upscale(img))
                if self.should_quit(): break
            cv2.destroyWindow(win)

        #project train and test datasets through the eigenvectors
        self.train_proj = self.project_all(self.train_datasets)
        self.test_proj = self.project_all(self.test_datasets)
        print 'shape of test_proj', self.test_proj.shape


    def demo_camera1(self):
        self.train_datasets = numpy.array( [ x.flatten() for x in self.train_datasets ] )
        self.mean, self.evecs = cv2.PCACompute(self.train_datasets, mean=None, maxComponents=20)
        self.num_vecs = self.evecs.shape[0]
        print 'got {} eigenvectors'.format(self.num_vecs)

        #project train and test datasets through the eigenvectors
        self.train_proj = self.project_all(self.train_datasets)


    def demo_camera2(self, img):
        img = img.astype('float32')/255.0
        if img.shape != self.image_shape:
            print 'bad size: ', imgfile
            sys.exit(1)

        self.test_proj = self.project(img).flatten()

    def demo_linearSVC(self):
        classifier = svm.LinearSVC()
        classifier.fit(self.train_proj,numpy.ravel(self.train_labels))
        matches = classifier.predict(self.test_proj)

        num_correct = 0
        wrong = []
        matchesstr = ""
        for i in range(matches.shape[0]):
            if matches[i] == 0:
                matchesstr += 'N'
            elif matches[i] == 1:
                matchesstr += 'U'
            elif matches[i] == 2:
                matchesstr += 'A'

            if matches[i] == self.test_labels[i]:
                num_correct += 1
            else:
                wrong.append(self.test_strlabels[i]+matchesstr[i])
        print '\ngot', num_correct, 'correct out of', matches.shape[0]
        print '% correct =', num_correct/float(matches.shape[0])
        print 'wrong were (L-actual, R-linearSVC guess):', wrong

        #display matches
        win = 'Left: test img, Right: mean img for linearSVC guess'
        self.make_window(win)
        i = 0
        while 1:
            tempTest = self.test_datasets[i]
            label_emotion(tempTest, self.test_labels[i])

            imgs = [tempTest]
            imgs.append( self.spacer() )
            imgs.append(self.mean_imgs[int(matches[i])])

            big = numpy.hstack(imgs)
            cv2.imshow(win, upscale(big))
            if self.should_quit(): break
            i = (i + 1) % self.num_test
        cv2.destroyWindow(win)



    def demo_classify(self):
        FLANN_INDEX_KDTREE = 1
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
        matcher = cv2.flann.Index(self.train_proj,flann_params)
        # flann - put vector or set of vectors into matcher
        # input to index: matrix w/ every row as potential vector to match
        # train_proj = m x n matrix where m is examples, n is dimensions
        # test_proj = p x n matrix where p is number of test vectors &
        # n is number of dimensions
        # can test a bunch at a time or one at a time
        # this code tests a bunch at a time
        # pass in k (single nearest neighbor)
        # gives back matches matrix - should be a matrix that is p x k
        # where p = number of vectors passed in to matcher & k is same k
        # matches stores integer index of vector i that we passed in
        # matches gives index of closest training example to vec 0, index
        # of second closest to 0, 3rd closest to 0
        # indexes in original training data
        # train_proj = PCA weights of every single training image

        k = 1 #how many closest matches to return
        matches, dist = matcher.knnSearch(self.test_proj, k, params={})

        print 'matches.shape', matches.shape
        num_correct = 0
        wrong = []
        for i in range(matches.shape[0]):
            """
            print 'matches[',i,'0]', matches[i,0]
            print '   actual   ', self.test_labels[i]
            print '   NN guess ', self.train_labels[matches[i,0]]
            """
            if self.train_labels[matches[i,0]] == self.test_labels[i]:
                num_correct += 1
            else:
                wrong.append(self.test_strlabels[i]+self.train_strlabels[matches[i,0]])
        print '\ngot', num_correct, 'correct out of', matches.shape[0]
        print '% correct =', num_correct/float(matches.shape[0])
        print 'wrong were (L-actual, R-NN guess):\n', wrong

        win = 'Left: test img, Right: NN guess'
        self.make_window(win)
        i = 0
        while 1:
            tempTest = self.test_datasets[i]
            label_emotion(tempTest, self.test_labels[i])

            imgs = [tempTest]
            imgs.append( self.spacer() )
            for j in range(k):
                tempTrain = self.backproject(self.train_proj[matches[i,j]])
                label_emotion(tempTrain, self.train_labels[matches[i,j]])
                imgs.append(tempTrain)
            big = numpy.hstack(imgs)
            cv2.imshow(win, upscale(big))
            if self.should_quit(): break
            i = (i + 1) % self.num_test
        cv2.destroyWindow(win)

    def load_all(self, pattern):
        rval = []
        for imgfile in glob.glob(pattern):
            img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

            # extract just the find_face
            img = resize_square(find_face(img,0))

            img = img.astype('float32')/255.0
            if self.image_shape is None:
                self.image_shape = img.shape
                print 'image shape:', img.shape
            elif img.shape != self.image_shape:
                print 'bad size: ', imgfile
                sys.exit(1)
            rval.append(img)
        return rval

    def project(self, img):
        return cv2.PCAProject(img.reshape(self.row_shape), self.mean, self.evecs)

    def backproject(self, w):
        return cv2.PCABackProject(w.reshape((1, self.num_vecs)), self.mean, self.evecs).reshape(self.image_shape)

    def project_all(self, images):
        return numpy.array([ self.project(x).flatten() for x in images ])

    def visualize_evec(self, vec):
        return ( 0.5 + (0.5 / numpy.linalg.norm(vec, numpy.inf)) *
                 vec.reshape(self.image_shape) )


######################################################################

if __name__ == '__main__':
    demo = EigenFacesDemo()

    if cap is None: #test images from folder
        demo.demo_menu()

    else: #test images from camera
        demo.demo_camera1()

        # Create our window
        #win = cv2.namedWindow('face')
        classifier = svm.LinearSVC()
        classifier.fit(demo.train_proj,numpy.ravel(demo.train_labels))

        while True:
            ok, img = cap.read()
            if not ok or img is None:
                print 'Error getting image from camera'
                break

            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # extract just the face
            face = find_face(img, 1)
            if face is None:
                continue

            face = resize_square(face)
            demo.demo_camera2(face)
            label = classifier.predict(demo.test_proj)
            label_emotion(face, label[0])

            cv2.imshow('emotion', face)
            if demo.should_quit(): break
