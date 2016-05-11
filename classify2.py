########################################################################
#
# File:   classify.py
# Author: Julie Harris, Rachel Diamond
# Date:   May, 2016
#
# This program is heavily based on Matt Zucker's code from April 2012
# in the programs textons.py and demo.py (for eigenfaces)
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
"""
Questions/Notes:
 - why does re-running result in worse detection rate?
 - deal with not finding a face
"""
########################################################################

import cv2
import numpy
import sys
import glob
#import sklearn

######################################################################
if len(sys.argv) != 2:
    print 'usage: python', sys.argv[0], 'filepath_to_train_and_test_folders'
    print ' e.g.: python', sys.argv[0], 'images/faces/'
    print '   or: python', sys.argv[0], 'images/small/'
    print
    sys.exit(0)

img_folder = sys.argv[1]
######################################################################

zoom = 1 #DEBUG was 4
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def upscale(img):
    return cv2.resize(img, (int(img.shape[1]*zoom), int(img.shape[0]*zoom)),
                      interpolation=cv2.INTER_NEAREST)

def resize_square(img):
    return cv2.resize(img, (250,250),interpolation=cv2.INTER_NEAREST)

def label_emotion(image, text):
    if text == 'N':
        label_image(image, 'angry')
    elif text == 'U':
        label_image(image, 'suprised')
    elif text == 'A':
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
def find_face(img):
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

        """
        # draw rectangle in big image
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)
        """

        # get subimage in ROI
        return img[y:y+h, x:x+h]

    return None
    """
    cv2.namedWindow('win')
    cv2.imshow('win', img)
    k = cv2.waitKey(5)
    if k == 27:
        sys.exit(0)
    """
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
        """
        # AN or N = angry, SU or U = suprised, SA or A = sad
        train_AN = self.load_all(img_folder+'train/*ANS.JPG')
        train_SU = self.load_all(img_folder+'train/*SUS.JPG')
        train_SA = self.load_all(img_folder+'train/*SAS.JPG')
        self.train_images = train_AN + train_SU + train_SA
        self.train_labels = len(train_AN)*'N'+len(train_SU)*'U'+len(train_SA)*'A'

        print 'loaded {} train_images'.format(self.num_train)

        self.train_data = numpy.array( [ x.flatten() for x in self.train_images ] )

        self.mean, self.evecs = cv2.PCACompute(self.train_data, mean=None, maxComponents=40)

        self.num_vecs = self.evecs.shape[0]
        print 'got {} eigenvectors'.format(self.num_vecs)

        self.train_proj = self.project_all(self.train_images)
        """
        # AN or N = angry, SU or U = suprised, SA or A = sad
        test_AN = self.load_all(img_folder+'test/*ANS.JPG')
        test_SU = self.load_all(img_folder+'test/*SUS.JPG')
        test_SA = self.load_all(img_folder+'test/*SAS.JPG')

        #self.test_datasets = [test_AN,test_SU,test_SA]
        self.test_datasets = test_AN + test_SU + test_SA
        #self.test_datasets = [resize_square(find_face(x)) for x in self.test_datasets]
        self.test_labels = len(test_AN)*'N'+len(test_SU)*'U'+len(test_SA)*'A'

        self.num_test = len(self.test_labels)
        #self.test_proj = self.project_all(self.test_images)
        print 'loaded {} test_images'.format(self.num_test)

        train_AN = self.load_all(img_folder+'train/*ANS.JPG')
        train_SU = self.load_all(img_folder+'train/*SUS.JPG')
        train_SA = self.load_all(img_folder+'train/*SAS.JPG')

        self.train_datasets = train_AN + train_SU + train_SA
        #self.train_datasets = [resize_square(find_face(x)) for x in self.train_datasets]

        self.train_labels = len(train_AN)*'N'+len(train_SU)*'U'+len(train_SA)*'A'
        self.num_train = len(self.train_labels)
        print 'loaded {} train_images'.format(self.num_train)

        #init variables to be filled by demo_mean()
        self.mean = [0,0,0]
        self.evecs = [0,0,0]
        self.train_proj = [0,0,0]

        self.image_w = self.image_shape[1]
        self.image_h = self.image_shape[0]
        self.image_pixels = self.image_w*self.image_h
        self.row_shape = (1, self.image_pixels)


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
        win = 'Menu'
        img = 255*numpy.ones((150, 250, 3), dtype='uint8')
        strings = [
            'Keys:',
            '',
            's - run & SHOW mean+evecs',
            'h - run & HIDE mean+evecs',
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
                func = self.demo_mean
            elif k == ord('h'):
                self.show = False
                func = self.demo_mean
            elif k == 27:
                break
            if func is not None:
                cv2.destroyWindow(win)
                func()
                self.demo_classify()
                self.make_window(win)
                cv2.imshow(win, img)

    """ #Version that had three separate sets of evecs and means
    def demo_means(self):
        win = 'Mean and vectors'
        self.make_window(win)

        for k in range(len(self.train_datasets)):
            self.train_datasets[k] = numpy.array( [ x.flatten() for x in self.train_datasets[k] ] )
            self.means[k], self.evecs[k] = cv2.PCACompute(self.train_datasets[k], mean=None, maxComponents=20)
            num_vecs = self.evecs[k].shape[0]
            print 'got {} eigenvectors'.format(num_vecs)

            print 'showing mean'
            img = self.means[k].reshape(self.image_shape)
            cv2.imshow(win, upscale(img))
            if self.should_quit(): break

            for i in range(num_vecs):
                print 'showing evec', i+1, 'of', num_vecs
                img = self.visualize_evec(self.evecs[k][i])
                cv2.imshow(win, upscale(img))
                if self.should_quit(): break

            self.train_proj[k] = self.project_all(self.train_datasets[k], k)
            #TODO: take out of loop?

        self.test_proj = self.project_all(self.test_datasets, k=0)
        print 'shape of test_proj', self.test_proj.shape


        cv2.destroyWindow(win)
        print len(self.train_proj)
        print self.train_proj[0].shape,
        print self.train_proj[1].shape,
        print self.train_proj[2].shape
    """
    def demo_mean(self):
        win = 'Mean and vectors'
        self.make_window(win)

        self.train_datasets = numpy.array( [ x.flatten() for x in self.train_datasets ] )
        self.mean, self.evecs = cv2.PCACompute(self.train_datasets, mean=None, maxComponents=20)
        self.num_vecs = self.evecs.shape[0]
        print 'got {} eigenvectors'.format(self.num_vecs)

        if self.show:
            print 'showing mean'
            img = self.mean.reshape(self.image_shape)
            cv2.imshow(win, upscale(img))
            if self.should_quit(): return

            for i in range(self.num_vecs):
                print 'showing evec', i+1, 'of', self.num_vecs
                img = self.visualize_evec(self.evecs[i])
                cv2.imshow(win, upscale(img))
                if self.should_quit(): break

        #project train and test datasets through the eigenvectors
        self.train_proj = self.project_all(self.train_datasets)
        self.test_proj = self.project_all(self.test_datasets)
        print 'shape of test_proj', self.test_proj.shape

        cv2.destroyWindow(win)

    def demo_reconstruct(self):
        win = 'Left: orig., Right: recons.'
        self.make_window(win)
        i = 0
        while 1:
            big = numpy.hstack( ( demo.train_images[i],
                                  self.spacer(),
                                  demo.backproject(demo.train_proj[i]) ) )
            cv2.imshow(win, upscale(big))
            if self.should_quit(): break
            i = (i + 1) % self.num_train
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
        #m = 20
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
                wrong.append(self.test_labels[i]+self.train_labels[matches[i,0]])
        print '\ngot', num_correct, 'correct out of', matches.shape[0]
        print '% correct =', num_correct/float(matches.shape[0])
        print 'wrong were (L-actual, R-NN guess):\n', wrong

#       Linear SVC Stuff
#        SVC = sklearn.svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#            intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#            multi_class='ovr', penalty='12', random_state=None, tol=0.0001,
#            verbose=0)
#
#        train_new = SVC.fit(train_proj, test_proj

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

    def demo_vecs(self):

        win = 'Mean and vectors'
        self.make_window(win)
        i = self.num_vecs


        while 1:
            if (i >= self.num_vecs):
                print 'showing mean'
                img = self.mean.reshape(self.image_shape)
            else:
                print 'showing evec'
                img = self.visualize_evec(self.evecs[i])
            cv2.imshow(win, upscale(img))
            if self.should_quit(): break
            i = (i + 1) % (self.num_vecs + 1)
        cv2.destroyWindow(win)

    def trk_change(self, i, value):
        #print '{}={}'.format(which, value)
        wmin = self.wmin[i]
        wmax = self.wmax[i]
        self.w[i] = wmin + (value/100.0)*(wmax-wmin)
        img = self.backproject(self.w)
        cv2.imshow('Output', upscale(img))

    def update_w(self):
        img = self.backproject(self.w)
        cv2.imshow('Output', upscale(img))
        v = (100 * (self.w - self.wmin) / (self.wmax - self.wmin)).astype('int')
        v[v < 0] = 0
        v[v > 100] = 100
        for i in range(self.num_vecs):
            cv2.setTrackbarPos(self.tnames[i], self.twins[i], v[i])

    def randomize_w(self, scale):
        self.w = numpy.random.normal(scale=scale, size=self.num_vecs)
        self.update_w()

    def reset_w(self):
        self.w[:] = 0
        self.update_w()

    def demo_pca(self):

        win = 'Output'
        self.make_window(win)

        numwins = 0
        wincnt = -1
        winx = self.image_w*zoom + 20+50

        self.w = numpy.zeros(self.num_vecs, dtype='float32')
        self.wmin = self.test_proj.min(axis=0)
        self.wmax = self.test_proj.max(axis=0)

        self.twins = []
        self.tnames = []
        ewins = []

        for i in range(self.num_vecs):

            if wincnt < 0 or wincnt > self.num_vecs/2:
                numwins += 1
                ewin = 'Eigenvectors {}'.format(numwins)
                self.make_window(ewin, x=winx)
                winx += 250
                wincnt = 0
                cv2.imshow(ewin, numpy.zeros((1,200)))
                ewins.append(ewin)

            tname = 'Eigenvector {}'.format(i+1)
            tfunc = lambda value, which=i: self.trk_change(which, value)
            cv2.createTrackbar(tname, ewin, 50, 100, tfunc)
            #cv2.imshow(ewin, numpy.zeros((1,200)))
            self.twins.append(ewin)
            self.tnames.append(tname)

            wincnt += 1

        img = self.mean.reshape(self.image_shape)
        cv2.imshow(win, upscale(img))
        while 1:
            k = cv2.waitKey(5)
            if k == 27:
                break
            elif k == ord('r'):
                self.randomize_w(0.75)
            elif k == ord('R'):
                self.randomize_w(3.0)
            elif k == ord('z'):
                self.reset_w()

        cv2.destroyWindow(win)
        for ewin in ewins: cv2.destroyWindow(ewin)

    def load_all(self, pattern):
        rval = []
        for imgfile in glob.glob(pattern):
            img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

            # extract just the find_face
            img = resize_square(find_face(img))

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

    demo.demo_menu()

    #demo.demo_pca()
    #demo.demo_vecs()
    #demo.demo_classify()
    #demo.demo_reconstruct()
