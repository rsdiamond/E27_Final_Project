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

import cv2
import numpy
import sys
import glob

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

def upscale(img):
    return cv2.resize(img, (img.shape[1]*zoom, img.shape[0]*zoom),
                      interpolation=cv2.INTER_NEAREST)

def label_image(image, text):

    h = image.shape[0]

    cv2.putText(image, text, (8, h-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,0), 3, cv2.LINE_AA)

    cv2.putText(image, text, (8, h-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 1, cv2.LINE_AA)


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

        self.image_w = self.image_shape[1]
        self.image_h = self.image_shape[0]
        self.image_pixels = self.image_w*self.image_h

        self.num_train = len(self.train_images)
        self.row_shape = (1, self.image_pixels)

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

        self.test_datasets = [test_AN,test_SU,test_SA]
        self.test_images = test_AN + test_SU + test_SA
        self.test_labels = len(test_AN)*'N'+len(test_SU)*'U'+len(test_SA)*'A'

        self.num_test = len(self.test_images)
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
        win = 'Menu'
        img = 255*numpy.ones((150, 250, 3), dtype='uint8')
        strings = [
            'Keys:',
            '',
            'v - Demo eigenvectors',
            'r - Demo reconstruction',
            'c - Demo classification',
            'p - Demo PCA',
            'm - class means (Actually run)'
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
            if k == ord('v'):
                func = self.demo_vecs
            elif k == ord('r'):
                func = self.demo_reconstruct
            elif k == ord('c'):
                func = self.demo_classify
            elif k == ord('p'):
                func = self.demo_pca
            elif k == ord('m'):
                func = self.demo_means
            elif k == 27:
                break
            if func is not None:
                cv2.destroyWindow(win)
                func()
                self.make_window(win)
                cv2.imshow(win, img)

    def demo_means(self):
        win = 'Mean and vectors'
        self.make_window(win)

        train_AN = self.load_all(img_folder+'train/*ANS.JPG')
        train_SU = self.load_all(img_folder+'train/*SUS.JPG')
        train_SA = self.load_all(img_folder+'train/*SAS.JPG')

        # test getting eigenvectors for 1 img
        pic = numpy.array(train_AN[0].flatten())
        pic_mean, pic_evecs= cv2.PCACompute(pic, mean=None, maxComponents=20)
        print 'pic_mean', pic_mean
        print 'evecs', pic_evecs
        """
        num_vecs = pic_evecs.shape[0]
        print 'got {} eigenvectors'.format(num_vecs)

        print 'showing mean'
        img = pic_mean.reshape(self.image_shape)
        cv2.imshow(win, upscale(img))
        if self.should_quit(): return
        for i in range(num_vecs):
            print 'showing evec', i, 'of', num_vecs
            img = self.visualize_evec(pic_evecs[i])
            cv2.imshow(win, upscale(img))
            if self.should_quit(): break
        """

        self.train_datasets = [train_AN,train_SU,train_SA]
        self.means = [0,0,0]
        self.evecs = [0,0,0]
        for k in range(len(self.train_datasets)):
            self.train_datasets[k] = numpy.array( [ x.flatten() for x in self.train_datasets[k] ] )
            self.means[k], self.evecs[k] = cv2.PCACompute(self.train_datasets[k], mean=None, maxComponents=20)
            num_vecs = self.evecs[k].shape[0]
            print 'got {} eigenvectors'.format(num_vecs)

            print 'showing mean'
            img = self.means[k].reshape(self.image_shape)
            cv2.imshow(win, upscale(img))
            if self.should_quit(): return
            for i in range(num_vecs):
                print 'showing evec', i+1, 'of', num_vecs
                img = self.visualize_evec(self.evecs[k][i])
                cv2.imshow(win, upscale(img))
                if self.should_quit(): break

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
        matcher = cv2.flann.Index(self.train_proj,
                                  dict(algorithm = FLANN_INDEX_KDTREE,
                                       trees = 4))
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
        
        k = 1
        matches, dist = matcher.knnSearch(self.test_proj, k, params={})
        #m = 20
        print 'matches.shape', matches.shape
        print matches[0,0]

        win = 'Left: query, Right: NN'
        self.make_window(win)
        i = 0
        while 1:
            imgs = [ self.test_images[i] ]
            imgs.append( self.spacer() )
            for j in range(k):
                imgs.append( self.train_images[matches[i,j]] )
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
            img = img.astype('float32')/255.0
            if self.image_shape is None:
                self.image_shape = img.shape
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
