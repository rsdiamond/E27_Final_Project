########################################################################
#
# File:   classify.py
# Author: Julie Harris, Rachel Diamond
# Date:   May, 2016
#
# This program is heavily based on Matt Zucker's code from April 2012
# in the program textons.py
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# This file implements the paper:
#
#    Varma, Manik, and Andrew Zisserman. "A statistical approach to
#    texture classification from single images." International Journal
#    of Computer Vision 62.1-2 (2005): 61-81.
#
# http://www.swarthmore.edu/NatSci/mzucker1/e27_s2015/varma05textons.pdf
#
########################################################################

import cv2
import numpy
import struct
import os
import sys
import math
import json
import re
import datetime

######################################################################

ccolors = [
    (255,   0,   0),
    (255,  63,   0),
    (255, 127,   0),
    (255, 191,   0),
    (255, 255,   0),
    (191, 255,   0),
    (127, 255,   0),
    ( 63, 255,   0),
    (  0, 255,   0),
    (  0, 255,  63),
    (  0, 255, 127),
    (  0, 255, 191),
    (  0, 255, 255),
    (  0, 191, 255),
    (  0, 127, 255),
    (  0,  63, 255),
    (  0,   0, 255),
    ( 63,   0, 255),
    (127,   0, 255),
    (191,   0, 255),
    (255,   0, 191),
    (255,   0, 255),
    (255,   0, 127),
    (255,   0,  63)
    ]

ccolors2 = []
x = 0

for i in range(len(ccolors)):
    ccolors2.append(tuple(reversed(ccolors[x])))
    x = (x + 7) % len(ccolors)

ccolors = ccolors2

######################################################################

def rescaleL1(image, meanShift):
    if (meanShift):
        image[:] -= numpy.mean(image)
    image[:] *= 1.0 / cv2.norm(image, cv2.NORM_L1)

def rescaleMinMax(src, dst=None, dtype='uint8', maxValue=255):
    fmin = numpy.min(src[:])
    fmax = numpy.max(src[:])
    if dst is None:
        dst = numpy.empty(src.shape, dtype=dtype)
    dst[:] = maxValue * (src[:]-fmin)/(fmax-fmin)
    return dst

def rescaleInf(src, dst=None, dtype='uint8', midValue=127):
    finf = cv2.norm(src, cv2.NORM_INF)
    if dst is None:
        dst = numpy.empty(src.shape, dtype=dtype)
    dst[:] = src*midValue/finf + midValue
    return dst

def whiten(image):
    image[:] -= numpy.mean(image)
    mag = cv2.norm(image, cv2.NORM_L2)
    stddev = mag / math.sqrt(image.shape[0] * image.shape[1])
    image[:] *= 1.0/stddev

def weberTransform(src, scale):
    L = cv2.norm(src, cv2.NORM_L2)
    s = math.log(1 + L/scale) / L

######################################################################

def g1d(sigma, mu, x, order):
    x -= mu
    num = x*x
    variance = sigma*sigma
    denom = 2*variance
    g = math.exp(-num/denom) / math.sqrt(math.pi*denom)
    if order == 1:
        g *= -(x/variance)
    elif order == 2:
        g *= ((num-variance)/(variance*variance))
    return g

def makeGaussDerivFilter(w, angle, sx, sy, dx, dy):
    hw = int(w / 2)
    w = 2*hw + 1
    c = math.cos(angle)
    s = math.sin(angle)
    kernel = numpy.empty((w,w), dtype='float32')
    for i in range(w):
        fi = i-hw
        for j in range(w):
            fj = j-hw
            x = c*fj - s*fi
            y = s*fj + c*fi
            gx = g1d(sx, 0, x, dx)
            gy = g1d(sy, 0, y, dy)
            g = gx*gy
            kernel[i,j] = g
    rescaleL1(kernel, False)
    return kernel

######################################################################

def makeLoGFilter(w, sigma):
    hw = int(w/2)
    w = 2*hw + 1
    kernel = numpy.empty((w,w), dtype='float32')
    s2 = sigma*sigma
    s4 = s2*s2
    for i in range(w):
        y = float(i-hw)
        y2 = y*y
        for j in range(w):
            x =  float(j-hw)
            x2 = x*x
            kernel[i,j] = ((x2+y2-2*s2)/(s4)) * math.exp( -(x2+y2)/(2*s2) );
    rescaleL1(kernel, False)
    return kernel

######################################################################

def makeRsymGaborFilter(w, sigma, tau):
    hw = int(w/2)
    w = 2*hw + 1
    kernel = numpy.empty((w,w), dtype='float32')
    cscl = math.pi * tau / sigma
    escl = 1./(2*sigma*sigma)
    for i in range(w):
        y = float(i-hw)
        y2 = y*y
        for j in range(w):
            x =  float(j-hw)
            x2 = x*x
            r2 = x2+y2
            r = math.sqrt(r2)
            kernel[i,j] = math.cos(r*cscl)*math.exp(-r2*escl)
    rescaleL1(kernel, True)
    return kernel

######################################################################

def filterFromString(config, fstr):
    m = re.search('\s*(\w+)\(([^)]+)\)\s*', fstr)
    assert(m)
    name = m.group(1)
    args = [config['filterSize']] + map(
        float, re.split(',\s*', m.group(2).strip()))
    if name == 'gaussian':
        args[1] = args[1] * math.pi / 180
        f = makeGaussDerivFilter(*args)
    elif name == 'laplacian':
        f = makeLoGFilter(*args)
    elif name == 'rsym':
        f = makeRsymGaborFilter(*args)
    else:
        assert(0 and 'bad name for filter')
    return f

def makeFilters(config, parsed):

    filters = []

    r = config['report']
    if r:
        imgdir = 'images_' + config['stamp']
        print >>r, '<h3>Filters:</h3>'

    count = 0

    for item in parsed:
        if isinstance(item, str) or isinstance(item, unicode):
            f = filterFromString(config, item)
            filters.append([f])
            if r:
                fdisplay = rescaleInf(f)
                fname = os.path.join(imgdir, 'filter%03d.png' % count)
                cv2.imwrite('html/'+fname, fdisplay)
                print >>r, '<img align="middle" src="' + fname + '"/>'
            count += 1
        else:
            ff = []
            if r:
                print >>r, '<p>max('
            for subitem in item:
                f = filterFromString(config, subitem)
                if r:
                    fdisplay = rescaleInf(f)
                    fname = os.path.join(imgdir, 'filter%03d.png' % count)
                    cv2.imwrite('html/'+fname, fdisplay)
                    print >>r, '<img align="middle" src="' + fname + '"/>'
                count += 1
                ff.append(f)
            if r:
                print >>r, ')</p>'
            filters.append(ff)

    if r: r.flush()

    fsz = config['filterSize']**2
    nfilt = len(filters)

    fbank = numpy.empty( (nfilt, fsz), dtype='float32' )

    for i in range(nfilt):
        frow = filters[i][0].reshape( (1, fsz ) )
        fbank[i,:] = frow

    filtersInv = cv2.SVDecomp(fbank)

    return (filters, filtersInv)

def invertFilters(config, finv, response):
    fsz = config['filterSize']
    img = cv2.SVBackSubst(finv[0], finv[1], finv[2], centers[c,:])
    img.shape = (fsz, fsz)
    return img


def getFilterResponses(config, timage, filters, dst=None):

    subrect = timage['subrect']
    area = timage['area']
    nfilt = len(filters)

    if dst is None:
        dst = numpy.empty( (area, nfilt), dtype='float32' )
    else:
        assert(dst.shape[0] == area)
        assert(dst.shape[1] == nfilt)

    for i in range(nfilt):

        maxResponse = None

        for f in filters[i]:
            response = cv2.filter2D(subrect, cv2.CV_32F, f)
            if maxResponse is None:
                maxResponse = response.copy()
            else:
                maxResponse[:] = numpy.maximum(maxResponse, response)

        dst[:,i] = maxResponse.reshape(area)

    if config['weberTransformResponses']:
        for i in range(dst.shape[0]):
            weberTransform(dst[i,:], config['weberScale'])

    return dst

######################################################################

def loadImage(config, timage):

    fname = os.path.join(config['baseDir'], timage)
    tclass = os.path.split(os.path.split(timage)[0])[1]

    hash32 = hash(timage)
    hash16 = [ hash32 / 65536, hash32 % 65536 ]
    hash8  = [ hash16[0] / 256, hash16[0] % 256,
               hash16[1] / 256, hash16[1] % 256 ]

    img = cv2.imread(fname)
    if img is None:
        print 'error opening ' + fname
        sys.exit(1)

    w = img.shape[1]
    h = img.shape[0]
    x = 0
    y = 0

    if w > config['maxImageSize'] or h > config['maxImageSize']:
        ncols = min(w, config['maxImageSize'])
        nrows = min(h, config['maxImageSize'])
        if config['randomOffset']:
            if (w > ncols):
                x = hash16[0] % (w - ncols)
            if (h > nrows):
                y = hash16[1] % (h - nrows)
        else:
            x = (w - ncols) / 2
            y = (h - nrows) / 2
        w = ncols
        h = nrows

    if config['randomRotate'] and (hash8[0] % 2):
        img = img.transpose()

    if config['randomFlip']:
        flip = hash8[1] % 4
        if (flip!=3):
            cv2.flip(img, flip-1, img)

    bwimg = cv2.cvtColor(img[y:y+h, x:x+w],
                         cv2.COLOR_RGB2GRAY).astype('float')/255.0

    w = bwimg.shape[1]
    h = bwimg.shape[0]
    whiten(bwimg)

    hw = int(config['filterSize'])/2
    subrect = bwimg[hw:h-2*hw, hw:w-2*hw]

    return (tclass, img, bwimg, subrect)

def loadImages(config, tset, warnDifferent):
    tdict = {'images': [], 'area': 0}
    prevclass = None
    same = True
    for timage in tset:
        (tclass, image, bwimage, subrect) = loadImage(config, timage)
        if same and prevclass is None:
            prevclass = tclass
        elif same and tclass != prevclass:
            if warnDifferent:
                print ('warning: class '+tclass+
                       ' differs from previous '+prevclass)
            same = False
            prevclass = None
        idict = {
            'filename': timage,
            'classname': tclass,
            'image': image,
            'bwimage': bwimage,
            'subrect': subrect,
            'area': subrect.shape[0] * subrect.shape[1],
            }
        tdict['area'] += idict['area']
        tdict['images'].append(idict)

    tdict['classname'] = prevclass
    return tdict

######################################################################

def drawHistogram(height, histogram, k):

    nhist = len(histogram)

    s = 2
    while (2*height > s*nhist-1): s += 1

    himage = 220 * numpy.ones( (height, s*nhist-1, 3), dtype='uint8' )

    hmax = numpy.max(histogram)

    for c in range(nhist):
        label = c / k
        color = ccolors[ label % len(ccolors) ]
        hc = histogram[c] / hmax
        y = int(hc*height)
        if (y):
            startcol = s*c
            endcol = s*(c+1)-1
            color = numpy.array(color).reshape( (1,1,3) )
            himage[0:y, startcol:endcol, :] = color

    cv2.flip(himage, 0, himage)

    return himage

######################################################################

def startReport(config):

    if not config['generateHTMLOutput']:
        config['report'] = None
        return

    config['stamp'] = datetime.datetime.now().strftime('%m%d%y_%H%M%S')
    htmldir = 'html'

    if not os.path.exists(htmldir):
        os.mkdir(htmldir)

    imgdir = os.path.join(htmldir, 'images_' + config['stamp'])
    if not os.path.exists(imgdir):
        os.mkdir(imgdir)

    reportfile = 'report_' + config['stamp'] + '.html'
    fullfile = os.path.join(htmldir, reportfile)

    config['report'] = open(fullfile, 'w')
    r = config['report']

    if not r:
        return

    if sys.platform != 'win32':

        latest = os.path.join(htmldir, 'report_latest.html')

        if os.path.exists(latest) and os.path.islink(latest):
            os.unlink(latest)

        if not os.path.exists(latest):
            os.symlink(reportfile, latest)

    print >>r, '<?xml version="1.0" encoding="utf-8"?>'
    print >>r, '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"',
    print >>r, '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">'
    print >>r, '<html><head><title>Textons</title></head><body>'

def endReport(config):
    r = config['report']
    if r:
        print >>r, '</body></html>'
        r.close()
        print
        print 'wrote report to html/report_' + config['stamp'] + '.html'

######################################################################

if len(sys.argv) != 4:
    print 'usage: python', sys.argv[0], 'K FILTERSET.js DATASET.js'
    print ' e.g.: python', sys.argv[0], '4 lm4_filters.js synth_easy_dataset.js'
    print
    sys.exit(0)

k = int(sys.argv[1])
filters_file = sys.argv[2]
dataset_file = sys.argv[3]

config = json.load(open('config.js'))

dataset = json.load(open(dataset_file))

assert(isinstance(dataset['textonSets'], list))
assert(isinstance(dataset['textonSets'][0], list))
assert(isinstance(dataset['textonSets'][0][0], str) or
       isinstance(dataset['textonSets'][0][0], unicode))

if dataset.has_key('trainingSets') and dataset['trainingSets'] is None:
    dataset['trainingSets'] = dataset['textonSets']

assert(isinstance(dataset['trainingSets'], list))
assert(isinstance(dataset['trainingSets'][0], list))
assert(isinstance(dataset['trainingSets'][0][0], str) or
       isinstance(dataset['trainingSets'][0][0], unicode))

assert(isinstance(dataset['testSet'], list))
assert(isinstance(dataset['testSet'][0], str) or
       isinstance(dataset['testSet'][0], unicode))

assert( k )

if k < 0:
    k = -k * len(dataset['textonSets'])
    dataset['textonSets'] = [ reduce(lambda x, y: x+y, dataset['textonSets']) ]

startReport(config)
r = config['report']
imgdir = 'images_' + config['stamp']
print >>r, '<h3>Arguments: %s</h3>' % (str(sys.argv))
(filters, filtersInv) = makeFilters(config, json.load(open(filters_file)))

textonSets = []
trainingSets = []

for tset in dataset['textonSets']:
    textonSets.append(loadImages(config, tset, False))

for tset in dataset['trainingSets']:
    trainingSets.append(loadImages(config, tset, True))

testSet = loadImages(config, dataset['testSet'], False)

nfilt = len(filters)
ntex = len(textonSets)
ntrain = len(trainingSets)
nbins = k*ntex

allcenters = numpy.empty( (nbins, nfilt), dtype = 'float32' )


######################################################################

for i in range(ntex):

    tset = textonSets[i]
    totalarea = tset['area']

    print 'texton set', i, 'has class', tset['classname']

    hugeresponses = numpy.empty( (totalarea, nfilt), dtype='float32' )

    print '  getting responses...'

    nimgs = len(tset['images'])

    if r:
        print >>r, '<h3>Texton batch %d</h3>' % (i+1)
        print >>r, '<p>Input images:'
        if (config['maxHTMLTrainImages'] and
            nimgs > config['maxHTMLTrainImages']):
            print >>r, '(%d of %d images shown)' % (config['maxHTMLTrainImages'], nimgs )
            nimgs = config['maxHTMLTrainImages']
        print >>r, '</p>'
        r.flush()

    startrow = 0
    count = 0

    for timage in tset['images']:
        print '    image %d of %d' % (count+1, len(tset['images']))
        endrow = startrow + timage['area']
        responses = hugeresponses[startrow:endrow, :]
        getFilterResponses( config, timage, filters, responses )
        startrow = endrow
        if r and count < nimgs:
            fdisplay = rescaleMinMax(timage['subrect'])
            fname = os.path.join(imgdir, os.path.basename(timage['filename']))
            cv2.imwrite('html/'+fname, fdisplay)
            print >>r, '<img align="middle" src="' + fname + '"/>'
        count += 1


    # now run k-means on hugeresponses
    ctype = cv2.TERM_CRITERIA_EPS
    if config['kmeansMaxIter']:
        ctype = ctype | cv2.TERM_CRITERIA_COUNT

    criteria = (ctype,
                config['kmeansMaxIter'],
                config['kmeansEps'])

    flags = cv2.KMEANS_RANDOM_CENTERS
    #flags = cv2.KMEANS_PP_CENTERS

    print '  running k-means...'

    (retval, bestLabels, centers) = cv2.kmeans(
        hugeresponses, k, None, criteria,
        config['kmeansAttempts'], flags)

    if r:
        print >>r, '<p/>Textons:</p>'
        for c in range(k):
            img = invertFilters(config, filtersInv, centers[c,:])
            fdisplay = rescaleMinMax(img)
            fname = os.path.join(imgdir, 'texton_%03d_%03d.png' % (i, c))
            cv2.imwrite('html/'+fname, fdisplay)
            print >>r, '<img align="middle" src="' + fname + '"/>'

    allcenters[k*i:k*(i+1), :] = centers


alllabels = numpy.array( range(0, nbins), dtype='int' )


FLANN_INDEX_KDTREE = 1

nn = cv2.flann.Index(allcenters,
                     dict(algorithm = FLANN_INDEX_KDTREE, trees = 4))

######################################################################

textonDict = []

for i in range(ntrain):

    tset = trainingSets[i]
    classname = tset['classname']

    print 'training set', i, 'has class', classname
    print '  getting responses...'

    nimgs = len(tset['images'])

    if r:
        print >>r, '<h3>Training images for <tt>' + classname + '</tt></h3>'
        if (config['maxHTMLTrainImages'] and
            nimgs > config['maxHTMLTrainImages']):
            print >>r, '(%d of %d images shown)<br/>' % (
                config['maxHTMLTrainImages'], nimgs )
            nimgs = config['maxHTMLTrainImages']
        r.flush()

    count = 0

    for timage in tset['images']:

        print '    image %d of %d' % (count+1, len(tset['images']))

        responses = getFilterResponses( config, timage, filters )

        matches, dist = nn.knnSearch(responses, 1, params={})
        results = alllabels[matches].astype('uint8')

        histogram = cv2.calcHist([results], [0], None, [nbins], [0, nbins])
        histogram /= timage['area']

        if r and count < nimgs:
            fdisplay = rescaleMinMax(timage['subrect'])
            fname = os.path.join(imgdir, os.path.basename(timage['filename']))
            cv2.imwrite('html/'+fname, fdisplay)
            print >>r, '<p>'
            print >>r, '<img align="middle" src="' + fname + '"/>'
            himage = drawHistogram( fdisplay.shape[0], histogram, k )
            fname = os.path.join(imgdir, 'histogram_%s_%d.png' % (
                classname, count))
            cv2.imwrite('html/'+fname, himage)
            print >>r, '<img align="middle" src="' + fname + '"/>'
            print >>r, '</p>'
        count += 1


        #cv2.imshow('win', himage)
        #while cv2.waitKey(5) < 0: pass
        textonDict.append({'classname': timage['classname'],
                           'histogram': histogram})

######################################################################

print 'testing...'
count = 0
accuracy = 0

nimgs = len(testSet['images'])

if r:
    print >>r, '<h3>Test images:</h3>'
    if (config['maxHTMLTestImages'] and
        nimgs > config['maxHTMLTestImages']):
        print >>r, '(%d of %d images shown)<br/>' % (
            config['maxHTMLTestImages'], nimgs )
        nimgs = config['maxHTMLTestImages']
    r.flush()


for timage in testSet['images']:

    responses = getFilterResponses( config, timage, filters )
    matches, dist = nn.knnSearch(responses, 1, params={})
    results = alllabels[matches].astype('uint8')
    histogram = cv2.calcHist([results], [0], None, [nbins], [0, nbins])
    histogram /= timage['area']

    bestDist = None
    bestClass = None

    for d in textonDict:
        dist = cv2.compareHist(histogram, d['histogram'], cv2.HISTCMP_CHISQR)
        if bestDist is None or dist < bestDist:
            bestDist = dist
            bestClass = d['classname']

    correct = (bestClass == timage['classname'])
    if correct:
        resultstr = 'correct'
        accuracy += 1
    else:
        resultstr = 'INCORRECT'

    print ('  image % 3d of % 3d: actual=% 20s, predicted=% 20s, result=%s'
           % (count+1, len(testSet['images']), timage['classname'],
              bestClass, resultstr) )

    if r and count < nimgs:
        print >>r, '<p>'
        fdisplay = rescaleMinMax(timage['subrect'])
        fname = os.path.join(imgdir, os.path.basename(timage['filename']))
        cv2.imwrite('html/'+fname, fdisplay)
        print >>r, '<img align="middle" src="' + fname + '"/>'
        himage = drawHistogram( fdisplay.shape[0], histogram, k )
        fname = os.path.join(
            imgdir, 'histogram_%s_%d.png' % (classname, count))
        cv2.imwrite('html/'+fname, himage)
        print >>r, '<img align="middle" src="' + fname + '"/>'
        print >>r, '<br/>'
        if not correct:
            print >>r, '<span style="color: red">'
        print >>r, 'Actual class: <tt>%s</tt>,' % timage['classname']
        print >>r, 'predicted class: <tt>%s</tt>' % bestClass
        if not correct:
            print >>r, '</span>'
        print >>r, '</p>'

    count += 1

print 'total accuracy: %d/%d = %0.3f' % (
    accuracy, count, float(accuracy) / count)

if r:
    print >>r, '<h3>Total accuracy: %d/%d = %0.3f</h3>' % (
        accuracy, count, float(accuracy) / count)

    #cv2.imshow('win', rescaleMinMax(timage['bwimage']))
    #while cv2.waitKey(5) < 0: pass

######################################################################
endReport(config)
