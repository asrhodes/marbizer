from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
from PIL import ImageChops

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from os import listdir
from os import getcwd
from os import makedirs
from os.path import *

import sys
import math
import numpy

# spheremap, blackbounds, samplearr, and dospherize functions are taken/adapted from https://gist.github.com/kspi/2820038


#########################################################################################
################### SPHERE ANIMATION SHEET PRODUCTION FUNCTIONS #########################
#########################################################################################
def sheetdim(numframes, imagesize):
    """
    Optimizes the width and height of the spritesheet given image size and number of
    animation frames (tries to make it square and <= a power of 2). This saves RAM which
    is important on mobile devices.
    """
    # does not yet use imagesize,
    # does not yet guarantee <= 2^n
    square = math.sqrt(numframes)
    frameswide = math.ceil(square)
    frameshigh = math.floor(square)

    return (frameswide, frameshigh)


def sheetform(img, imagesize, numframes, frameswide, frameshigh):
    """
    Produces an animated spritesheet by first making a column of shifted image copies,
    then realigning them into a(n approximately) square sheet.
    """
    print("FORMING ANIMATION SHEET")

    # resize image
    img.thumbnail((imagesize, imagesize))

    # form row
    img_row = Image.new("RGBA", (imagesize * 2, imagesize * numframes))
    for row in range(numframes):
        offset = row * (imagesize // numframes)
        img_row.paste(img, (offset, row * imagesize))
        img_row.paste(img, (offset + imagesize, row * imagesize))

    left = imagesize
    top = 0
    right = imagesize * 2
    bottom = imagesize * numframes
    img_row = img_row.crop((left, top, right, bottom))

    # form sheet
    img_sheet = Image.new("RGBA", (img.width * frameswide, img.height * frameshigh))
    for i in range(1, numframes, 1):
        left = 0
        top = imagesize * i
        right = imagesize
        bottom = (imagesize * i) + imagesize
        temp_img = img_row.crop((left, top, right, bottom))

        x = ((i - 1) % frameswide) * imagesize
        y = ((i - 1) // frameshigh) * imagesize
        img_sheet.paste(temp_img, (x, y))

    # the first animation frame goes in the final sheet spot (AGK requirement)
    temp_img = img_row.crop((0, 0, imagesize, imagesize))
    x = ((numframes - 1) % frameswide) * imagesize
    y = ((numframes - 1) // frameshigh) * imagesize
    img_sheet.paste(temp_img, (x, y))

    return img_sheet


def spheremap(x2, y2):
    """
    Maps normalized cartesian coordinates to radial coordinates, and displaces them if
    within radius = 1 according to the form of a sphere.
    """
    r2 = math.sqrt(x2**2 + y2**2)
    if r2 > 1:
        x1 = x2
        y1 = y2
    else:
        theta1 = math.atan2(y2, x2)
        r1 = math.asin(r2) / (math.pi / 2)  # projection formula
        x1 = r1 * math.cos(theta1)
        y1 = r1 * math.sin(theta1)
    return (x1, y1)


def black_bounds(f, x, y):
    """
    Limits the sample function according to the size of the numpy array.
    """
    if 0 <= x < f.shape[0] and 0 <= y < f.shape[1]:
        return f[x, y]
    else:
        return 0


def samplearr(f, x, y):
    """
    Samples the pixel values of nearby pixels in the case that the image used is not the
    same size as the target animation frame size.and
    """
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1
    fy0 = black_bounds(f, x0, y0) * (x1 - x) + black_bounds(f, x1, y0) * (x - x0)
    fy1 = black_bounds(f, x0, y1) * (x1 - x) + black_bounds(f, x1, y1) * (x - x0)
    return fy0 * (y1 - y) + fy1 * (y - y0)


def dospherize(img, framesize, frameswide, frameshigh, dosample):
    """
    Iterates through the spritesheet and applies the spheremap formula.
    """
    old = numpy.array(img, dtype=numpy.double)
    depth = old.shape[2]
    arraywidth = framesize * frameswide
    arrayheight = framesize * frameshigh
    new = numpy.zeros((arraywidth, arrayheight, depth), dtype=numpy.double)

    for i in range(frameshigh):
        for j in range(frameswide):
            for ny in range(framesize):
                for nx in range(framesize):
                    newy = ny + (i * framesize)
                    newx = nx + (j * framesize)

                    # normalize to [-1, 1]
                    normx = nx / (framesize - 1) * 2 - 1
                    normy = ny / (framesize - 1) * 2 - 1

                    # project new coordinates using the spherize transform
                    projectedx, projectedy = spheremap(normx, normy)

                    # denormalize back to framesize pixel coordinates
                    denormx = ((projectedx * 0.5 + 0.5) * (framesize - 1)) + (
                        j * framesize
                    )

                    denormy = ((projectedy * 0.5 + 0.5) * (framesize - 1)) + (
                        i * framesize
                    )

                    if dosample:
                        new[newx, newy] = samplearr(old, denormx, denormy)
                    else:
                        denormx = int(denormx)
                        denormy = int(denormy)
                        new[newx, newy] = old[denormx, denormy]

    img2 = Image.fromarray(numpy.clip(new, 0, 0xFF).astype(numpy.uint8))
    return img2


#########################################################################################
########################## ORB SHEET PRODUCTION FUNCTIONS ###############################
#########################################################################################
def initglsphere(orbsize, ambc, difc, ca, la):
    """
    OpenGL function calls to establish viewport, perspective, lighting. This is called
    twice to create two orbs with distinct characteristics: a lighter one for a linear
    burn effect, and a dimmer one for an overlay effect.
    """
    glClearColor(1.0, 0.0, 1.0, 1.0)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    diffusepos = [6.5, 5.0, -12.0, 1.0]
    diffusecolor = [difc, difc, difc, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, diffusepos)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffusecolor)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, ca)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, la)
    glEnable(GL_LIGHT0)

    ambientcolor = [ambc, ambc, ambc, 1.0]
    glLightfv(GL_LIGHT1, GL_AMBIENT, ambientcolor)
    glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION, 0.0)
    glEnable(GL_LIGHT1)

    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)
    glViewport(0, 0, orbsize, orbsize)


def renderglsphere():
    """
    OpenGL function calls which simply uses glutSolidSphere. initglsphere must be called
    before this.
    """
    glClear(GL_COLOR_BUFFER_BIT)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glutSolidSphere(1.0, 100, 100)
    glFlush()


def gltopil(orbsize, ambc, difc, ca, la):
    """
    For staging, rendering, storing and converting an OpenGL generated orb to a PIL img.
    """
    initglsphere(orbsize, ambc, difc, ca, la)
    renderglsphere()
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, orbsize, orbsize, GL_RGBA, GL_UNSIGNED_BYTE)
    orbimg = Image.frombytes("RGBA", (orbsize, orbsize), data)
    orbimg = ImageOps.flip(orbimg)

    return orbimg


def makeorbs(orbsize):
    """
    OpenGL function calls which uses a disposable window to render and produce PIL
    images of the desired orbs.
    """
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(orbsize, orbsize)
    glutCreateWindow("render window")
    glutHideWindow()

    # for the light orb (overlay)
    lo = gltopil(orbsize, 1.4, 0.8, 1.15, 0.016)
    # for the shadow orb (linear burn)
    so = gltopil(orbsize, 2.0, 1.0, 1.5, 0.0085)

    return (lo, so)


def cleanorb(orb):
    """
    Image cleaning function which removes the magenta border (see glClearColor in
    function initglsphere) and gives the orbs a transparent background.
    """
    alphadat = []
    for pixel in orb.getdata():
        if pixel == (255, 0, 255, 255):
            alphadat.append((0, 0, 0, 0))
        else:
            alphadat.append(pixel)
    alphaorb = Image.new(orb.mode, orb.size)
    alphaorb.putdata(alphadat)
    return alphaorb


def orbsheetform(orb, framesize, frameswide, frameshigh):
    """
    Produces a sprite sheet (to be the same size as the animation sheet) which simply
    contains copies of the input orb. There will be two of these formed.
    """
    orbs = Image.new("RGBA", (framesize * frameswide, framesize * frameshigh))
    for i in range(frameshigh):
        for j in range(frameswide):
            orbs.paste(orb, (j * framesize, i * framesize))
    return orbs


def processorbs(orbsize, frameswide, frameshigh):
    """
    Generates, cleans, and forms orb sheets
    """
    lo, so = makeorbs(orbsize)
    lo, so = cleanorb(lo), cleanorb(so)
    lo = orbsheetform(lo, orbsize, frameswide, frameshigh)
    so = orbsheetform(so, orbsize, frameswide, frameshigh)
    return (lo, so)


#########################################################################################
###################### SHEET BLENDING AND CLEANING FUNCTIONS ############################
#########################################################################################
def shadows(img, so):
    """
    Application of the linear burn algorithm to animation sheet and shadow orb sheet.
    """
    print("APPLYING SHADOWS")
    invertedsorb = ImageChops.invert(so)
    invertedimg = ImageChops.invert(img)
    img = ImageChops.add(invertedimg, invertedsorb, 1.0, 0)
    img = ImageChops.invert(img)
    return img


def lights(img, lo):
    """
    Application of the overlay algorithm to animation sheet and light orb sheet.
    """
    print("APPLYING LIGHTS")
    img = ImageChops.overlay(img, lo)
    return img


def masksheet(img, frameswide, frameshigh):
    """
    Application of ellipses when cull and dangling pixels leaving an alpha background.
    """
    print("APPLYING MASKS")
    maskimg = Image.new("L", img.size, 0)
    width = img.width // frameswide
    height = img.height // frameshigh

    draw = ImageDraw.Draw(maskimg)
    for i in range(frameshigh):
        for j in range(frameswide):
            left = j * width
            top = i * height
            right = (j * width) + (width - 1)
            bottom = (i * height) + (height - 1)
            draw.ellipse((left, top, right, bottom), fill=255)

    img.putalpha(maskimg)
    return img


#########################################################################################
###################### FILE PARSING AND PROCESSING FUNCTIONS ############################
#########################################################################################
def checkimage(fstr):
    """
    Verifies that the given file argument is compatible with the program (PNG).
    """
    if fstr.endswith(".png"):
        return True
    else:
        print(fstr + " is not a png")
        print()
        return False


def walkargs():
    """
    Walks through provided command line arguments to find valid files. Currently accepts
    PNG files and directories which contain them (only one level deep).
    """
    files = []
    for i in range(1, len(sys.argv)):
        filename = sys.argv[i]
        if isdir(filename):
            for dfilename in listdir(filename):
                if isfile(join(filename, dfilename)):
                    if checkimage(dfilename):
                        files.append(join(filename, dfilename))

        else:
            if checkimage(filename):
                files.append(filename)
    return files


def processfile(f, n, lo, so, imgsize, numframes, frameswide, frameshigh, dosample):
    """
    Executes the spherize processes on the given file and saves the image.
    """
    print("USING FILE: {}".format(f))
    img = Image.open(f)
    img = sheetform(img, imgsize, numframes, frameswide, frameshigh)
    img = dospherize(img, imgsize, frameswide, frameshigh, dosample)
    img = shadows(img, so)
    img = lights(img, lo)
    img = masksheet(img, frameswide, frameshigh)

    if not exists(join(getcwd(), "marbizer-out")):
        makedirs(join(getcwd(), "marbizer-out"))

    img.save(join(getcwd(), "marbizer-out", "Marb" + str(n) + ".png"), format="PNG")
    img.close()
    print("COMPLETED MARBLE: {}".format("Marb" + str(n) + ".png"))
    print()


if __name__ == "__main__":
    # Haven't implemented full argument interpretation, for now just provide the default for my usage...
    imgsize = 128
    numframes = 16
    frameswide, frameshigh = sheetdim(numframes, imgsize)
    lo, so = processorbs(imgsize, frameswide, frameshigh)
    dosample = True  # set to false for slow computers...
    toprocess = walkargs()

    for i in range(len(toprocess)):
        f = toprocess[i]
        processfile(f, i, lo, so, imgsize, numframes, frameswide, frameshigh, dosample)
