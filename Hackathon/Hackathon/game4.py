import pygame
import random
import sys
import pandas as pd
import numpy as np
import csv, random
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import pyaudio
import wave

import speech_recognition as sr


from PIL import Image
from pygame.locals import *
import time

FPS = 30
#sys.setrecursionlimit(3000)

basicfontsize = 200
bigfontsize = 40

windowHeight = 1000
windowWidth = 1000

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BRIGHTBLUE = (0, 50, 255)
DARKGREY = (60, 60, 60)
DARKTURQUOISE = (3, 54, 73)
GREEN = (34, 139, 34)
LIGHTGREEN = (154, 205, 50)
RED = (178, 34, 34)
LIGHTRED = (205, 92, 92)
PURPLE = (255, 0, 255)
GREY = (205, 201, 201)
NEW = (3, 150, 100)
NORMALGREY = (105, 105, 105)
ORANGERED = (255, 69, 0)
MAROON = (176, 48, 96)
DEEPPINK = (255, 20, 147)
DARKVIOLET = (148, 0, 211)
LIGHTYELLOW = (255, 215, 0)
YELLOW = (255, 165, 0)
HOTPINK = (255, 105, 180)
PERU = (205, 133, 65)
BLUE = (0, 0, 128)
LIGHTBLUE = (0, 0, 205)
secondWindowColor = DARKTURQUOISE
boardercolor = BLUE

filenames = ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt']

with open('training1.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())



df = pd.read_csv("training1.txt",sep = '\t',names = ['txt','liked'])
#df.head()

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf = True, lowercase = True, strip_accents = 'ascii', stop_words = stopset)

y = df.liked
x = vectorizer.fit_transform(df.txt)

print y.shape
print x.shape

#print y.head()
#print x

#x_train, x_test , y_train, y_test = train_test_split(x,y,random_state = 42)

clf = naive_bayes.MultinomialNB()
clf.fit(x,y)

print roc_auc_score(y, clf.predict_proba(x)[:,1])




class GIFImage(object):
    def __init__(self, filename):
        self.filename = filename
        self.image = Image.open(filename)
        self.frames = []
        self.get_frames()

        self.cur = 0
        self.ptime = time.time()

        self.running = True
        self.breakpoint = len(self.frames)-1
        self.startpoint = 0
        self.reversed = False

    def get_rect(self):
        return pygame.rect.Rect((0,0), self.image.size)

    def get_frames(self):
        image = self.image

        pal = image.getpalette()
        base_palette = []
        for i in range(0, len(pal), 3):
            rgb = pal[i:i+3]
            base_palette.append(rgb)

        all_tiles = []
        try:
            while 1:
                if not image.tile:
                    image.seek(0)
                if image.tile:
                    all_tiles.append(image.tile[0][3][0])
                image.seek(image.tell()+1)
        except EOFError:
            image.seek(0)

        all_tiles = tuple(set(all_tiles))

        try:
            while 1:
                try:
                    duration = image.info["duration"]
                except:
                    duration = 100

                duration *= .001 #convert to milliseconds!
                cons = False

                x0, y0, x1, y1 = (0, 0) + image.size
                if image.tile:
                    tile = image.tile
                else:
                    image.seek(0)
                    tile = image.tile
                if len(tile) > 0:
                    x0, y0, x1, y1 = tile[0][1]

                if all_tiles:
                    if all_tiles in ((6,), (7,)):
                        cons = True
                        pal = image.getpalette()
                        palette = []
                        for i in range(0, len(pal), 3):
                            rgb = pal[i:i+3]
                            palette.append(rgb)
                    elif all_tiles in ((7, 8), (8, 7)):
                        pal = image.getpalette()
                        palette = []
                        for i in range(0, len(pal), 3):
                            rgb = pal[i:i+3]
                            palette.append(rgb)
                    else:
                        palette = base_palette
                else:
                    palette = base_palette

                pi = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
                pi.set_palette(palette)
                if "transparency" in image.info:
                    pi.set_colorkey(image.info["transparency"])
                pi2 = pygame.Surface(image.size, SRCALPHA)
                if cons:
                    for i in self.frames:
                        pi2.blit(i[0], (0,0))
                pi2.blit(pi, (x0, y0), (x0, y0, x1-x0, y1-y0))

                self.frames.append([pi2, duration])
                image.seek(image.tell()+1)
        except EOFError:
            pass

    def render(self, screen, pos):
        if self.running:
            if time.time() - self.ptime > self.frames[self.cur][1]:
                if self.reversed:
                    self.cur -= 1
                    if self.cur < self.startpoint:
                        self.cur = self.breakpoint
                else:
                    self.cur += 1
                    if self.cur > self.breakpoint:
                        self.cur = self.startpoint

                self.ptime = time.time()

        screen.blit(self.frames[self.cur][0], pos)

    def seek(self, num):
        self.cur = num
        if self.cur < 0:
            self.cur = 0
        if self.cur >= len(self.frames):
            self.cur = len(self.frames)-1

    def set_bounds(self, start, end):
        if start < 0:
            start = 0
        if start >= len(self.frames):
            start = len(self.frames) - 1
        if end < 0:
            end = 0
        if end >= len(self.frames):
            end = len(self.frames) - 1
        if end < start:
            end = start
        self.startpoint = start
        self.breakpoint = end

    def pause(self):
        self.running = False

    def play(self):
        self.running = True

    def rewind(self):
        self.seek(0)
    def fastforward(self):
        self.seek(self.length()-1)

    def get_height(self):
        return self.image.size[1]
    def get_width(self):
        return self.image.size[0]
    def get_size(self):
        return self.image.size
    def length(self):
        return len(self.frames)
    def reverse(self):
        self.reversed = not self.reversed
    def reset(self):
        self.cur = 0
        self.ptime = time.time()
        self.reversed = False

    def copy(self):
        new = GIFImage(self.filename)
        new.running = self.running
        new.breakpoint = self.breakpoint
        new.startpoint = self.startpoint
        new.cur = self.cur
        new.ptime = self.ptime
        new.reversed = self.reversed
        return new



class Block1(pygame.sprite.Sprite):
    def __init__(self, filename,color):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = pygame.image.load(filename).convert()
        self.image = self.original_image
        self.rect = self.image.get_rect()
    
    def update(self):
        self.rect.x += self.change_x
        self.rect.y += self.change_y
 
        if self.rect.right >= self.right_boundary or self.rect.left <= self.left_boundary:
            self.change_x *= -1
 
        if self.rect.bottom >= self.bottom_boundary or self.rect.top <= self.top_boundary:
            self.change_y *= -1

 

class Player1(pygame.sprite.Sprite):
    def __init__(self, filename, color):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = pygame.image.load(filename).convert()
        self.image = self.original_image
        #self.image.set_colorkey(color)
        self.rect = self.image.get_rect()
 
    def update(self):
        pos = pygame.mouse.get_pos()
        self.rect.x = pos[0]


class Bullet(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
 
        self.image = pygame.Surface([10, 20])
        self.image.fill(RED)
 
        self.rect = self.image.get_rect()
 
    def update(self):
        self.rect.y -= 20




class Block(pygame.sprite.Sprite):
 
    def __init__(self, color, filename):
        pygame.sprite.Sprite.__init__(self)
        self.col = color
        self.original_image = pygame.image.load(filename).convert()
        self.image = self.original_image
        #self.image.set_colorkey(color)
        self.rect = self.image.get_rect()
 
    def update(self):
        self.rect.x += self.change_x
        self.rect.y += self.change_y
 
        if self.rect.right >= self.right_boundary or self.rect.left <= self.left_boundary:
            self.change_x *= -1
 
        if self.rect.bottom >= self.bottom_boundary or self.rect.top <= self.top_boundary:
            self.change_y *= -1
 


class Player(Block):
    def update(self):
        pos = pygame.mouse.get_pos()
        self.rect.x = pos[0]
        self.rect.y = pos[1]


quote = [("t1.wav","t1.jpg",4),("t2.wav","t2.jpg",6),("t3.wav","t3.jpg",8),("t4.wav","t4.jpg",7),("t5.wav","t5.jpg",5),("t6.wav","t6.jpg",5),("t7.wav","t7.jpg",9),("t8.wav","t8.jpg",4),("t9.wav","t9.jpg",4),("t10.wav","t10.jpg",3),("t11.wav","t11.jpg",10),("t12.wav","t12.jpg",6),("t13.wav","t13.jpg",7),("t14.wav","t14.jpg",3),("t15.wav","t15.jpg",4),("t16.wav","t16.jpg",4),("t17.wav","t17.jpg",9),("t18.wav","t18.jpg",3),("t19.wav","t19.jpg",4),("t20.wav","t20.jpg",6)]


def main():
    global FPSCLOCK, displaySurf, BASICFONT, BASICFONT2

    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    displaySurf = pygame.display.set_mode((windowWidth, windowHeight))
    pygame.display.set_caption('BEST FRIEND')

    BASICFONT = pygame.font.Font('freesansbold.ttf', basicfontsize)
    BASICFONT2 = pygame.font.Font('freesansbold.ttf', bigfontsize)

    firstWindow()
    sarah1Window()
    mood = secondWindow()
    
    if mood == "happy":
        sarah2Window()
        selectedGame = happyWindow()
        if selectedGame == 1:
            game1()
        elif selectedGame == 2:
            game2()

    elif mood == "sad":
        qut = random.choice(quote)
        
        sarah3Window()
        sarah4Window(qut)
        sadWindow()

    
def firstWindow():
     
     bgif = GIFImage("sarah.gif")
     pygame.mixer.music.load('intro.wav')
     pygame.mixer.music.play(0)
     time = pygame.time.get_ticks() + 10000
     while time >= pygame.time.get_ticks():
         ifquit()
         displaySurf.fill(WHITE)
         pygame.draw.rect(displaySurf, boardercolor, (0, 0, 1000, 1000), 100)
         # bgif = GIFImage("sarahNew.gif")
         bgif.render(displaySurf, (350, 350))
         pygame.display.flip()
     

     return

def sarah1Window():
     
     bgif = GIFImage("sarah.gif")
     pygame.mixer.music.load('AskDay.wav')
     pygame.mixer.music.play(0)
     time = pygame.time.get_ticks() + 5000
     while time >= pygame.time.get_ticks():
         ifquit()
         displaySurf.fill(WHITE)
         pygame.draw.rect(displaySurf, boardercolor, (0, 0, 1000, 1000), 100)
         # bgif = GIFImage("sarahNew.gif")
         bgif.render(displaySurf, (350, 350))
         pygame.display.flip()
     

     return

def sarah2Window():
     
     bgif = GIFImage("sarah.gif")
     pygame.mixer.music.load('happyConsole.wav')
     pygame.mixer.music.play(0)
     time = pygame.time.get_ticks() + 7000
     while time >= pygame.time.get_ticks():
         ifquit()
         displaySurf.fill(WHITE)
         pygame.draw.rect(displaySurf, boardercolor, (0, 0, 1000, 1000), 100)
         # bgif = GIFImage("sarahNew.gif")
         bgif.render(displaySurf, (350, 350))
         pygame.display.flip()
     

     return

def sarah3Window():
     
     bgif = GIFImage("sarah.gif")
     pygame.mixer.music.load('sadConsoling.wav')
     pygame.mixer.music.play(0)
     time = pygame.time.get_ticks() + 4000
     while time >= pygame.time.get_ticks():
         ifquit()
         displaySurf.fill(WHITE)
         pygame.draw.rect(displaySurf, boardercolor, (0, 0, 1000, 1000), 100)
         # bgif = GIFImage("sarahNew.gif")
         bgif.render(displaySurf, (350, 350))
         pygame.display.flip()
     

     return


def sarah4Window(qut):
     
     bgif = GIFImage("sarah.gif")
     pygame.mixer.music.load(qut[0])
     pygame.mixer.music.play(0)
     time = pygame.time.get_ticks() + 1000*(qut[2]) + 1000
     while time >= pygame.time.get_ticks():
         ifquit()
         displaySurf.fill(WHITE)
         #pygame.draw.rect(displaySurf, boardercolor, (0, 0, 1000, 1000), 100)
         bgif.render(displaySurf, (0, 300))
         img1 = pygame.image.load(qut[1])
         displaySurf.blit(img1, (400, 100))
         pygame.display.flip()
     

     return


def maketext2(
    text,
    color,
    backcolor,
    top,
    left,
    ):
    textsurf = BASICFONT2.render(text, True, color, backcolor)
    textrect = textsurf.get_rect()
    textrect.topleft = (top, left)
    return (textsurf, textrect)



def ifquit():
    for event in pygame.event.get(QUIT):
        pygame.quit()
        sys.exit()
    for event in pygame.event.get(KEYUP):
        if event.key == K_ESCAPE:
            pygame.quit()
            sys.exit()
        pygame.event.post(event)


def getCenter(a, b):
    for i in range(0, 1200, 40):
        for j in range(0, 750, 40):
            tilerect = pygame.Rect(i, j, 40, 40)
            if tilerect.collidepoint(a, b):
                return (i, j)

    return (None, None)

def secondWindow():
    t = 25
    strng = ''
    listOfString = []
    drawSecondWindow(strng)
    cnt1 = 0
    while True:
        
        drawSecondWindow(strng)
        ifquit()
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONUP:
                (spotx, spoty) = (event.pos[0], event.pos[1])
                (centerx, centery) = getCenter(spotx, spoty)
                if (centerx >= 440 and centerx < 520) and centery == 600:
                     if strng != '':
                         listOfString.append(strng)
                     answer = machineLearn(listOfString)
                     return answer     
            elif event.type == KEYUP:
                l = len(strng)
                if event.key == K_a and l <t:
                    strng += 'A'
                elif event.key == K_b and l < t:
                    strng += 'B'
                elif event.key == K_c and l <t:
                    strng += 'C'
                elif event.key == K_d and l <t:
                    strng += 'D'
                elif event.key == K_e and l <t:
                    strng += 'E'
                elif event.key == K_f and l <t:
                    strng += 'F'
                elif event.key == K_g and l <t:
                    strng += 'G'
                elif event.key == K_h and l <t:
                    strng += 'H'
                elif event.key == K_i and l <t: 
                    strng += 'I'
                elif event.key == K_j and l <t:
                    strng += 'J'
                elif event.key == K_k and l <t:
                    strng += 'K'
                elif event.key == K_l and l <t:
                    strng += 'L'
                elif event.key == K_m and l <t:
                    strng += 'M'
                elif event.key == K_n and l <t:
                    strng += 'N'
                elif event.key == K_o and l <t:
                    strng += 'O'
                elif event.key == K_p and l <t:
                    strng += 'P'
                elif event.key == K_q and l <t:
                    strng += 'Q'
                elif event.key == K_r and l <t:
                    strng += 'R'
                elif event.key == K_s and l <t:
                    strng += 'S'
                elif event.key == K_t and l <t:
                    strng += 'T'
                elif event.key == K_u and l <t:
                    strng += 'U'
                elif event.key == K_v and l <t:
                    strng += 'V'
                elif event.key == K_w and l <t:
                    strng += 'W'
                elif event.key == K_x and l <t:
                    strng += 'X'
                elif event.key == K_y and l <t:
                    strng += 'Y'
                elif event.key == K_z and l <t:
                    strng += 'Z'
                elif event.key == K_SPACE and l <t:
                    strng += ' '
                elif event.key == K_BACKSPACE and l >= 0:
                    strng = strng[0:len(strng) - 1]
                elif l >= t:
                    listOfString.append(strng)
                    l = 0
                    strng = ""
                

        pygame.display.update()
    

def drawSecondWindow(strng):

    displaySurf.fill(secondWindowColor)
    img1 = pygame.image.load('enter.png')
    displaySurf.blit(img1, (440, 600))
    (textsurf, textrect) = maketext2(strng, WHITE, BLACK, 150, 350)
    displaySurf.blit(textsurf, textrect)



def trainingModel(strng):
    m = np.array([strng])
    m_vector=vectorizer.transform(m)
    print clf.predict(m_vector)
    return clf.predict(m_vector)


def machineLearn(listOfString):
    cnt1 = 0
    cnt2 = 0 
    for strng in listOfString:
         ans = trainingModel(strng)
         if ans == 1:
              cnt1 = cnt1 + 1
         else :
              cnt2 = cnt2 + 1
    
    if cnt1 >= cnt2:
         return "happy"
    else :
         return "sad"




def happyWindow():
    
    bgif1 = GIFImage("games.gif")
    #bgif2 = GIFImage("shoot.gif")
    pygame.mixer.music.load('Motherfucker.mp3')
    pygame.mixer.music.play(0)
    #time = pygame.time.get_ticks() + 10000
    while True:
         ifquit()
         img1 = pygame.image.load('gameSelect.png')
         displaySurf.blit(img1, (0, 0))
         # displaySurf.fill(WHITE)
         # pygame.draw.rect(displaySurf, boardercolor, (0, 0, 1000, 1000), 100)
         # bgif = GIFImage("sarahNew.gif")
         bgif1.render(displaySurf, (80, 300))
         #bgif2.render(displaySurf, (580, 300))
         img1 = pygame.image.load('start.png')
         displaySurf.blit(img1, (620, 520))
         displaySurf.blit(img1, (120, 520))
   
         for event in pygame.event.get():
            if event.type == MOUSEBUTTONUP:
                (spotx, spoty) = (event.pos[0], event.pos[1])
                (centerx, centery) = getCenter(spotx, spoty)
                if (centerx >= 120 and centerx < 200) and centery == 520:
                     pygame.mixer.music.stop()
                     return 1
                elif (centerx >= 620 and centerx < 700) and centery == 520:
                     pygame.mixer.music.stop()
                     return 2 
         pygame.display.flip()
     

    return


def maketext(
    text,
    color,
    backcolor,
    top,
    left,
    ):
    textsurf = BASICFONT.render(text, True, DARKGREY, BLACK)
    textrect = textsurf.get_rect()
    textrect.topleft = (top, left)
    return (textsurf, textrect)


def game1():
    list1 = pygame.sprite.Group()
    sprites_list = pygame.sprite.Group()
    pygame.mixer.music.load('niceMusic.mp3')
    pygame.mixer.music.play(0)
    #list1 = []
    for i in range(50):
    
         block = Block(RED, "red.png")
         block.rect.x = random.randrange(windowWidth)
         block.rect.y = random.randrange(windowHeight)
         block.change_x = random.randrange(-3, 4)
         block.change_y = random.randrange(-3, 4)
         block.left_boundary = 0
         block.top_boundary = 0
         block.right_boundary = windowWidth
         block.bottom_boundary = windowHeight
         list1.add(block)
         sprites_list.add(block)

    player = Player(RED, "red.png")
    sprites_list.add(player)         
    done = False
 

    clock = pygame.time.Clock()
 
    score = 0

    while not done:
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 done = True
      
         
         displaySurf.fill(BLACK)
         #bgif = GIFImage("squaresBG.gif")
         #bgif.render(displaySurf, (0, 0))
         (textsurf, textrect) = maketext(str(score), GREY, BLACK, 400, 400)
         displaySurf.blit(textsurf, textrect)
         sprites_list.update()
         hit_list = pygame.sprite.spritecollide(player, list1, True)
      
         
         for block in hit_list:
             if block.col == BLACK:
                 done = True 
             else:
                 block = Block(BLACK, "black.png")
      
         
                 block.rect.x = random.randrange(windowWidth)
                 block.rect.y = random.randrange(windowHeight)
      
                 block.change_x = random.randrange(-3, 4)
                 block.change_y = random.randrange(-3, 4)
                 block.left_boundary = 0
                 block.top_boundary = 0
                 block.right_boundary = windowWidth
                 block.bottom_boundary = windowHeight
       
         
                 list1.add(block)
                 sprites_list.add(block)
                   
             score += 1
             print(score)
      
         
         sprites_list.draw(displaySurf)
         pygame.mouse.set_visible(False)
         clock.tick(3*FPS)
         pygame.display.flip()




def game2():
    list1 = pygame.sprite.Group()
    sprites_list = pygame.sprite.Group()

    bullet_list = pygame.sprite.Group()

    for i in range(30):
         block = Block1("monster.jpg",BLUE)
         block.rect.x = random.randrange(windowWidth)
         block.rect.y = random.randrange(windowHeight-400)
         block.change_x = random.randrange(-3, 4)
         block.change_y = random.randrange(-3, 4)
         block.left_boundary = 0
         block.top_boundary = 0
         block.right_boundary = windowWidth
         block.bottom_boundary = windowHeight-400
         list1.add(block)
         sprites_list.add(block)

    player = Player1("tank.jpg",RED)
    sprites_list.add(player)
    done = False
    clock = pygame.time.Clock()
    score = 0
    player.rect.y = windowHeight - 150
    #bgif = GIFImage("shootBG.gif")
    while not done:
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 done = True
      
             if event.type == KEYUP:
                
                 if event.key == K_SPACE:
                     bullet = Bullet()
                     bullet.rect.x = player.rect.x
                     bullet.rect.y = player.rect.y
                     sprites_list.add(bullet)
                     bullet_list.add(bullet)
                     pygame.mixer.music.load('button.mp3')
                     pygame.mixer.music.play(0)
      
         sprites_list.update()
         for bullet in bullet_list:
      
             
             hit_list = pygame.sprite.spritecollide(bullet, list1, True)
             for block in hit_list:
                 bullet_list.remove(bullet)
                 sprites_list.remove(bullet)
                 pygame.mixer.music.load('splat.mp3')
                 pygame.mixer.music.play(0)
                 score += 1
                 print(score)
             if bullet.rect.y < -50:
                 bullet_list.remove(bullet)
                 sprites_list.remove(bullet)
             
         displaySurf.fill(WHITE)
         img1 = pygame.image.load('shoot.png')
         displaySurf.blit(img1, (0, 0))
         #bgif = GIFImage("shootBG.gif")
         #bgif.render(displaySurf, (0, 0))
         sprites_list.draw(displaySurf)
         pygame.mouse.set_visible(False)
         pygame.display.flip()
         clock.tick(60)


def sadWindow():
     img1 = pygame.image.load('writeSand.jpg')
     displaySurf.blit(img1, (0, 0))
     bgif = GIFImage("sarah.gif")
     pygame.mixer.music.load('sandWrite.wav')
     pygame.mixer.music.play(0)
     time = pygame.time.get_ticks() + 14000
     while time >= pygame.time.get_ticks():
         ifquit()
         img1 = pygame.image.load('writeSand.jpg')
         displaySurf.blit(img1, (0, 0))
         bgif.render(displaySurf, (350, 350))
         pygame.display.flip()
     img1 = pygame.image.load('writeSand.jpg')
     displaySurf.blit(img1, (0, 0))

     CHUNK = 1
     FORMAT = pyaudio.paInt16
     CHANNELS = 2
     RATE = 44195
     RECORD_SECONDS = 5
     WAVE_OUTPUT_FILENAME = "output.wav"

     p = pyaudio.PyAudio()

     stream = p.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

     print("* recording")

     frames = []

     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
         data = stream.read(CHUNK)
         frames.append(data)

     print("* done recording")

     stream.stop_stream()
     stream.close()
     p.terminate()

     wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
     wf.setnchannels(CHANNELS)
     wf.setsampwidth(p.get_sample_size(FORMAT))
     wf.setframerate(RATE)
     wf.writeframes(b''.join(frames))
     wf.close()

     AUDIO_FILE = ("output.wav")
 
     # use the audio file as the audio source
      
     r = sr.Recognizer()
      
     with sr.AudioFile(AUDIO_FILE) as source:
         #reads the audio file. Here we use record instead of
         #listen
         audio = r.record(source)  
      
     try:
         print("The audio file contains: " + r.recognize_google(audio))
      
     except sr.UnknownValueError:
         print("Google Speech Recognition could not understand audio")
      
     except sr.RequestError as e:
         print("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == '__main__':
    main()



















