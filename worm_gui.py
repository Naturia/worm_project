# by HengHuang@2021
from re import T
import pygame
import math
from pygame.locals import *
from brian2 import *
from sympy import bottom_up
from Worm import Worm
from WormNet import WormNet
from myworm import Nacl
import time
from multiprocessing import Process, Queue, Array
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import math

# pg.setConfigOption('background', (188, 210, 230))
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

import collections
import random
import numpy as np
import sys
# from PIL import Image
# from pygame.math import Vector3

from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *


matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg
import pylab

step_size = 0.01
use_defult = False
WINDOW_HEIGHT = 600 * 2
WINDOW_WIDTH = 1080 * 2

# each_par= [0.37771727214567363, 0.9866986402776092, -0.8426217364612967,
#            0.9411787800490856, 1275.30118804425, 1108.3052031230181,
#            0.5361416230443865, 1.2744932672940195, -58.967636961024255,
#            -56.5039853611961, -64.38890231074765, -56.08220000518486,
#            -17.757977620931342, 6.810994091210887, -19.728963274974376,
#            4.0213864971883595, 4.046380661893636, 0.7026210583280772,
#            1.0624693064019084, -55.18996997503564, -62.9006303451024,
#            -56.553218979388475, -64.92072048829868, 3.3629796060267836,
#            -18.90466681215912, 2.9889292339794338, 0.6998673977795988, 3.409669080283493]

# each_par=  [0.9690260360948741, 0.5701910387724638, -0.9212484885938466,
#             0.9794382073450834, 1412.5651386100799, 927.8485512826592,
#             0.5141193028539419, 0.6744308590423316, -64.34486942132935,
#             -55.874782972969115, -64.29088310571387, -58.57985694659874,
#             -16.205403790809214, 7.10071271751076, -21.779596171109006,
#             4.217939774971455, 4.995192727074027, 0.9444623878225684,
#             1.482516119023785, -58.12202509958297, -62.51176098827273,
#             -62.364755279850215, -61.491209864616394, 13.205328077310696,
#             -17.147274262970313, 0.28660339303314686, 2.088111466728151, 4.745317091001198]


each_par = [0.37771727214567363, 0.9866986402776092, -0.8426217364612967,
            0.9411787800490856, 1275.30118804425, 1108.3052031230181,
            0.5361416230443865, 1.2744932672940195, -58.967636961024255,
            -56.5039853611961, -64.38890231074765, -56.08220000518486,
            -17.757977620931342, 6.810994091210887, -19.728963274974376,
            4.0213864971883595, 4.046380661893636, 0.7026210583280772,
            1.0624693064019084, -55.18996997503564, -62.9006303451024,
            -56.553218979388475, -64.92072048829868, 3.3629796060267836,
            -18.90466681215912, 2.9889292339794338, 0.6998673977795988, 3.409669080283493]

if not use_defult:
    worm_body_parameters = {
        "head_ms_SMD_gain": each_par[0],  # 0.4
        "head_ms_RMD_gain": each_par[1],  # 0.6
        "tail_ms_SMD_gain": each_par[0],  # 0.4
        "tail_ms_RMD_gain": each_par[1],  # 0.6
        "vnc_ms_D_gain": each_par[2],
        "vnc_ms_B_gain": each_par[3],
        "vnc_ms_A_gain": each_par[3],
        "vncb_ms_D_gain": each_par[2],
        "vncb_ms_A_gain": each_par[3],
        "vnc_sr_gain": each_par[4],
        "vncb_sr_gain": each_par[4],
        "head_sr_gain": each_par[5],
        "tail_sr_gain": each_par[5],
        "muscle_tau": 0.1,
        "turn_gain": 0.7022261694073677,
        "concentration_th": 0.2,  # 1.5630552610382438,
        "min_turn_time": 1.3548880419693887, }

    head_parameters = {
        "SMDD_cm": each_par[6],
        "SMDV_cm": each_par[6],
        "RMDD_cm": each_par[7],
        "RMDV_cm": each_par[7],

        "SMDD_delta": each_par[8],
        "SMDV_delta": each_par[8],
        "RMDD_delta": each_par[9],
        "RMDV_delta": each_par[9],

        "SMDD_v": each_par[10],
        "SMDV_v": each_par[11],
        "RMDD_v": each_par[11],
        "RMDV_v": each_par[10], }

    head_chemical_parameters = {
        "SMDD_to_SMDV": each_par[12],
        "SMDV_to_SMDD": each_par[12],
        "SMDD_to_RMDV": each_par[13],
        "SMDV_to_RMDD": each_par[13],
        "RMDD_to_RMDV": each_par[14],
        "RMDV_to_RMDD": each_par[14], }

    head_gap_parameters = {
        "SMDD_RMDD": each_par[15],
        "SMDV_RMDV": each_par[15],
        "RMDD_RMDV": each_par[16], }

    tail_parameters = {
        "SMDDB_cm": each_par[6],
        "SMDVB_cm": each_par[6],
        "RMDDB_cm": each_par[7],
        "RMDVB_cm": each_par[7],

        "SMDDB_delta": each_par[8],
        "SMDVB_delta": each_par[8],
        "RMDDB_delta": each_par[9],
        "RMDVB_delta": each_par[9],

        "SMDDB_v": each_par[10],
        "SMDVB_v": each_par[11],
        "RMDDB_v": each_par[11],
        "RMDVB_v": each_par[10], }

    tail_chemical_parameters = {
        "SMDDB_to_SMDVB": each_par[12],
        "SMDVB_to_SMDDB": each_par[12],
        "SMDDB_to_RMDVB": each_par[13],
        "SMDVB_to_RMDDB": each_par[13],
        "RMDDB_to_RMDVB": each_par[14],
        "RMDVB_to_RMDDB": each_par[14], }

    tail_gap_parameters = {
        "SMDDB_RMDDB": each_par[15],
        "SMDVB_RMDVB": each_par[15],
        "RMDDB_RMDVB": each_par[16], }

    vnc_parameters = {
        "VB_cm": each_par[17],
        "DB_cm": each_par[17],
        "VD_cm": each_par[18],
        "DD_cm": each_par[18],

        "VB_delta": each_par[19],
        "DB_delta": each_par[19],
        "VD_delta": each_par[20],
        "DD_delta": each_par[20],

        "VB_v": each_par[21],
        "DB_v": each_par[21],
        "VD_v": each_par[22],
        "DD_v": each_par[22],
    }

    vnc_chemical_parameters = {
        "DB_to_VD": each_par[23],
        "VB_to_DD": each_par[23],
        "DB_to_DD": each_par[24],
        "VB_to_VD": each_par[24],
    }

    vnc_gap_parameters = {
        "DB_DB": each_par[25],
        "VB_VB": each_par[25],
        "DD_DD": each_par[26],
        "VD_VD": each_par[26],
        "AVB_DB": each_par[27],
        "AVB_VB": each_par[27],
    }

    vncb_parameters = {
        "VA_cm": each_par[17],
        "DA_cm": each_par[17],
        "VD_cm": each_par[18],
        "DD_cm": each_par[18],

        "VA_delta": each_par[19],
        "DA_delta": each_par[19],
        "VD_delta": each_par[20],
        "DD_delta": each_par[20],

        "VA_v": each_par[21],
        "DA_v": each_par[21],
        "VD_v": each_par[22],
        "DD_v": each_par[22],
    }

    vncb_chemical_parameters = {
        "DA_to_VD": each_par[23],
        "VA_to_DD": each_par[23],
        "DA_to_DD": each_par[24],
        "VA_to_VD": each_par[24],
    }

    vncb_gap_parameters = {
        "DA_DA": each_par[25],
        "VA_VA": each_par[25],
        "DD_DD": each_par[26],
        "VD_VD": each_par[26],
        "AVA_DA": each_par[27],
        "AVA_VA": each_par[27]}

else:
    worm_body_parameters = None
    head_parameters = None
    head_chemical_parameters = None
    head_gap_parameters = None
    tail_parameters = None
    tail_chemical_parameters = None
    tail_gap_parameters = None
    vnc_parameters = None
    vnc_chemical_parameters = None
    vnc_gap_parameters = None
    vncb_parameters = None
    vncb_chemical_parameters = None
    vncb_gap_parameters = None

use_defult_klinotaxis = False
each_par_klinotaxis = [1.392676408169791, 1.1696825344115496, 0.5365449015516788, 1.0949907049071044,
                       0.5165577421430498, 1.3565628617070615, -57.59759918320924, -55.60414323583245,
                       -63.84581831516698, -61.72273434465751, -55.38804127601907, -63.86664470192045,
                       0.12203647168353202, 0.6440665861591697, 1.356307763163932, 0.8434836344560609,
                       -29.14954416686669, -149.81658950680867, 152.2260981053114, 174.1984326345846,
                       187.64023459982127, 86.29761338233948, -275.3541210805997, 56.735633541829884, 7.724542827345431,
                       9.634364016354084, 0.7022261694073677, 1.5630552610382438, 1.3548880419693887]

if not use_defult_klinotaxis:
    klinotaxis_parameters = {
        "AIYL_cm": each_par_klinotaxis[0],
        "AIYR_cm": each_par_klinotaxis[1],
        "AIZL_cm": each_par_klinotaxis[2],
        "AIZR_cm": each_par_klinotaxis[3],
        "SMBV_cm": each_par_klinotaxis[4],
        "SMBD_cm": each_par_klinotaxis[5],

        "AIYL_delta": each_par_klinotaxis[6],
        "AIYR_delta": each_par_klinotaxis[7],
        "AIZL_delta": each_par_klinotaxis[8],
        "AIZR_delta": each_par_klinotaxis[9],
        "SMBV_delta": each_par_klinotaxis[10],
        "SMBD_delta": each_par_klinotaxis[11],

        "AIYL_v": -72,
        "AIYR_v": -72,
        "AIZL_v": -72,
        "AIZR_v": -72,
        "SMBV_v": 0,
        "SMBD_v": 0,

        "ASEL_N": each_par_klinotaxis[12],
        "ASEL_M": each_par_klinotaxis[13],
        "ASEL_v": 0.0,

        "ASER_N": each_par_klinotaxis[14],
        "ASER_M": each_par_klinotaxis[15],
        "ASER_v": 0.0,
    }

    klinotaxis_chemical_parameters = {
        "ASEL_to_AIYL": each_par_klinotaxis[16],
        "ASEL_to_AIYR": each_par_klinotaxis[17],
        "ASER_to_AIYL": each_par_klinotaxis[18],
        "ASER_to_AIYR": each_par_klinotaxis[19],

        "AIYL_to_AIZL": each_par_klinotaxis[20],
        "AIYR_to_AIZR": each_par_klinotaxis[21],
        "AIZL_to_SMBV": each_par_klinotaxis[22],  # -20.0,
        "AIZR_to_SMBD": each_par_klinotaxis[23],  # 20.0,
    }

    klinotaxis_gap_parameters = {
        "AIYL_AIYR": each_par_klinotaxis[24],
        "AIZL_AIZR": each_par_klinotaxis[25],
    }
else:
    klinotaxis_parameters = None
    klinotaxis_chemical_parameters = None
    klinotaxis_gap_parameters = None


def rectangle(locations):
    location_x = [location[0][0] for location in locations] + [location[1][0] for location in locations]
    location_y = [location[0][1] for location in locations] + [location[1][1] for location in locations]

    x_min = min(location_x)
    x_max = max(location_x)
    y_min = min(location_y)
    y_max = max(location_y)

    top_left = [x_min, y_min]
    bottom_right = [x_max, y_max]
    return top_left, bottom_right


def height(x, y):
    # return np.exp(-(x - 2) ** 2) + 1.2 * np.exp(-x ** 2 - y ** 2)
    return np.exp(-(x - 2) ** 2 - (y - 2) ** 2) + 1.2 * np.exp(-x ** 2 - y ** 2)
    # return ((y-0.1) ** 5 + (x+0.4) ** 3) * 5 * np.exp(-x ** 2 - y ** 2)


def drawGrid():
    blockSize = 20  # Set the size of the grid block
    for x in range(WINDOW_WIDTH):
        for y in range(WINDOW_HEIGHT):
            rect = pygame.Rect(x * blockSize, y * blockSize, blockSize, blockSize)
            pygame.draw.rect(SCREEN, WHITE, rect, 1)
    pygame.draw.lines()


def gen_colors(N):
    values = [int(i * 250 / N) for i in range(N)]
    # print(values)
    colors = ["#%02x%02x%02x" % (200, int(g), 40) for g in values]  # 250 250 250 ,g值越小越靠近0红色
    return colors


def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    return r, g, b


def update(self):
    self.speedx = 0
    keystate = pygame.key.get_pressed()
    # 按左方向键时横坐标减小，按右方向键时横坐标增大
    if keystate[pygame.K_LEFT]:
        self.speedx -= 10
    if keystate[pygame.K_RIGHT]:
        self.speedx += 10
    self.rect.x += self.speedx
    # 阻止角色跑出左右边界
    if self.rect.right > WIDTH:
        self.rect.right = WIDTH
    if self.rect.left < 0:
        self.rect.left = 0


def run_gui():
    # worm_net.run(200*ms)
    trace = []
    trace_step = 30
    # print(nacl.get_concentration_pool())
    size = width, height = 1960, 1080

    pygame.init()
    pygame.display.set_caption("c.elegans")
    infoObject = pygame.display.Info()
    # screen = pygame.display.set_mode((infoObject.current_w,infoObject.current_h)) #depth=32，flags=pygame.FULLSCREEN

    # screen = pygame.display.set_mode((1800*2, 1200*2)) #1800,1200

    screen = pygame.display.set_mode((600 * 2, 1080 * 2))  # 1800*2,1200*2

    # # set temperature area
    # a = np.arange(-1.5, 3.5, 0.01)
    # b = np.arange(-1.5, 3.5, 0.01)
    #
    # X, Y = np.meshgrid(a, b)
    # Z = height(X, Y)
    #
    # # N=np.arange(-0.2,1.5,0.1)
    # N = np.arange(-0.2, 1.5, 0.01)
    # #CS = plt.contourf(Z, N, linewidth=2, cmap=mpl.cm.jet)
    #
    # #fig, axes = plt.contourf(Z, N, linewidth=2, cmap=mpl.cm.jet)
    # fig=plt.contourf(X,Y,Z)
    # #axes.plot([1, 2], [1, 2], color='green', label='test')
    #
    # fig.canvas.draw()
    #
    # fig = pylab.figure(CS)
    # #
    # canvas = agg.FigureCanvasAgg(fig)
    # canvas.draw()
    # renderer = canvas.get_renderer()
    # raw_data = renderer.tostring_rgb()

    game_clock = pygame.time.Clock()

    # Variable to keep our main loop running
    running = True
    step = 0
    # Our main loop!

    button_clicked_img1 = False
    button_clicked_img2 = False
    button_clicked_img3 = False
    button_clicked_img4 = False
    button_clicked_img5 = False
    button_clicked_img6 = False

    #background = pygame.image.load('./blue2.jpeg').convert_alpha()

    while running:
        step += 1
        # 绘制线条
        # screen.fill((0,0,0))
        # screen.fill((188, 210, 230)) #  blue
        screen.fill((255, 255, 250))
        # screen.blit(pygame.transform.scale(background, size), (0, 0))
        # pygame.display.flip()
        # screen.fill((230, 230, 230))
        # size = canvas.get_width_height()
        #
        # surf = pygame.image.fromstring(raw_data, size, "RGB")
        # screen.blit(surf, (0, 0))
        # pygame.display.flip()

        # insert img
        font = pygame.font.SysFont("Arial", 35)
        color_text = (200, 200, 200)

        text_clicked_img1 = font.render("Clicked", True, color_text)
        text_clicked_img2 = font.render("Clicked", True, color_text)
        text_clicked_img3 = font.render("Clicked", True, color_text)
        text_clicked_img4 = font.render("Clicked", True, color_text)
        text_clicked_img5 = font.render("Clicked", True, color_text)
        text_clicked_img6 = font.render("Clicked", True, color_text)

        # text = font.render("Button", True, color_text)
        # text_rect = text_clicked.get_rect(center=(370, 10))

        img1 = pygame.image.load("./img/gentle_touch.png").convert_alpha()
        img1 = pygame.transform.scale(img1, (50, 50))
        screen.blit(img1, [10, 10])
        img2 = pygame.image.load("./img/harsh_tou.jpeg").convert()
        img2 = pygame.transform.scale(img2, (50, 50))
        img2_1 = pygame.image.load("./img/touch.png").convert()
        img2_1 = pygame.transform.scale(img2_1, (50, 50))
        # screen.blit(img2, [70, 10])
        if button_clicked_img2 == True:
            screen.blit(img2_1, [70, 10])
        else:
            screen.blit(img2, [70, 10])
        img3 = pygame.image.load("./img/nacl3.png").convert_alpha()
        img3 = pygame.transform.scale(img3, (50, 50))
        img3_1 = pygame.image.load("./img/nacl4.png").convert_alpha()
        img3_1 = pygame.transform.scale(img3_1, (50, 50))
        # screen.blit(img3, [130, 10])
        if button_clicked_img3 == True:
            screen.blit(img3_1, [130, 10])
        else:
            screen.blit(img3, [130, 10])
        img4 = pygame.image.load("./img/poison2.png").convert_alpha()
        img4 = pygame.transform.scale(img4, (50, 50))
        screen.blit(img4, [190, 10])
        img5 = pygame.image.load("./img/warm2.png").convert_alpha()
        img5 = pygame.transform.scale(img5, (50, 50))
        screen.blit(img5, [250, 10])
        img6 = pygame.image.load("./img/neuron.png").convert()
        img6 = pygame.transform.scale(img6, (50, 50))
        img6_1 = pygame.image.load("./img/worm2.png").convert()
        img6_1 = pygame.transform.scale(img6_1, (50, 50))
        # screen.blit(img6, [310, 10])
        if button_clicked_img6 == True:
            screen.blit(img6_1, [310, 10])
        else:
            screen.blit(img6, [310, 10])
        # pygame.display.flip()

        text_rect_img1 = pygame.Rect(10, 10, 50, 50)
        text_rect_img2 = pygame.Rect(70, 10, 50, 50)
        # text_rect_img3 = text_clicked_img1.get_rect(130, 10,50,50)
        text_rect_img3 = pygame.Rect(130, 10, 50, 50)
        text_rect_img4 = pygame.Rect(190, 10, 50, 50)
        text_rect_img5 = pygame.Rect(250, 10, 50, 50)
        text_rect_img6 = pygame.Rect(310, 10, 50, 50)

        # # Draw neurons activition 1
        # # sensory neurons
        # pygame.draw.circle(screen, (255, 215, 0),(20, 110), 10, 2)
        # pygame.draw.circle(screen, (255, 215, 0),(50, 110), 10, 2)
        # pygame.draw.circle(screen, (255, 215, 0),(80, 110), 10, 2)
        # pygame.draw.circle(screen, (255, 215, 0),(110, 110), 10,2)
        # pygame.draw.circle(screen, (255, 215, 0),(150, 110), 10, 2)
        # pygame.draw.circle(screen, (255, 215, 0),(180, 110), 10, 2)
        #
        # #interneurons
        # pygame.draw.circle(screen,  (47, 79, 79), (30, 140), 10, 2)
        # pygame.draw.circle(screen,  (47, 79, 79), (60, 140), 10, 2)
        # pygame.draw.circle(screen,  (47, 79, 79), (30, 170), 10, 2)
        # pygame.draw.circle(screen,  (47, 79, 79), (60, 170), 10, 2)
        # pygame.draw.circle(screen,  (47, 79, 79), (100, 140), 10, 2)
        # pygame.draw.circle(screen,  (47, 79, 79), (130, 140), 10, 2)
        # pygame.draw.circle(screen,  (47, 79, 79), (110, 170), 10, 2)
        # pygame.draw.circle(screen,  (47, 79, 79), (160, 140), 10, 2)
        # pygame.draw.circle(screen, (47, 79, 79), (160, 170), 10, 2)
        #
        # #motor neurons
        #
        # pygame.draw.circle(screen, (188, 210, 230), (20, 200), 10, 2)
        # pygame.draw.circle(screen, (188, 210, 230), (20, 230), 10, 2)
        # pygame.draw.circle(screen, (188, 210, 230), (60, 200), 10, 2)
        # pygame.draw.circle(screen, (188, 210, 230), (60, 230), 10, 2)
        # pygame.draw.circle(screen, (188, 210, 230), (100, 200), 10, 2)
        # pygame.draw.circle(screen, (188, 210, 230), (100, 230), 10, 2)
        # pygame.draw.circle(screen, (188, 210, 230), (140, 200), 10, 2)
        # pygame.draw.circle(screen, (188, 210, 230), (140, 230), 10, 2)
        # pygame.draw.circle(screen, (188, 210, 230), (180, 200), 10, 2)
        # pygame.draw.circle(screen, (188, 210, 230), (180, 230), 10, 2)

        # Draw neurons activition 2
        # sensory neurons

        x0 = 1080
        y0 = 120
        r = 100
        font1 = pygame.font.SysFont('arial', 5)
        # screen.blit(text, (500, 300))
        text=['PLM','PVM','ALM','AVM','AFD','AWC','ASE','AIY','AIZ','RIA','SMDD','SMDV','RMDD','RMDV','SMBD','SMBV','AVB','PVC','AVA','DB','VB','VD','DD','VA','DA']

        state_dict = state_queue.get()
        print("state_dict", state_dict)

        for i in range(1,26):
            theta=((i-1)*14.4*math.pi)/180
            x1=x0+r*math.cos(theta)
            y1=y0+r*math.sin(theta)

            if i < 10:
                pygame.draw.circle(screen, (255, 215, 0), (x1,y1), 10, 2)
            elif i>=10 and i<20:
                pygame.draw.circle(screen, (47, 79, 79), (x1, y1), 10, 2)
            else:
                pygame.draw.circle(screen, (0, 255, 255), (x1, y1), 10, 2)
            neuron_text = font1.render(text[i-1], True, (0, 0, 0))
            screen.blit(neuron_text, (x1-5,y1-3))

        if state_dict['SMDD'] >= -0.02:
            pygame.draw.circle(screen, (188, 210, 230),(x0+r*math.cos(((11-1)*14.4*math.pi)/180),y0 + r * math.sin(((11-1)*14.4*math.pi)/180)), 10, 10)
        if state_dict['SMDV'] >= -0.02:
            pygame.draw.circle(screen, (188, 210, 230), (x0+r*math.cos(((12-1)*14.4*math.pi)/180),y0 + r * math.sin(((12-1)*14.4*math.pi)/180)), 10, 10)
        if state_dict['RMDD'] >= -0.04:
            pygame.draw.circle(screen, (188, 210, 230), (x0+r*math.cos(((13-1)*14.4*math.pi)/180),y0 + r * math.sin(((13-1)*14.4*math.pi)/180)), 10, 10)
        if state_dict['RMDV'] >= -0.04:
            pygame.draw.circle(screen, (188, 210, 230), (x0+r*math.cos(((14-1)*14.4*math.pi)/180),y0 + r * math.sin(((14-1)*14.4*math.pi)/180)), 10, 10)
        if state_dict['ASEL'] != 0:
            pygame.draw.circle(screen, (188, 210, 230),(x0+r*math.cos(((7-1)*14.4*math.pi)/180),y0 + r * math.sin(((7-1)*14.4*math.pi)/180)), 10, 10)
        if state_dict['ASER'] != 0:
            pygame.draw.circle(screen, (188, 210, 230), (x0+r*math.cos(((7-1)*14.4*math.pi)/180),y0 + r * math.sin(((7-1)*14.4*math.pi)/180)), 10, 10)
        if state_dict['AVM'] != 0:
            pygame.draw.circle(screen, (188, 210, 230), (x0+r*math.cos(((4-1)*14.4*math.pi)/180),y0 + r * math.sin(((4-1)*14.4*math.pi)/180)), 10, 10)
        if state_dict['PLM'] != 0:
            pygame.draw.circle(screen, (188, 210, 230), (x0+r*math.cos(((1-1)*14.4*math.pi)/180),y0 + r * math.sin(((1-1)*14.4*math.pi)/180)), 10, 10)

        # # 画直线，起点坐标为start_pos,终点坐标为end_pos取值为二元组，width是粗细程度
        pygame.draw.aaline(screen, (47, 79, 79), (x0+r*math.cos(((1-1)*14.4*math.pi)/180),y0 + r * math.sin(((1-1)*14.4*math.pi)/180)), (x0+r*math.cos(((17-1)*14.4*math.pi)/180),y0 + r * math.sin(((17-1)*14.4*math.pi)/180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (x0+r*math.cos(((1-1)*14.4*math.pi)/180),y0 + r * math.sin(((1-1)*14.4*math.pi)/180)), (x0+r*math.cos(((18-1)*14.4*math.pi)/180),y0 + r * math.sin(((18-1)*14.4*math.pi)/180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (x0+r*math.cos(((2-1)*14.4*math.pi)/180),y0 + r * math.sin(((2-1)*14.4*math.pi)/180)), (x0+r*math.cos(((17-1)*14.4*math.pi)/180),y0 + r * math.sin(((17-1)*14.4*math.pi)/180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (x0+r*math.cos(((2-1)*14.4*math.pi)/180),y0 + r * math.sin(((2-1)*14.4*math.pi)/180)), (x0+r*math.cos(((18-1)*14.4*math.pi)/180),y0 + r * math.sin(((18-1)*14.4*math.pi)/180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((3 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((3 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((19 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((19 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((4 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((4 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((19 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((19 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((5- 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((5 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((8 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((8 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((6 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((6- 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((8 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((8 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((7 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((7 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((8 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((8 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((8- 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((8- 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((9 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((9 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((8- 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((8 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((10 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((10 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((9 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((9- 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((10 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((10 - 1) * 14.4 * math.pi) / 180)), 10)

        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((10 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((10 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((15 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((15 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((10 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((10 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((16 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((16 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((11 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((11 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((13 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((13 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((11 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((11 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((14 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((14 - 1) * 14.4 * math.pi) / 180)), 10)

        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((12 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((12 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((13 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((13 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((12 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((12 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((14 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((14 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((11 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((11 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((17 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((17 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((20 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((20 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((22 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((22 - 1) * 14.4 * math.pi) / 180)), 10)

        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((20 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((20 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((22 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((22 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((21 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((21 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((23 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((23- 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((21 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((21 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((23 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((23 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((22 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((22 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((24 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((24- 1) * 14.4 * math.pi) / 180)), 10)

        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((23- 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((23 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((25 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((25 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((17 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((17 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((23 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((23 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((18 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((18 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((21 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((21 - 1) * 14.4 * math.pi) / 180)), 10)
        pygame.draw.aaline(screen, (47, 79, 79), (
        x0 + r * math.cos(((19 - 1) * 14.4 * math.pi) / 180), y0 + r * math.sin(((19 - 1) * 14.4 * math.pi) / 180)), (
                           x0 + r * math.cos(((24 - 1) * 14.4 * math.pi) / 180),
                           y0 + r * math.sin(((24 - 1) * 14.4 * math.pi) / 180)), 10)


        rods_locations, lateral_locations, diagonal_locations = queue.get()
        # start_time = time.time()

        first_rod = rods_locations[0]
        last_rod = rods_locations[-1]
        head_location = ((first_rod[0][0] + first_rod[1][0]) / 2, (first_rod[0][1] + first_rod[1][1]) / 2)
        tail_location = ((last_rod[0][0] + last_rod[1][0]) / 2, (last_rod[0][1] + last_rod[1][1]) / 2)

        # print("first_rod,last_rod,head_location,tail_location",first_rod,last_rod,head_location,tail_location)
        # infoObject.current_w, infoObject.current_h

        # print("first_rod[0][0]",first_rod[0][0])

        # bound processing
        if first_rod[0][0] == 0:
            touch_signal[0] = 1
            # first_rod[0][0] = 0
        if first_rod[0][1] >= infoObject.current_h:
            touch_signal[0] = 1
            # first_rod[0][1] = infoObject.current_h
        if last_rod[1][0] >= infoObject.current_w:
            touch_signal[1] = 1
            # last_rod[1][0] = infoObject.current_w
        if last_rod[1][1] >= infoObject.current_h:
            touch_signal[1] = 1
            # last_rod[1][1] = infoObject.current_h

        # if distance_to_head < distance_to_tail:
        #     touch_signal[0] = 1
        #     print("touch worm head at:", (pos[0] - shift[0]) / scale, (pos[1] - shift[1]) / scale)
        # else:
        #     touch_signal[1] = 1
        #     print("touch worm tail at:", (pos[0] - shift[0]) / scale, (pos[1] - shift[1]) / scale)

        if step % trace_step == 0:
            trace.append(rods_locations[-1][0])

        # print("rods_locations",rods_locations)
        # print("lateral_locations", lateral_locations)
        # print("diagonal_locations", diagonal_locations)

        # print(rods_locations)

        if button_clicked_img6 == False:
            for location in rods_locations:  # cut
                # pygame.draw.aaline(screen, (188, 210, 230), location[0], location[1], 2)
                pygame.draw.aaline(screen, (255, 215, 0), location[0], location[1], 2)
                # pygame.draw.aaline(screen, (255, 215, 0), (location[0][0], location[0][1] + 5),(location[1][0], location[1][1] + 5), 2)
                # pygame.draw.ellipse(screen, (255, 182, 193), [location[0][0] - 1, location[0][1] - 1, 15, 5], 0)
                # pygame.draw.ellipse(screen, (255, 182, 193), [location[1][0] - 1, location[1][1] + 1, 15, 5], 0)
                # pygame.draw.line(screen, (255, 130, 71), location[0], location[1], 2)
            for location in lateral_locations:  # body
                pygame.draw.aaline(screen, (47, 79, 79), location[0], location[1], 10)
                # pygame.draw.aaline(screen, (47, 79, 79), location[0], location[1], 10)
                # 238, 99, 99
                # screen.fill((255, 0, 0), (location[0][0], location[1][0], location[0][1], location[1][1]))
                # pygame.draw.aaline(screen,(238,99,99),location[0],location[1], 2)
                # pygame.draw.aaline(screen, (255, 215, 0), (location[0][0],location[0][1]+5), (location[1][0],location[1][1]+5), 2)
                # pygame.draw.aaline(screen, (173,255,47), (location[0][0], location[0][1] + 10), (location[1][0], location[1][1] + 10), 2)
                # print("location[0],location[1]",location[0],location[1])
            for i in range(len(lateral_locations) - 1):
                # print("len(lateral_locations)",len(lateral_locations))
                # 169,169,169
                # 188,143,143
                # 192,192,192
                pygame.draw.polygon(screen, (160, 160, 160), [(lateral_locations[i][0][0], lateral_locations[i][0][1]),
                                                              (lateral_locations[i + 1][1][0],
                                                               lateral_locations[i + 1][1][1]),
                                                              (lateral_locations[i + 1][1][0],
                                                               lateral_locations[i + 1][1][1]),
                                                              (lateral_locations[i][1][0], lateral_locations[i][1][1])],
                                    0)
            for location in diagonal_locations:  # diagonal
                # pygame.draw.aaline(screen,(188, 210, 230),location[0],location[1], 2)
                # pygame.draw.line(screen,(30,144,255),location[0],location[1], 2)
                # 245,222,179
                # pygame.draw.line(screen, (105,105,105), location[0], location[1], 2)
                pygame.draw.line(screen, (160, 160, 160), location[0], location[1], 2)
        else:
            # if state_dict['SMDD'] >= -0.02:
            #     pygame.draw.ellipse(screen, (124,252,0),[rods_locations[3][0][0] - 1, rods_locations[3][0][1] + 1, 10, 5], 0)
            # if state_dict['SMDV'] >= -0.02:
            #     pygame.draw.ellipse(screen, (124,252,0),[rods_locations[3][0][0] - 1, rods_locations[3][0][1] + 3, 10, 5], 0)
            # if state_dict['RMDD'] >= -0.04:
            #     pygame.draw.ellipse(screen, (124,252,0),[rods_locations[5][0][0] - 1, rods_locations[5][0][1] + 1, 10, 5], 0)
            # if state_dict['RMDV'] >= -0.04:
            #     pygame.draw.ellipse(screen, (124,252,0),[rods_locations[5][0][0] - 1, rods_locations[5][0][1] + 3, 10, 5], 0)
            # if state_dict['ASEL'] != 0:
            #     pygame.draw.ellipse(screen, (124,252,0),[rods_locations[7][0][0] - 1, rods_locations[7][0][1] + 1, 10, 5], 0)
            # if state_dict['ASER'] != 0:
            #     pygame.draw.ellipse(screen, (124,252,0),[rods_locations[8][0][0] - 1, rods_locations[8][0][1] + 1, 10, 5], 0)
            # if state_dict['AVM'] != 0:
            #     pygame.draw.ellipse(screen, (124,252,0),[rods_locations[9][0][0] - 1, rods_locations[9][0][1] + 1, 10, 5], 0)
            # if state_dict['PLM'] != 0:
            #     pygame.draw.ellipse(screen, (124,252,0),[rods_locations[10][0][0] - 1, rods_locations[10][0][1] + 1, 10, 5], 0)
            for i in range(len(rods_locations)):  # cut
                #print("location", rods_locations[i])
                if i >= 2 and i % 5 == 0:
                    pygame.draw.aaline(screen, (0, 0, 205), rods_locations[i][0], rods_locations[i][1], 2)
                    pygame.draw.ellipse(screen, (153, 50, 204),
                                        [rods_locations[i][0][0] - 1, rods_locations[i][0][1] + 1, 10, 5], 0)
                    # pygame.draw.aaline(screen,(255, 215, 0),location[0],location[1], 2)
                    pygame.draw.ellipse(screen, (240, 128, 128),
                                        [rods_locations[i][0][0] - 1, rods_locations[i][0][1] + 10, 10, 5], 0)
                if i >= 2 and i % 3 == 0:
                    pygame.draw.aaline(screen, (188, 210, 230), rods_locations[i][0], rods_locations[i][1], 2)
                if i >= 2 and i % 20 == 0:
                    pygame.draw.ellipse(screen, (255, 215, 0),
                                        [rods_locations[i][0][0] - 1, rods_locations[i][0][1] + 5, 10, 5], 0)
                if i >= 2 and i % 25 == 0:
                    pygame.draw.ellipse(screen, (255, 215, 0),
                                        [rods_locations[i][0][0] + 5, rods_locations[i][0][1] + 5, 10, 5], 0)
                if i >= 2 and i < 6:
                    pygame.draw.ellipse(screen, (222, 184, 135),
                                        [rods_locations[i][0][0], rods_locations[i][0][1] + 5, 10, 5], 0)
                    pygame.draw.ellipse(screen, (222, 184, 135),
                                        [rods_locations[i][0][0], rods_locations[i][0][1] + 7, 10, 5], 0)
                    pygame.draw.ellipse(screen, (222, 184, 135),
                                        [rods_locations[i][0][0], rods_locations[i][0][1] + 9, 10, 5], 0)

                # pygame.draw.ellipse(screen, (255, 182, 193), [location[0][0] - 1, location[0][1] - 1, 15, 5], 0)
                # pygame.draw.ellipse(screen, (255, 182, 193), [location[1][0] - 1, location[1][1] + 1, 15, 5], 0)
                # pygame.draw.line(screen, (255, 130, 71), location[0], location[1], 2)

            for i in range(len(lateral_locations)):
                if i % 2 == 0:
                    pygame.draw.aaline(screen, (0, 0, 205), lateral_locations[i][0], lateral_locations[i][1], 5)
                    if i <= 6:
                        pygame.draw.aaline(screen, (106, 90, 205),
                                           (lateral_locations[i][0][0], lateral_locations[i][0][1]),
                                           (lateral_locations[i][1][0], lateral_locations[i][1][1]), 2)
                    else:
                        pygame.draw.aaline(screen, (106, 90, 205),
                                           (lateral_locations[i][0][0], lateral_locations[i][0][1] + 10),
                                           (lateral_locations[i][1][0], lateral_locations[i][1][1] + 10), 2)
                else:
                    pygame.draw.aaline(screen, (153, 50, 204), lateral_locations[i][0], lateral_locations[i][1], 5)
                    if i <= 6:
                        pygame.draw.aaline(screen, (95, 158, 160),
                                           (lateral_locations[i][0][0], lateral_locations[i][0][1]),
                                           (lateral_locations[i][1][0], lateral_locations[i][1][1]), 2)
                    else:
                        pygame.draw.aaline(screen, (95, 158, 160),
                                           (lateral_locations[i][0][0], lateral_locations[i][0][1] - 10),
                                           (lateral_locations[i][1][0], lateral_locations[i][1][1] - 10), 2)
                # pygame.draw.aaline(screen, (255, 215, 0), (location[0][0],location[0][1]+5), (location[1][0],location[1][1]+5), 2)
                # pygame.draw.aaline(screen, (173,255,47), (location[0][0], location[0][1] + 10), (location[1][0], location[1][1] + 10), 2)
                # print("location[0],location[1]",location[0],location[1])
            for location in diagonal_locations:  # diagonal
                pygame.draw.aaline(screen, (188, 210, 230), location[0], location[1], 2)
                # pygame.draw.line(screen,(30,144,255),location[0],location[1], 2)

        if button_clicked_img2 == True:
            top_left, bottom_right = rectangle(rods_locations)
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(top_left[0], top_left[1], bottom_right[0] - top_left[0],
                                                              bottom_right[1] - top_left[1]), 3)

        # check mouse click
        # left, middle, right = pygame.mouse.get_pressed()

        # if left:
        # print("Left Mouse Key is being pressed")

        for t in trace:
            pygame.draw.circle(screen, (255, 0, 0), (t[0], t[1]), 2)

        xy_pos = nacl_location
        # print("xy_pos",xy_pos[0] * scale + shift[0],xy_pos[1] * scale + shift[1])
        colors = gen_colors(6)

        if button_clicked_img3 == True and (xy_pos[0] * scale + shift[0] > 310 and xy_pos[1] * scale + shift[1] > 100):
            pygame.draw.circle(screen, (Hex_to_RGB(colors[5])),
                               (xy_pos[0] * scale + shift[0], xy_pos[1] * scale + shift[1]), r + 100)
            pygame.draw.circle(screen, (Hex_to_RGB(colors[4])),
                               (xy_pos[0] * scale + shift[0], xy_pos[1] * scale + shift[1]), r + 60)
            pygame.draw.circle(screen, (Hex_to_RGB(colors[3])),
                               (xy_pos[0] * scale + shift[0], xy_pos[1] * scale + shift[1]), r + 35)
            pygame.draw.circle(screen, (Hex_to_RGB(colors[2])),
                               (xy_pos[0] * scale + shift[0], xy_pos[1] * scale + shift[1]), r + 20)
            pygame.draw.circle(screen, (Hex_to_RGB(colors[1])),
                               (xy_pos[0] * scale + shift[0], xy_pos[1] * scale + shift[1]), r + 10)
            pygame.draw.circle(screen, (Hex_to_RGB(colors[0])),
                               (xy_pos[0] * scale + shift[0], xy_pos[1] * scale + shift[1]), r + 5)

        # pygame.draw.circle(screen, (50, 0, 0), (xy_pos[0] * scale + shift[0], xy_pos[1] * scale + shift[1]),r + 100 + 100)
        # pygame.draw.circle(screen, (100, 0, 0), (xy_pos[0] * scale + shift[0], xy_pos[1] * scale + shift[1]), r + 100)
        # pygame.draw.circle(screen, (255, 0, 0), (xy_pos[0] * scale + shift[0], xy_pos[1] * scale + shift[1]), r)

        # for x,y,concentration in nacl.get_concentration_pool():
        # pygame.Surface.set_at(screen,(round(x*scale+shift[0]), round(y*scale+shift[1])), (round(255*(concentration/nacl.get_max_concentration())),0,0))
        # print(concentration)

        pygame.display.update()
        # for loop through the event queue
        for event in pygame.event.get():
            # Check for KEYDOWN event; KEYDOWN is a constant defined in pygame.locals, which we imported earlier
            if event.type == KEYDOWN:
                # If the Esc key has been pressed set running to false to exit the main loop
                if event.key == K_ESCAPE:
                    running = False
            # Check for QUIT event; if QUIT, set running to false
            elif event.type == QUIT:
                running = False
            # handle MOUSEBUTTONUP
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # button_clicked_img2 = True if text_rect_img2.collidepoint(event.pos) else False
                # button_clicked_img3 = True if text_rect_img3.collidepoint(event.pos) else False
                left, middle, right = pygame.mouse.get_pressed()
                #print("left, middle, right", left, middle, right)
                pos = pygame.mouse.get_pos()
                # print("event.pos",event.pos)
                # print("button_clicked_img3",button_clicked_img3)
                # print("button_clicked_img2",button_clicked_img2)
                if text_rect_img6.collidepoint(event.pos):
                    button_clicked_img6 = not button_clicked_img6
                    #print("button_clicked_img6", button_clicked_img6)

                if text_rect_img3.collidepoint(event.pos):
                    button_clicked_img3 = not button_clicked_img3
                    #print("button_clicked_img3", button_clicked_img3)

                if button_clicked_img3 == True and left:
                    #print("Left Mouse Key is being pressed")
                    #print("pos", pos)
                    #print((pos[0] - shift[0]) / scale, (pos[1] - shift[1]) / scale)
                    nacl_location[0] = (pos[0] - shift[0]) / scale
                    nacl_location[1] = (pos[1] - shift[1]) / scale
                    #print(nacl_location[0], nacl_location[1])

                if text_rect_img2.collidepoint(event.pos):
                    button_clicked_img2 = not button_clicked_img2
                    #print("button_clicked_img2", button_clicked_img2)

                if button_clicked_img2 == True and right:
                    #print("Right Mouse Key is being pressed")
                    distance_to_head = (head_location[0] - pos[0]) ** 2 + (head_location[1] - pos[1]) ** 2
                    distance_to_tail = (tail_location[0] - pos[0]) ** 2 + (tail_location[1] - pos[1]) ** 2
                    if distance_to_head < distance_to_tail:
                        touch_signal[0] = 1
                        #print("touch worm head at:", (pos[0] - shift[0]) / scale, (pos[1] - shift[1]) / scale)
                    else:
                        touch_signal[1] = 1
                        #print("touch worm tail at:", (pos[0] - shift[0]) / scale, (pos[1] - shift[1]) / scale)

                    # print(nacl_location[0],nacl_location[1])

        # for event in pygame.event.get():
        #     # Check for KEYDOWN event; KEYDOWN is a constant defined in pygame.locals, which we imported earlier
        #     if event.type == KEYDOWN:
        #         # If the Esc key has been pressed set running to false to exit the main loop
        #         if event.key == K_ESCAPE:
        #             running = False
        #     # Check for QUIT event; if QUIT, set running to false
        #     elif event.type == QUIT:
        #         running = False
        #     # handle MOUSEBUTTONUP
        #     elif event.type == pygame.MOUSEBUTTONDOWN:
        #         left, middle, right = pygame.mouse.get_pressed()
        #         pos = pygame.mouse.get_pos()
        #         if left:
        #             # print(pos)
        #             print((pos[0]-shift[0])/scale,(pos[1]-shift[1])/scale)
        #             nacl_location[0] = (pos[0]-shift[0])/scale
        #             nacl_location[1] = (pos[1]-shift[1])/scale
        #         elif right:
        #             distance_to_head = (head_location[0]-pos[0])**2 + (head_location[1]-pos[1])**2
        #             distance_to_tail = (tail_location[0]-pos[0])**2 + (tail_location[1]-pos[1])**2
        #             if distance_to_head < distance_to_tail:
        #                 touch_signal[0] = 1
        #                 print("touch worm head at:",(pos[0]-shift[0])/scale,(pos[1]-shift[1])/scale)
        #             else:
        #                 touch_signal[1] = 1
        #                 print("touch worm tail at:",(pos[0]-shift[0])/scale,(pos[1]-shift[1])/scale)
        #
        #             # print(nacl_location[0],nacl_location[1])

        game_clock.tick(1000)

        # end_time = time.time()
        # print("draw time:",end_time - start_time)


# def plot_state():
#     plt.ion()
#     plt.figure(1)
#     time = [ ]
#     value = []
#     while True:
#         plt.clf()
#         plt.plot(time,value)
#         # fig.canvas.draw()
#         state_dict = state_queue.get()
#         time.append(state_dict["time"])
#         value.append(state_dict["SMDD"])
#         plt.pause(0.0000001)
#         plt.ioff()

# class WormPlotter():

#     def __init__(self, sampleinterval=0.01, timewindow=10., size=(1000,600)):
#         # Data stuff
#         self._interval = int(sampleinterval*1000)
#         self._bufsize = int(timewindow/sampleinterval)
#         self.databuffer_time = collections.deque([0.0]*self._bufsize, self._bufsize)
#         self.databuffer_y= collections.deque([0.0]*self._bufsize, self._bufsize)
#         self.x = np.zeros(self._bufsize, dtype=np.float) #linspace(-timewindow, 0.0, self._bufsize)
#         self.y = np.zeros(self._bufsize, dtype=np.float)
#         # PyQtGraph stuff
#         self.app = QtGui.QApplication([])
#         pg.setConfigOptions(antialias=True)
#         self.plt = pg.plot(title='c.elegans states')
#         self.plt.resize(*size)
#         self.plt.showGrid(x=True, y=True)
#         self.plt.setLabel('left', 'amplitude', 'V')
#         self.plt.setLabel('bottom', 'time', 's')
#         self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))
#         # QTimer
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.updateplot)
#         self.timer.start(self._interval)

#     def updateplot(self):
#         state_dict = state_queue.get()
#         self.databuffer_y.append(state_dict["SMDV"])
#         self.databuffer_time.append(state_dict["time"])

#         self.x[:] = self.databuffer_time
#         self.y[:] = self.databuffer_y
#         self.curve.setData(self.x, self.y)
#         self.app.processEvents()

#     def run(self):
#         self.app.exec_()

def run_worm3d():
    pygame.init()
    viewport = (800, 600)
    hx = viewport[0] / 2
    hy = viewport[1] / 2
    srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

    glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
    # glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (0.9, 0.9, 0.9, 1.0))

    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)  # most obj files expect to be smooth-shaded
    glEnable(GL_CULL_FACE)
    glFrontFace(GL_CCW)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

    # LOAD OBJECT AFTER PYGAME INIT
    # obj = OBJ(sys.argv[1], swapyz=True)
    obj = OBJ('worm3d6.obj', swapyz=True)

    clock = pygame.time.Clock()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    width, height = viewport
    gluPerspective(90.0, width / float(height), 1, 100.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)

    rx, ry = (0, 0)
    tx, ty = (0, 0)
    zpos = 5
    rotate = move = False
    while 1:
        clock.tick(30)
        for e in pygame.event.get():
            if e.type == QUIT:
                sys.exit()
            elif e.type == KEYDOWN and e.key == K_ESCAPE:
                sys.exit()
            elif e.type == MOUSEBUTTONDOWN:
                if e.button == 4:
                    zpos = max(1, zpos - 1)
                elif e.button == 5:
                    zpos += 1
                elif e.button == 1:
                    rotate = True
                elif e.button == 3:
                    move = True
            elif e.type == MOUSEBUTTONUP:
                if e.button == 1:
                    rotate = False
                elif e.button == 3:
                    move = False
            elif e.type == MOUSEMOTION:
                i, j = e.rel
                if rotate:
                    rx += i
                    ry += j
                if move:
                    tx += i
                    ty -= j

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # RENDER OBJECT
        glTranslate(tx / 20., ty / 20., - zpos)
        glRotate(ry, 1, 0, 0)
        glRotate(rx, 0, 1, 0)
        glCallList(obj.gl_list)

        pygame.display.flip()


class WormPlotter():

    def __init__(self, sampleinterval=0.01, timewindow=10., size=(720, 1080)):  # 1000,600
        # Data stuff
        self._interval = int(sampleinterval)
        self._bufsize = int(timewindow / sampleinterval)
        self.databuffer_time = collections.deque([0.0] * self._bufsize, self._bufsize)
        # self.databuffer_y= collections.deque([0.0]*self._bufsize, self._bufsize)
        self.x = np.zeros(self._bufsize, dtype=np.float)  # linspace(-timewindow, 0.0, self._bufsize)
        self.y = np.zeros(self._bufsize, dtype=np.float)
        # PyQtGraph stuff
        # self.app = QtGui.QApplication([])
        # pg.setConfigOptions(antialias=True)
        # self.plt = pg.plot(title='c.elegans states')
        # self.plt.resize(*size)
        # self.plt.showGrid(x=True, y=True)
        # self.plt.setLabel('left', 'amplitude', 'V')
        # self.plt.setLabel('bottom', 'time', 's')
        # self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))

        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget(show=True, size=size, title="c.elegans states")
        # self.plt = pg.PlotWidget(background="w")

        self.state_to_plot = {"Membrane potential (Head neurons)": ["SMDD", "SMDV", "RMDD", "RMDV"],
                              # "Membrane potential (tail neurons)": ["SMDDB", "SMDVB", "RMDDB", "RMDVB"],
                              "Nacl sensor neurons": ["ASEL", "ASER"],
                              "Nacl Klinotaxis outputs to head muscles": ["ventral_klinotaxis", "dorsal_klinotaxis"],
                              "Touch snsor neurons": ["AVM", "PLM"]}

        self.state_color = {"SMDD": (138, 43, 226), "SMDV": (139, 0, 139), "RMDD": (123, 104, 238),
                            "RMDV": (0, 139, 139),
                            # "SMDDB": (138, 43, 226), "SMDVB": (139, 0, 139), "RMDDB": (123, 104, 238),"RMDVB": (0, 139, 139),
                            "ASEL": (255, 127, 80), "ASER": (255, 69, 0),
                            "ventral_klinotaxis": (138, 43, 226), "dorsal_klinotaxis": (123, 104, 238),
                            "AVM": (255, 127, 80), "PLM": (69, 255, 0)}

        self.plots = []
        self.curve = {}
        self.curve_buffer = {}
        for key in self.state_to_plot.keys():
            ploter = self.win.addPlot(left="amplitude", bottom="time", title=key)
            ploter.addLegend()
            self.curve[key] = {}
            self.curve_buffer[key] = {}
            for item_name in self.state_to_plot[key]:
                #print("key", "item_name", key, item_name)
                #print("self.x", self.x)
                #print("self.y", self.y)
                self.curve[key][item_name] = ploter.plot(self.x, self.y,
                                                         pen=pg.mkPen(self.state_color[item_name], width=2),
                                                         name=item_name)
                self.curve_buffer[key][item_name] = collections.deque([0.0] * self._bufsize, self._bufsize)
            self.plots.append(ploter)
            #print("*******")
            self.win.nextRow()
        # self.win.addLabel(text= "label")

        # QTimer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateplot)
        self.timer.start(self._interval)

        #print("self.timer", self.timer)

    def updateplot(self):
        state_dict = state_queue.get()
        #print("state_dict", state_dict)
        self.databuffer_time.append(state_dict["time"])
        for plot_key, item_dict in self.curve.items():
            for item_key in item_dict.keys():
                self.curve_buffer[plot_key][item_key].append(state_dict[item_key])
                #print("self.databuffer_time", self.databuffer_time)
                self.curve[plot_key][item_key].setData(self.databuffer_time, self.curve_buffer[plot_key][item_key])
                #print("self.curve_buffer[plot_key][item_key]", self.curve_buffer[plot_key][item_key])
                #print("plot_key", plot_key)
                #print("item_key", item_key)
        self.app.processEvents()

    def run(self):
        sys.exit(self.app.exec_())


def plot_state():
    m = WormPlotter(sampleinterval=0.05, timewindow=10.)
    m.run()


def run_worm():
    # start_time = time.time()
    while True:
        worm_net.run(100000)
    # end_time = time.time()
    # print(end_time - start_time)


if __name__ == '__main__':
    # scale = 1e6*0.0001
    scale = 1e6 * 0.5  # 0.5
    shift = (950, 500)

    plot_state_flag = True

    xy_pos = (0.001, 0.0009)
    r = 5  # 10
    nacl = Nacl(xy_pos[0], xy_pos[1], 10000, 50 * 5, r, scale)  # x_center,y_center,alpha,peak,r_pixel,scale

    # nacl.update_pool()

    if plot_state_flag:
        state_queue = Queue()
        # funcs = [run_worm, run_gui, plot_state, run_worm3d]
        funcs = [run_worm, run_gui, plot_state]
    else:
        state_queue = None
        # funcs = [run_worm, run_gui,run_worm3d]
        funcs = [run_worm, run_gui]

    queue = Queue()

    nacl_location = Array('d', range(2))
    nacl_location[0] = xy_pos[0]
    nacl_location[1] = xy_pos[1]

    touch_signal = Array('d', range(2))
    touch_signal[0] = 0
    touch_signal[1] = 0

    worm = Worm(step_size=step_size, test=False, parameters=worm_body_parameters, shared_lists=queue,
                gui_parameters=(scale, shift), nacl_location=nacl_location, state_queue=state_queue)
    worm_net = WormNet(step_size=step_size * ms, worm_entity=worm, nacl_entity=nacl, touch_entity=touch_signal,
                       head_parameters=head_parameters, head_chemical_parameters=head_chemical_parameters,
                       head_gap_parameters=head_gap_parameters,
                       tail_parameters=tail_parameters, tail_chemical_parameters=tail_chemical_parameters,
                       tail_gap_parameters=tail_gap_parameters,
                       vnc_parameters=vnc_parameters, vnc_chemical_parameters=vnc_chemical_parameters,
                       vnc_gap_parameters=vnc_gap_parameters,
                       vncb_parameters=vncb_parameters, vncb_chemical_parameters=vncb_chemical_parameters,
                       vncb_gap_parameters=vncb_gap_parameters,
                       klinotaxis_parameters=klinotaxis_parameters,
                       klinotaxis_chemical_parameters=klinotaxis_chemical_parameters,
                       klinotaxis_gap_parameters=klinotaxis_gap_parameters
                       )

    p_lists = [Process(target=f) for f in funcs]
    for p in p_lists:
        p.start()
    for p in p_lists:
        p.join()
