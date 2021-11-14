from __future__ import division
import threading
import sys
import math
import random
import time

from collections import deque
from pyglet import image
from pyglet.g      l import *
from pyglet.graphics import TextureGroup
from pyglet.window import key, mouse

TICKS_PER_SEC = 60

# Size of sectors used to ease block loading.
SECTOR_SIZE = 16
LADO_DEL_MUNDO = 100
WALKING_SPEED = 0.5
FLYING_SPEED = 15

GRAVITY = 20.0
MAX_JUMP_HEIGHT = 1.0 # About the height of a block.
# To derive the formula for calculating jump speed, first solve
#    v_t = v_0 + a * t
# for the time at which you achieve maximum height, where a is the acceleration
# due to gravity and v_t = 0. This gives:
#    t = - v_0 / a
# Use t and the desired MAX_JUMP_HEIGHT to solve for v_0 (jump speed) in
#    s = s_0 + v_0 * t + (a * t^2) / 2
JUMP_SPEED = math.sqrt(2 * GRAVITY * MAX_JUMP_HEIGHT)
TERMINAL_VELOCITY = 50

PLAYER_HEIGHT = 2

if sys.version_info[0] >= 3:
    xrange = range

def cube_vertices(x, y, z, n):
    """ Return the vertices of the cube at position x, y, z with size 2*n.

    """
    return [
        x-n,y+n,z-n, x-n,y+n,z+n, x+n,y+n,z+n, x+n,y+n,z-n,  # top
        x-n,y-n,z-n, x+n,y-n,z-n, x+n,y-n,z+n, x-n,y-n,z+n,  # bottom
        x-n,y-n,z-n, x-n,y-n,z+n, x-n,y+n,z+n, x-n,y+n,z-n,  # left
        x+n,y-n,z+n, x+n,y-n,z-n, x+n,y+n,z-n, x+n,y+n,z+n,  # right
        x-n,y-n,z+n, x+n,y-n,z+n, x+n,y+n,z+n, x-n,y+n,z+n,  # front
        x+n,y-n,z-n, x-n,y-n,z-n, x-n,y+n,z-n, x+n,y+n,z-n,  # back
    ]


def tex_coord(x, y, n=16):
    """ Return the bounding vertices of the texture square.

    """
    m = 1.0 / n
    dx = x * m
    dy = y * m
    return dx, dy, dx + m, dy, dx + m, dy + m, dx, dy + m


def tex_coords(top, bottom, side):
    """ Return a list of the texture squares for the top, bottom and side.

    """
    top = tex_coord(*top)
    bottom = tex_coord(*bottom)
    side = tex_coord(*side)
    result = []
    result.extend(top)
    result.extend(bottom)
    result.extend(side * 4)
    return result


TEXTURE_PATH = 'terrain.png'

GRASS = tex_coords((0,15),(2,15),(2,15))
DIRT = tex_coords((2,15),(2,15),(2,15))
SAND = tex_coords((1, 1), (1, 1), (1, 1))
BRICK = tex_coords((2, 0), (2, 0), (2, 0))
STONE = tex_coords((2, 1), (2, 1), (2, 1))

FACES = [
    ( 0, 1, 0),
    ( 0,-1, 0),
    (-1, 0, 0),
    ( 1, 0, 0),
    ( 0, 0, 1),
    ( 0, 0,-1),
]


def normalize(position):
    x, y, z = position
    x, y, z = (int(round(x)), int(round(y)), int(round(z)))
    return (x, y, z)


def sectorize(position):
    """ Returns a tuple representing the sector for the given `position`.

    Parameters
    ----------
    position : tuple of len 3

    Returns
    -------
    sector : tuple of len 3

    """
    x, y, z = normalize(position)
    x, y, z = x // SECTOR_SIZE, y // SECTOR_SIZE, z // SECTOR_SIZE
    return (x, 0, z)


class Model(object):

    def __init__(self):
        self.batch = pyglet.graphics.Batch()
        self.group = TextureGroup(image.load(TEXTURE_PATH).get_texture())
        self.world = {}
        self.shown = {}
        self._shown = {}
        self.sectors = {}
        self.queue = deque()
        self._initialize()

    def _initialize(self):
        x = 1
        times = 0 
        z = 1
        p = ""
        f = open("world.txt","r")
        for k in f.read():
            if k !=",":
                p+=k
            elif k ==",":
                if p == "":continue
                self.add_block((x,round(int(p)*0.2),z),GRASS)
                x+=1    
                p = ""
            if x ==LADO_DEL_MUNDO:
                z+=1
                x=0
            if z ==LADO_DEL_MUNDO:
                break
        times += 0
        f.close()
    def world_2(self,m,texture):
        self.add_block((m[0],m[1],m[2]),texture)

    def exposed(self, position):
        """ Returns False is given `position` is surrounded on all 6 sides by
        blocks, True otherwise.

        """
        x, y, z = position
        for dx, dy, dz in FACES:
            if (x + dx, y + dy, z + dz) not in self.world:
                return True
        return False

    def add_block(self, position, texture):

        if position in self.world:
            self.remove_block(position)
        self.world[position] = texture
        self.sectors.setdefault(sectorize(position), []).append(position)
        if self.exposed(position):
            self.show_block(position)
            self.check_neighbors(position)

    def remove_block(self, position):
        del self.world[position]
        self.sectors[sectorize(position)].remove(position)
        if position in self.shown:
            self.hide_block(position)


    def check_neighbors(self, position):
        """ Check all blocks surrounding `position` and ensure their visual
        state is current. This means hiding blocks that are not exposed and
        ensuring that all exposed blocks are shown. Usually used after a block
        is added or removed.

        """
        x, y, z = position
        for dx, dy, dz in FACES:
            key = (x + dx, y + dy, z + dz)
            if key not in self.world:
                continue
            if self.exposed(key):
                if key not in self.shown:
                    self.show_block(key)
            else:
                if key in self.shown:
                    self.hide_block(key) 

    def show_block(self, position, immediate=True):
        texture = self.world[position]
        self.shown[position] = texture
        x, y, z = position
        vertex_data = cube_vertices(x, y, z, 0.5)
        texture_data = list(texture)
        # create vertex list
        # FIXME Maybe `add_indexed()` should be used instead
        self._shown[position] = self.batch.add(24, GL_QUADS, self.group,
            ('v3f/static', vertex_data),
            ('t2f/static', texture_data))

    def hide_block(self, position):
        self.shown.pop(position)
        self._shown.pop(position).delete()

    def show_sector(self, sector):
        for position in self.sectors.get(sector, []):
            if position not in self.shown and self.exposed(position):
                self.show_block(position, False)

    def hide_sector(self, sector):
        for position in self.sectors.get(sector, []):
            if position in self.shown:
                self.hide_block(position, False)

    def change_sectors(self, before, after):
        before_set = set()
        after_set = set()
        pad = 4
        for dx in xrange(-pad, pad + 1):
            for dy in [0]:  # xrange(-pad, pad + 1):
                for dz in xrange(-pad, pad + 1):
                    if dx ** 2 + dy ** 2 + dz ** 2 > (pad + 1) ** 2:
                        continue
                    if before:
                        x, y, z = before
                        before_set.add((x + dx, y + dy, z + dz))
                    if after:
                        x, y, z = after
                        after_set.add((x + dx, y + dy, z + dz))
        show = after_set - before_set
        hide = before_set - after_set
        for sector in show:
            self.show_sector(sector)
        for sector in hide:
            self.hide_sector(sector)

class Window(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)

        self.exclusive = False
        self.flying = False
        self.position = (0, 15, 1)
        self.rotation = (0.1, 0) 
        self.mov = [0,0,0,0]
        self.model = Model()
        self.sector = None
        self.model.show_sector(0)
        pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SEC)
        self.world = []
        self.times = 0
        mmm = self.model.world.keys()
        for l in mmm:
            self.world.append(l)
        print()
        

    
    def set_exclusive_mouse(self, exclusive):
        super(Window, self).set_exclusive_mouse(exclusive)
        self.exclusive = exclusive
        


    def update(self, dt):
        self.times += 1
        if not self.times-300 < 300:
            self.model.add_block((self.world[self.times-300][0],self.world[self.times-300][1]-1,self.world[self.times-300][2]),DIRT)
            self.model.add_block((self.world[self.times-300][0],self.world[self.times-300][1]-2,self.world[self.times-300][2]),DIRT)
            
        x, y = self.rotation
        m = math.cos(math.radians(y))
        dz = math.sin(math.radians(x - 90)) * m
        dx = round(math.cos(math.radians(x - 90)) * m,2)
        l = list(self.position)
        def w_mov():
            if self.mov[0]==1:
                l[0] += dx*WALKING_SPEED
                l[2] += dz*WALKING_SPEED
                self.position = tuple(l)
        def s_mov():
            if self.mov[0]==-1:
                l[0] -= dx*WALKING_SPEED
                l[2] -= dz*WALKING_SPEED
                self.position = tuple(l)
        def a_mov():
            if self.mov[1]==1:
                l[2] -= dx*WALKING_SPEED
                l[0] += dz*WALKING_SPEED
                self.position = tuple(l)
        def d_mov():
            if self.mov[1]==-1:
                l[2] += dx*WALKING_SPEED
                l[0] -= dz*WALKING_SPEED
                self.position = tuple(l)
        def thfly():
            if self.mov[2]==1:
                l[1]+= JUMP_SPEED*0.05
                self.position = tuple(l)
            if self.mov[3]==1:
                l[1] -= JUMP_SPEED*0.05  
                self.position = tuple(l)    
        thW=threading.Thread(target=w_mov)
        thS=threading.Thread(target=s_mov)
        thA=threading.Thread(target=a_mov)
        thD=threading.Thread(target=d_mov)
        thfly=threading.Thread(target=thfly)
        thW.start()
        thS.start()
        thA.start()
        thD.start()
        thfly.start()


        return
    def on_key_press(self, symbol,modifiers):
            if symbol == key.W:
                self.mov[0] = 1
            elif symbol == key.S:
                self.mov[0] = -1
            elif symbol == key.A:
                self.mov[1] = 1
            elif symbol == key.D:
                self.mov[1] = -1
            elif symbol == key.SPACE:
                self.mov[2] = 1
            if modifiers & key.MOD_SHIFT:
                self.mov[3]= 1
    def on_key_release(self, symbol, modifiers):
            if symbol == key.W:
                self.mov[0] = 0
            elif symbol == key.S:
                self.mov[0] = 0
            elif symbol == key.A:
                self.mov[1] = 0
            elif symbol == key.D:
                self.mov[1] = 0
            elif symbol == key.SPACE:
                self.mov[2]=0
            elif modifiers == 17:
                self.mov[3]=0
            elif not modifiers & key.MOD_SHIFT:
                self.mov[3]= 0
    def on_mouse_motion(self, x, y, dx, dy):
        if self.exclusive:
            m = 0.15
            x, y = self.rotation
            x, y = x + dx * m, y + dy * m
            y = max(-90, min(90, y))
            self.rotation = (x, y)        

    def colisiones(self):
        return
    def set_3d(self):
        width, height = self.get_size()
        glEnable(GL_DEPTH_TEST)
        viewport = self.get_viewport_size()
        glViewport(0, 0, max(1, viewport[0]), max(1, viewport[1]))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65.0, width / float(height), 0.1, 600.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        x, y = self.rotation
        glRotatef(x,0, 1, 0)
        glRotatef(-y, math.cos(math.radians(x)), 0, math.sin(math.radians(x)))
        x, y, z = self.position
        glTranslatef(-x, -y, -z)
    def on_draw(self):
        self.clear()
        self.set_3d()
        glColor3d(1, 1, 1)
        self.model.batch.draw()
    def initialize_world(self):
        time.sleep(2)
        self.model.add_block((5,2,3),STONE)
        """

        """
class Perlin():
    def __init__(self):
        import numpy as np
        import matplotlib.pyplot as plt
        def generate_perlin_noise_2d(shape, res):
            
            def f(t):
                return 6*t**5 - 15*t**4 + 10*t**3

            delta = (res[0] / shape[0], res[1] / shape[1])
            d = (shape[0] // res[0], shape[1] // res[1])
            grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
            # Gradients
            angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
            gradients = np.dstack((np.cos(angles), np.sin(angles)))
            g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
            g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
            g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
            g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
            # Ramps
            n00 = np.sum(grid * g00, 2)
            n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
            n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
            n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
            # Interpolation
            t = f(grid)
            n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
            n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
            return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)
        noise = generate_perlin_noise_2d([1000,1000],[1,1])
        a_file = open("world.txt", "a")

        xpix, ypix = 1000,1000
        for j in range(xpix):
            for i in range(ypix):
                p = noise[j][i]
                k = str(p)
                if k[0]=="-":
                    k = k[3:6]
                elif k[0]=="0":
                    k = k[2:5]
                else:print("Error:"+k);k="0"
                k+=","
                a_file.write(k)
        a_file.close()
        plt.imshow(noise, cmap='gray')
        plt.show()
                

def setup_fog():
    """ Configure the OpenGL fog properties.

    """
    # Enable fog. Fog "blends a fog color with each rasterized pixel fragment's
    # post-texturing color."
    glEnable(GL_FOG)
    # Set the fog color.
    glFogfv(GL_FOG_COLOR, (GLfloat * 4)(0.5, 0.69, 1.0, 1))
    # Say we have no preference between rendering speed and quality.
    glHint(GL_FOG_HINT, GL_DONT_CARE)
    # Specify the equation used to compute the blending factor.
    glFogi(GL_FOG_MODE, GL_LINEAR)
    # How close and far away fog starts and ends. The closer the start and end,
    # the denser the fog in the fog range.
    glFogf(GL_FOG_START, 20.0)
    glFogf(GL_FOG_END, 60.0)



def setup():

    # Set the color of "clear", i.e. the sky, in rgba.
    glClearColor(0.5, 0.69, 1.0, 1)

    glEnable(GL_CULL_FACE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

def main():
    window = Window(width=1920, height=1080, caption='Minecraft Python', resizable=True)
    # Hide the mouse cursor and prevent the mouse from leaving the window.
    window.set_exclusive_mouse(True)
    setup()
    pyglet.app.run()


if __name__ == '__main__':
    main()
    