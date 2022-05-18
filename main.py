from tkinter import *
from PIL import ImageTk, Image
from random import randint
import copy
import threading
import numpy as np

SPECIMENS_COUNT = 200
BEST_COUNT = 2
INT32_MIN = 0
INT32_MAX = 2**32
SCALE_CHANGE_GENERATION = 5


def int8(la):
    return la & 0xff
 
def la(int8):
    return int8 & 0xff

def rand_int32():
    return randint(INT32_MIN, INT32_MAX)

class Specimen(object):

    def __init__(self, width, height, data=None):
        self._width = width
        self._height = height
        self._score = float(0)

        if data is None:
            self._new_data()
        else:
            self._make_data(data)

        self._vectorized_mutator = np.vectorize(self._mutate)

    @property
    def data(self):
        return self._data

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def size(self):
        return self.width * self.height

    @property
    def score(self):
        return self._score

    @property
    def image_data(self):
        return [la(v) for v in self._data]

    def _new_data(self):
        self._data = np.full(self.size, 0, dtype=int)

    def _make_data(self, data):
        self._data = np.array([int8(v) for v in data])

    def iterator(self):
        return range(self.size)

    def __getitem__(self, position):
        return self._data[position]

    def __setitem__(self, position, color):
        self._data[position] = color

    def mutate_pixel(self, position, change):
        self._data[position] = self._mutate(self._data[position], change)

    def _mutate(self, value, change):
        return (value + change) >> 1

    def mutate(self, scale=1.0):
        x = rand_int32() % (self.width - 1)
        y = rand_int32() % (self.height - 1)
        w = rand_int32() % (self.width - x - 1) + 1
        h = rand_int32() % (self.height - y - 1) + 1

        w = max(1, int(w * scale))
        h = max(1, int(h * scale))

        c = rand_int32() & 0xff

        for n in range(y, y + h):
            begin = n * self.width
            end = begin + w

            self._data[begin:end] = self._vectorized_mutator(self._data[begin:end], c)

        # for n in range(y, y + h):
        #     for m in range(x, x + w): 
        #         self.mutate_pixel(n * self.width + m, c)

    def compare(self, specimen):
        self._score = np.linalg.norm(self.data - specimen.data)

    @property
    def image(self):
        image = Image.new('L', (self.width, self.height))
        image.putdata(self.image_data)

        return image

    def save(self, filename):
        self.image.save(filename)

class Population(threading.Thread):

    def __init__(self, original, canvas, specimens_count=SPECIMENS_COUNT):
        super().__init__()

        self.daemon = False
        self.stopped = False
        self.original = original
        self.canvas = canvas
        self.width = self.original.width
        self.height = self.original.height
        self.specimens_count = specimens_count
        self.generation = 0
        self.scale = 1.0
        self._init_specimens()

    def _init_specimens(self):
        self.specimens = [Specimen(self.width, self.height) 
                            for _ in range(self.specimens_count)]

    @property
    def best_specimen(self):
        return self.specimens[0]

    def stop(self):
        self.stopped = True

    def calculate_scale(self):
        self.scale = max(0.001, 1.0 - (self.generation / SCALE_CHANGE_GENERATION * 0.001))

    def mutate(self):
        for specimen in self.specimens:
            specimen.mutate(self.scale)
            specimen.compare(self.original)

        self.specimens = sorted(self.specimens, key=lambda specimen: specimen.score)

    def cross(self):
        for i in range(BEST_COUNT - 1, SPECIMENS_COUNT - 1):
            self.specimens[i] = copy.deepcopy(self.specimens[i % BEST_COUNT])

    def run(self):
        while not self.stopped:
            self.mutate()
            self.cross()

            self.generation += 1
            self.calculate_scale()

def render(canvas, data):
    image = copy.deepcopy(data)
    image = ImageTk.PhotoImage(image)

    canvas.create_image((0, 0), anchor='nw', image=image)
    canvas.image = image

def main():
    root = Tk()
    root.title('Genetic')

    original = Image.open('input/monalisa.png').convert('L')
    original = Specimen(original.width, original.height, data=list(original.getdata()))

    original_canvas = Canvas(root, width=original.width, height=original.height)
    original_canvas.pack(side=LEFT)
    render(original_canvas, original.image)

    best_canvas = Canvas(root, width=original.width, height=original.height, bd=0, highlightthickness=0)
    best_canvas.create_rectangle(0, 0, original.width, original.height, fill='black')
    best_canvas.pack(side=LEFT)

    population = Population(original, best_canvas)
    population.start()

    def update_population_canvas():
        render(best_canvas, population.best_specimen.image)

        root.after(2000, update_population_canvas)

    update_population_canvas()

    def print_generation():
        print('GENERATION:', population.generation, 'DISTANCE:', population.best_specimen.score, 'SCALE:', population.scale)

        root.after(5000, print_generation)

    print_generation()

    def when_close_window():
        population.stop()
        population.join()

        root.destroy()

    root.protocol("WM_DELETE_WINDOW", when_close_window)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()