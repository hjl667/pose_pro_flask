class Coordinate:
    def __init__(self, x, y, z, label):
        self.x = x
        self.y = y
        self.z = z
        self.label = label


class Frame:
    def __init__(self, coordinates):
        self.coordinates = coordinates


class Video:

    def __init__(self, frames):
        self.frames = frames