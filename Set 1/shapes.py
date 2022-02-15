class Shape:

    def __init__(self, position):
        self.position = position

    def contains(self, coordinate):
        raise NotImplementedError()


class Rectangle(Shape):

    def __init__(self, position, width, height):
        super().__init__(position)
        self.width = width
        self.height = height

    def contains(self, coordinate):

        return self.position[0] < coordinate[0] < self.position[0] + self.width and self.position[1] < coordinate[1] < self.position[1] + self.height


class Circle(Shape):

    def __init__(self, position, radius):
        super().__init__(position)
        self.radius = radius

    def contains(self, coordinate):

        return ((coordinate[0] - self.position[0]) ** 2 + (coordinate[1] - self.position[1]) ** 2) ** 0.5 < self.radius
