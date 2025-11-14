import math

class Stroke(list[tuple[int, int, int, int, int]]):
    """Un trazo es una lista de coordenadas (x, y).

    Attributes:
        pd_predicted: int. 0 para H y 1 para PD.
    """

    def __init__(self, initial_coordinate: tuple[int, int, int, int, int]):
        super().__init__([initial_coordinate])
        x, y, _, _, _, _, _= initial_coordinate
        self.min_x = self.max_x = x
        self.min_y = self.max_y = y
        self.pd_predicted: int

    def append(self, coordinate: tuple[int, int, int, int, int]):
        x, y, _, _, _, _, _ = coordinate
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)
        super().append(coordinate)

    def getMinX(self):
        return self.min_x
    
    def getMaxX(self):
        return self.max_x
    
    def getMinY(self):
        return self.min_y
    
    def getMaxY(self):
        return self.max_y

    def set_pd_predicted(self, pd_predicted: int):
        self.pd_predicted = pd_predicted

    def get_x_coordinates_list(self) -> list[int]:
        """Devuelve la lista de coordenadas x del trazo."""

        return list(zip(*self))[0]

    def get_y_coordinates_list(self) -> list[int]:
        """Devuelve la lista de coordenadas y del trazo."""

        return list(zip(*self))[1]
    
    def getTimestamps(self):
        return list(zip(*self))[2]
    
    def getButtonState(self):
        return list(zip(*self))[3]
    
    def getAzimuths(self) -> list[int]:
        return list(zip(*self))[4]
    
    def getAltitudes(self) -> list[int]:
        return list(zip(*self))[5]
    
    def getPressures(self) -> list[int]:
        return list(zip(*self))[6]

    def distance(self, stroke: "Stroke") -> int:
        """Devuelve la distancia a otro trazo."""

        dist_a = math.dist((self[0][0], self[0][1]), (stroke[-1][0], stroke[-1][1]))
        dist_b = math.dist((self[-1][0], self[-1][1]), (stroke[0][0], stroke[0][1]))

        return round(min(dist_a, dist_b))

    def length(self) -> int:
        """Devuelve la longitud del trazo."""

        total_length = 0
        i = 1
        while i < len(self):
            total_length += math.dist(
                (self[i - 1][0], self[i - 1][1]), (self[i][0], self[i][1])
            )
            i += 1

        return round(total_length)
