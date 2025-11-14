import sys
import random

from domain.Stroke import Stroke
from domain.Clipping import Clipping

class LetterSet:
    """Un conjunto de letras es una lista de trazos.

    Attributes:
        strokes_list: list. Trazos del conjunto de letras.
        task: Task. Tarea a la que pertenece el conjunto de letras.
        number: int. Posición del conjunto de letras en la tarea.
        clippings_list: list. Recortes de la letra.
        coordinates_clippings_dict: dict. Recortes a los que pertenece cada coordenada.
        predicted_strokes_list: list. Trazos generados tras la predicción.
        unpredicted_coordinates_list: list. Coordenadas sin recorte.
        predicted_h_length: int. Longitud total de los trazos H en distancia.
        predicted_pd_length: int. Longitud total de los trazos PD en distancia.
        pd_predicted: int. 0 para H y 1 para PD.
    """

    def __init__(self, strokes_list: list[Stroke], task: "Task", number: int):
        self.strokes_list = strokes_list
        self.task = task
        self.number = number
        self.clippings_list: list[Clipping] = []
        self.coordinates_clippings_dict: dict[tuple[int, int], list[Clipping]] = {}
        self.predicted_strokes_list: list[Stroke] = []
        self.unpredicted_coordinates_list: list[tuple[int, int]] = []
        self.predicted_h_length: int = 0
        self.predicted_pd_length: int = 0
        self.pd_predicted: int
        self.min_x = float("inf")
        self.max_x = float("-inf")
        self.min_y = float("inf")
        self.max_y = float("-inf")
        for stroke in strokes_list:
            self.min_x = min(self.min_x, stroke.getMinX())
            self.max_x = max(self.max_x, stroke.getMaxX())
            self.min_y = min(self.min_y, stroke.getMinY())
            self.max_y = max(self.max_y, stroke.getMaxY())

    def setMinX(self, min_x: int):
        self.min_x = min_x

    def setMaxX(self, max_x: int):
        self.max_x = max_x

    def setMinY(self, min_y: int):
        self.min_y = min_y

    def setMaxY(self, max_y: int):
        self.max_y = max_y

    def getMinX(self):
        return self.min_x
    
    def getMaxX(self):
        return self.max_x
    
    def getMinY(self):
        return self.min_y
    
    def getMaxY(self):
        return self.max_y

    def get_subject_id(self) -> int:
        """Devuelve la ID del sujeto al que pertenece el conjunto de letras."""

        return self.task.subject_id

    def get_task_number(self) -> int:
        """Devuelve el número de la tarea a la que pertenece el conjunto de letras."""

        return self.task.task_number

    def generate_clippings(self, clipping_side_size: int, clipping_jump_size: int):
        """Genera los recortes del conjunto de letras."""

        self.clippings_list = []
        self.coordinates_clippings_dict = {}

        clipping_number = 0
        for set_stroke in self.strokes_list:
            coordinate_i = 0
            while coordinate_i < len(set_stroke):
                (
                    clipping_strokes_list,
                    clipping_coordinates_list,
                ) = self._find_clipping_strokes(
                    set_stroke[coordinate_i], clipping_side_size
                )

                clipping = Clipping(
                    clipping_strokes_list,
                    set_stroke[coordinate_i],
                    clipping_side_size,
                    clipping_jump_size,
                    self,
                    clipping_number,
                )
                self.clippings_list.append(clipping)
                for coordinate in clipping_coordinates_list:
                    if coordinate in self.coordinates_clippings_dict:
                        self.coordinates_clippings_dict[coordinate].append(clipping)
                    else:
                        self.coordinates_clippings_dict[coordinate] = [clipping]

                clipping_number += 1
                coordinate_i += clipping_jump_size

    def _find_clipping_strokes(
        self, center_coordinate: tuple[int, int], clipping_side_size
    ) -> tuple[list[Stroke], list[tuple[int, int]]]:
        """Devuelve la lista de los trazos del recorte."""

        clipping_strokes_list = []
        clipping_coordinates_list = []
        max_range = (clipping_side_size - 1) / 2
        from_in_range = False
        for set_stroke in self.strokes_list:
            current_stroke = set_stroke
            for current_coordinate in set_stroke:
                x_distance = abs(center_coordinate[0] - current_coordinate[0])
                y_distance = abs(center_coordinate[1] - current_coordinate[1])
                if (x_distance <= max_range) and (y_distance <= max_range):
                    clipping_coordinates_list.append(current_coordinate)
                    if from_in_range:
                        if current_stroke == set_stroke:
                            clipping_strokes_list[-1].append(current_coordinate)
                        else:
                            clipping_strokes_list.append(Stroke(current_coordinate))
                            current_stroke = set_stroke
                    else:
                        clipping_strokes_list.append(Stroke(current_coordinate))
                        from_in_range = True
                else:
                    from_in_range = False

        return clipping_strokes_list, clipping_coordinates_list

    def get_text_coordinates(self) -> tuple[int, int]:
        """Devuelve las coordenadas para escribir texto bajo el conjunto de letras."""

        height = sys.maxsize
        x_start = sys.maxsize
        x_end = 0

        for stroke in self.strokes_list:
            for coordinate in stroke:
                if coordinate[1] < height:
                    height = coordinate[1]
                if coordinate[0] < x_start:
                    x_start = coordinate[0]
                if coordinate[0] > x_end:
                    x_end = coordinate[0]

        x_coordinate = round((x_start + x_end) / 2 - 60)
        height -= 60

        return x_coordinate, height

    def generate_prediction_results(self):
        """Genera las predicciones a nivel de trazo y de conjunto de letras."""

        for stroke in self.strokes_list:
            current_predicted_pd = -1
            for coordinate in stroke:
                if coordinate in self.coordinates_clippings_dict:
                    coordinate_h = 0
                    coordinate_pd = 0
                    for clipping in self.coordinates_clippings_dict[coordinate]:
                        if clipping.pd_predicted == 0:
                            coordinate_h += 1
                        else:
                            coordinate_pd += 1
                    if coordinate_h == coordinate_pd:
                        predicted_pd = random.randint(0, 1)
                    elif coordinate_h > coordinate_pd:
                        predicted_pd = 0
                    else:
                        predicted_pd = 1

                    # Se generan los nuevos trazos en base
                    # a la predicción de las coordenadas.
                    if predicted_pd != current_predicted_pd:
                        new_stroke = Stroke(coordinate)
                        new_stroke.pd_predicted = predicted_pd
                        self.predicted_strokes_list.append(new_stroke)
                        current_predicted_pd = predicted_pd
                    else:
                        self.predicted_strokes_list[-1].append(coordinate)
                else:
                    self.unpredicted_coordinates_list.append(coordinate)
                    current_predicted_pd = -1

        for stroke in self.predicted_strokes_list:
            if stroke.pd_predicted == 0:
                self.predicted_h_length += stroke.length()
            else:
                self.predicted_pd_length += stroke.length()

        if self.predicted_h_length > self.predicted_pd_length:
            self.pd_predicted = 0
        else:
            self.pd_predicted = 1
