import matplotlib
from matplotlib import pyplot as plt
import os
import shutil

from domain.Stroke import Stroke

class Clipping:
    """Un recorte es una lista de trazos.

    Attributes:
        strokes_list: list. Trazos del recorte.
        center_coordinate: tuple. (x, y) en el centro del recorte.
        side_size: int. Tamaño del lado del recorte.
        jump_size: int. Coordenadas que se saltan entre recortes.
        letters_set: LetterSet. Conjunto de letras al que pertenece el recorte.
        number: int. Posición del recorte en el conjunto de letras.
        folder: str. Directorio en el que se encuentra el recorte.
        name: str. Nombre del recorte.
        pd_predicted: int. 0 para H y 1 para PD.
    """

    def __init__(
        self,
        strokes_list: list[Stroke],
        center_coordinate: tuple[int, int],
        side_size: int,
        jump_size: int,
        letters_set,
        number: int,
    ):
        self.strokes_list = strokes_list
        self.center_coordinate = center_coordinate
        self.side_size = side_size
        self.jump_size = jump_size
        self.letters_set = letters_set
        self.number = number
        self.folder = f"{letters_set.get_task_number()}_{side_size}_{jump_size}"
        self.name = f"{letters_set.get_subject_id()}_{letters_set.number}_{number}.png"
        self.pd_predicted: int
        self.plot()

    def plot(self):
        """Genera el plot del recorte y lo guarda."""

        # Se crea el directorio si es necesario.
        os.makedirs(os.path.join("generated", self.folder), exist_ok=True)

        # Se genera el plot y se guarda si no existe.
        if not os.path.exists(os.path.join("generated", self.folder, self.name)):
            plt.ioff()
            matplotlib.rcParams["savefig.pad_inches"] = 0
            px = 1 / plt.rcParams["figure.dpi"]
            fig = plt.figure(figsize=(self.side_size * px, self.side_size * px))

            ax = plt.axes((0, 0, 1, 1), frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.autoscale(tight=True)

            for stroke in self.strokes_list:
                stroke_x_list = stroke.get_x_coordinates_list()
                stroke_y_list = stroke.get_y_coordinates_list()
                plt.plot(stroke_x_list, stroke_y_list, color="black")

            # Necesario para que el centro del recorte sea el correcto.
            x_axis = (
                self.center_coordinate[0] - self.side_size / 2,
                self.center_coordinate[0] + self.side_size / 2,
            )
            y_axis = (
                self.center_coordinate[1] - self.side_size / 2,
                self.center_coordinate[1] + self.side_size / 2,
            )
            plt.plot(x_axis, y_axis, alpha=0)

            plt.savefig(os.path.join("generated", self.folder, self.name))
            plt.close()

    def copy_clipping(self, clipping_type: str):
        """Copia el recorte al directorio correspondiente."""

        clipping_origin = os.path.join("generated", self.folder, self.name)
        clipping_dest = os.path.join("generated", clipping_type, self.name)

        shutil.copyfile(clipping_origin, clipping_dest)
