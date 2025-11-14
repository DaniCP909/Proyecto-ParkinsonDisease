import numpy as np
import sys
import os
import cv2

from utils.CustomMorphOps import normalize, bresenham_line

from domain.Stroke import Stroke
from domain.LetterSet import LetterSet

class Task:
    """Una tarea es una lista de cinco conjuntos de letras.

    Attributes:
        subject_id: int. ID del sujeto al que pertenece la tarea.
        task_number: int. 2: "l" | 3: "le" | 4: "les"
        letters_sets_list: list. Lista de los conjuntos de letras que forman la tarea.
        predicted_h_length: int. Longitud total de los trazos H en distancia.
        predicted_pd_length: int. Longitud total de los trazos PD en distancia.
        pd_predicted: int. 0 para H y 1 para PD.
    """

    def __init__(self, subject_id: int, task_number: int, strokes_list: list[Stroke], all_coords: list[tuple[int, int, int, int, int, int, int]], pd_status=0):
        self.subject_id = subject_id
        self.task_number = task_number
        self.min_vals = {
            'x_surface': float('inf'),
            'y_surface': float('inf'),
            'x_all': float('inf'),
            'y_all': float('inf'),
            'timestamp': float('inf'),
            'pressure': float('inf'),
            'altitude': float('inf'),
            'azimuth': float('inf'),
        }
        self.max_vals = {
            'x_surface': float('-inf'),
            'y_surface': float('-inf'),
            'x_all': float('-inf'),
            'y_all': float('-inf'),
            'timestamp': float('-inf'),
            'pressure': float('-inf'),
            'altitude': float('-inf'),
            'azimuth': float('-inf'),
        }
        self.all_coords = all_coords #List with all coords (in air and on surface)
        self.letters_sets_list = self._get_letters_sets(strokes_list) #Generate letters sets and compute min/max (only on surface coords)
        self._compute_bounds() #Compute bounds of all coords and its data
        self.all_coords = self._normalize_coords_data() #normalize all coords and its data
        self.predicted_h_length: int = 0
        self.predicted_pd_length: int = 0
        self.pd_status = pd_status
        self.pd_predicted: int
        self.canvases = None

    def getHeight(self):
        return self.max_vals['y_surface'] - self.min_vals['y_surface']
    
    def getWidth(self):
        return self.max_vals['x_surface'] - self.min_vals['x_surface']

    def getCanvases(self):
        return self.canvases

    def setCanvases(self, canvases):
        self.canvases = canvases

    def getAllCordinates(self):
        return np.array(self.all_coords, dtype=np.float32)

    def _get_letters_sets(self, strokes_list: list[Stroke]) -> list[LetterSet]:
        """Recibe una lista de trazos y los agrupa en varios conjuntos de letras."""

        # Los trazos se agrupan en torno a los cinco trazos más largos.
        sorted_strokes = sorted(strokes_list, reverse=True, key=len)
        first_strokes = sorted_strokes[:5]

        # Excepción: Tarea 2, Sujeto 26.
        if self.subject_id == 26 and self.task_number == 2:
            first_strokes = sorted_strokes[1:6]

        # Excepción: Tarea 3, Sujeto 54.
        if self.subject_id == 54 and self.task_number == 3:
            first_strokes = sorted_strokes[:3]
            first_strokes = first_strokes + sorted_strokes[4:6]

        # Excepción: Tarea 2, Sujetos (27, 57). Tarea 4, Sujetos (2, 48, 85).
        if (self.subject_id in (27, 57) and self.task_number == 2) or (
            self.subject_id in (2, 48, 85) and self.task_number == 4
        ):
            first_strokes = sorted_strokes[:6]

        # Se ordenan los trazos por su posición original.
        first_strokes = sorted(first_strokes, key=lambda stroke: stroke[0][0])

        sets_strokes_list = []
        for task_stroke in first_strokes:
            sets_strokes_list.append([task_stroke])

        for task_stroke in strokes_list:
            letters_set = -1
            distance = sys.maxsize
            stroke_exists = False
            for letters_set_i in range(len(sets_strokes_list)):
                for set_stroke in sets_strokes_list[letters_set_i]:
                    if task_stroke == set_stroke:
                        stroke_exists = True
                        break
                    current_distance = task_stroke.distance(set_stroke)
                    if current_distance < distance:
                        distance = current_distance
                        letters_set = letters_set_i
                if stroke_exists:
                    break
            if not stroke_exists:
                sets_strokes_list[letters_set].append(task_stroke)

        # Ahora se ordena cada lista de trazos por su posición original.
        for i in range(0, len(sets_strokes_list)):
            sets_strokes_list[i] = sorted(
                sets_strokes_list[i],
                key=lambda stroke: strokes_list.index(stroke),
            )

        # Por último se generan todos los conjuntos de letras.
        letters_sets_list = []
        letters_set_i = 0
        while letters_set_i < len(sets_strokes_list):
            new_letter_set = LetterSet(sets_strokes_list[letters_set_i], self, letters_set_i)
            letters_sets_list.append(
                new_letter_set
            )
            self.min_vals['x_surface'] = min(self.min_vals['x_surface'], new_letter_set.getMinX())
            self.max_vals['x_surface'] = max(self.max_vals['x_surface'], new_letter_set.getMaxX())
            self.min_vals['y_surface'] = min(self.min_vals['y_surface'], new_letter_set.getMinY())
            self.max_vals['y_surface'] = max(self.max_vals['y_surface'], new_letter_set.getMaxY())
            letters_set_i += 1

        return letters_sets_list
    
    def _compute_bounds(self):
        for coord in self. all_coords:
            x, y, timestamp, _, azimuth, altitude, pressure = coord
            self.min_vals['x_all'] = min(self.min_vals['x_all'], x)
            self.max_vals['x_all'] = max(self.max_vals['x_all'], x)
            self.min_vals['y_all'] = min(self.min_vals['y_all'], y)
            self.max_vals['y_all'] = max(self.max_vals['y_all'], y)
            self.min_vals['timestamp'] = min(self.min_vals['timestamp'], timestamp)
            self.max_vals['timestamp'] = max(self.max_vals['timestamp'], timestamp)
            self.min_vals['azimuth'] = min(self.min_vals['azimuth'], azimuth)
            self.max_vals['azimuth'] = max(self.max_vals['azimuth'], azimuth)
            self.min_vals['altitude'] = min(self.min_vals['altitude'], altitude)
            self.max_vals['altitude'] = max(self.max_vals['altitude'], altitude)
            self.min_vals['pressure'] = min(self.min_vals['pressure'], pressure)
            self.max_vals['pressure'] = max(self.max_vals['pressure'], pressure)

    def _normalize_coords_data(self) -> list[tuple[float, float, float, float, float, float, float]]:
        norm_coords = []

        def norm(val, min_val, max_val):
            return (val - min_val) / (max_val - min_val) if max_val > min_val else 0.0
        
        for coord in self.all_coords:
            x, y, timestamp, button_state, azimuth, altitude, pressure = coord

            norm_x = norm(x, self.min_vals['x_all'], self.max_vals['x_all'])
            norm_y = norm(y, self.min_vals['y_all'], self.max_vals['y_all'])
            norm_timestamp = norm(timestamp, self.min_vals['timestamp'], self.max_vals['timestamp'])
            norm_button_state = float(button_state)
            norm_azimuth = norm(azimuth, self.min_vals['azimuth'], self.max_vals['azimuth'])
            norm_altitude = norm(altitude, self.min_vals['altitude'], self.max_vals['altitude'])
            norm_pressure = norm(pressure, self.min_vals['pressure'], self.max_vals['pressure'])

            norm_coords.append((
                norm_x,
                norm_y,
                norm_timestamp,
                norm_button_state,
                norm_azimuth,
                norm_altitude,
                norm_pressure
            ))
        return norm_coords

   
    def plot_task(self, subdir=None, min_thickness = 2, max_thickness = 10, min_dark_factor = 0.7, max_dark_factor = 0.99):
        final_w = int(self.max_vals['x_surface'] - self.min_vals['x_surface'])
        final_h = int(self.max_vals['y_surface'] - self.min_vals['y_surface'])
        canvases = {
            name: (
                np.ones((final_h, final_w), dtype=np.float64) * 255.0 if name == 'stroke'
                else np.zeros((final_h, final_w), dtype=np.float64)
            )
            for name in ['stroke', 'timestamp', 'azimuth', 'altitude', 'pressure']
        }
        for letters_set in self.letters_sets_list:
            for stroke in letters_set.strokes_list:
                stroke_x_list = stroke.get_x_coordinates_list()
                stroke_y_list = stroke.get_y_coordinates_list()
                normalized_x = [x - self.min_vals['x_surface'] for x in stroke_x_list]
                normalized_y = [y - self.min_vals['y_surface'] for y in stroke_y_list]
                normalized_timestamp = stroke.getTimestamps()
                normalized_azimuths = stroke.getAzimuths()
                altitudes = stroke.getAltitudes()
                normalized_altitudes = normalize(altitudes)
                pressures = stroke.getPressures()
                normalized_pressures = normalize(pressures)
                for i in range(len(stroke_x_list) -1):
                    darkening_factor = min_dark_factor + (max_dark_factor - min_dark_factor) * (1 - normalized_pressures[i])
                    thickness_factor = min_thickness + (max_thickness - min_thickness) * (1 - normalized_altitudes[i])
                    pixels = bresenham_line(
                        normalized_x[i],
                        normalized_y[i],
                        normalized_x[i+1],
                        normalized_y[i+1],
                        height=final_h,
                        width=final_w,
                        thickness=int(thickness_factor),
                        )
                    for y, x in pixels:
                        canvases['stroke'][y, x] = darkening_factor
                        canvases['timestamp'][y, x] = normalized_timestamp[i]
                        canvases['azimuth'][y, x] = normalized_azimuths[i]
                        canvases['altitude'][y, x] = normalized_altitudes[i]
                        canvases['pressure'][y, x] = normalized_pressures[i]

        output_path = os.path.join("tareas_generadas", subdir)
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.join(output_path, f"tarea{self.task_number}.png")
        stroke_canvas = canvases['stroke']
        stroke_canvas = cv2.flip(stroke_canvas, 0)
        stroke_canvas_uint8 = stroke_canvas.astype(np.uint8)
        negative_img = 255 - stroke_canvas_uint8
        canvases['stroke'] = (255 - stroke_canvas) / 255.0
        #canvas_resized = cv2.resize(canvas_uint8, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        png_img = (canvases['stroke'] * 255).astype(np.uint8)
        cv2.imwrite(filename, png_img)
        self.setCanvases(canvases)

    
    def generate_prediction_results(self):
        """Genera las predicciones a nivel de trazo,
        de conjunto de letras y de tarea."""

        for letters_sets in self.letters_sets_list:
            letters_sets.generate_prediction_results()
            self.predicted_h_length += letters_sets.predicted_h_length
            self.predicted_pd_length += letters_sets.predicted_pd_length

        if self.predicted_h_length > self.predicted_pd_length:
            self.pd_predicted = 0
        else:
            self.pd_predicted = 1