import numpy as np
import sys
import os
import cv2

from matplotlib import pyplot as plt

from utils.CustomMorphOps import normalize, simple_bresenham_line, bresenham_line, fit_into_normalized_canvas, clean_and_refill

from domain.Stroke import Stroke
from domain.LetterSet import LetterSet
from domain.RepresentationType import RepresentationType

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

    def __init__(self, subject_id: int, task_number: int, strokes_list: list[Stroke], all_coords: list[tuple[int, int, int, int, int, int, int]], pd_status=0, rep_type: RepresentationType = None, cache_base_dir = "cache"):
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
        self.rep_type = rep_type
        self.cache_base_dir = cache_base_dir
        #Representation attributtes
        self.data = None
        self.data_cache_path = None

    def generate_data(self, final_h, final_w, task1 = False):
        """
        Generates representation data.
        For SIMPLE_STROKE and ENHANCED_STROKE stores/read PNG from disk.
        For MULTICHANNEL_STROKE tries to use stored ENHANCED_STROKE
        No cache for ONLINE_SIGNAL 
        """
        os.makedirs(self.cache_base_dir, exist_ok=True)

        #Cache path depending on RepresentationType
        base_name = f"{self.subject_id}_task{self.task_number}"

        cache_simple = os.path.join(self.cache_base_dir, base_name + "_simple.png")
        cache_enhanced = os.path.join(self.cache_base_dir, base_name + "_enhanced.png")
        cache_multichannel = os.path.join(self.cache_base_dir, base_name + "_multichannel.npz")

        # --- SIMPLE_STROKE -------------------------
        if self.rep_type == RepresentationType.SIMPLE_STROKE:
            if os.path.exists(cache_simple):
                self.data = cv2.imread(cache_simple, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                self.data_cache_path = cache_simple
                return

            result = self._rep_simple_stroke()
            normalized = fit_into_normalized_canvas(result, final_h, final_w)
            if not task1:
                cr_result = clean_and_refill(normalized)
                write_img = (cr_result * 255).astype(np.uint8)
            else:
                write_img = (normalized * 255).astype(np.uint8)
            cv2.imwrite(cache_simple, write_img)
            self.data = result
            self.data_cache_path = cache_simple
            return
        
        # --- ENHANCED_STROKE -------------------------
        if self.rep_type == RepresentationType.ENHANCED_STROKE:
            if os.path.exists(cache_enhanced):
                self.data = cv2.imread(cache_enhanced, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                self.data_cache_path = cache_enhanced
                return
            result = self._rep_enhanced_stroke()
            normalized = fit_into_normalized_canvas(result, final_h, final_w)
            if not task1:
                cr_result = clean_and_refill(normalized)
                write_img = (cr_result * 255).astype(np.uint8)
            else:
                write_img = (normalized * 255).astype(np.uint8)

            cv2.imwrite(cache_enhanced, write_img)
            self.data = result
            self.data_cache_path = cache_enhanced
            return
        
        # --- MULTICHANNEL_STROKE ---------------------
        if self.rep_type == RepresentationType.MULTICHANNEL:
            if os.path.exists(cache_multichannel):
                loaded = np.load(cache_multichannel)
                self.data = {key: loaded[key] for key in loaded.files}
                self.data_cache_path = cache_multichannel
                return
            
            if os.path.exists(cache_enhanced):
                base_img = cv2.imread(cache_enhanced, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            else:
                base_img = self._rep_enhanced_stroke()
                cv2.imwrite(cache_enhanced, (base_img * 255).astype(np.uint8))

            result_dict = self._rep_multichannel(base_img)

            np.savez(cache_multichannel, **result_dict)

            self.data = result_dict
            self.data_cache_path = cache_multichannel
            return
        
        # --- ONLINE_SIGNAL -------------------------
        if self.rep_type == RepresentationType.ONLINE_SIGNAL:
            self.data = self.all_coords
            return
    
    def _rep_simple_stroke(self):
        final_w = int(self.max_vals['x_surface'] - self.min_vals['x_surface'])
        final_h = int(self.max_vals['y_surface'] - self.min_vals['y_surface'])
        canvas = np.zeros((final_h, final_w), dtype=np.float32)
        
        for letters_set in self.letters_sets_list:
            for stroke in letters_set.strokes_list:
                stroke_x_list = stroke.get_x_coordinates_list()
                stroke_y_list = stroke.get_y_coordinates_list()
                normalized_x = [x - self.min_vals['x_surface'] for x in stroke_x_list]
                normalized_y = [y - self.min_vals['y_surface'] for y in stroke_y_list]
                
                for i in range(len(stroke_x_list) -1):
                    pixels = simple_bresenham_line(
                        normalized_x[i],
                        normalized_y[i],
                        normalized_x[i+1],
                        normalized_y[i+1],
                        thickness=2,
                        )
                    for y, x in pixels:
                        if 0 <= y < final_h and 0 <= x < final_w:
                            canvas[y, x] = 1.0
        
        flip_img = cv2.flip(canvas, 0)
        return flip_img

    def _rep_enhanced_stroke(self, min_thickness = 2, max_thickness = 10, min_dark_factor = 0.7, max_dark_factor = 0.99):
        final_w = int(self.max_vals['x_surface'] - self.min_vals['x_surface'])
        final_h = int(self.max_vals['y_surface'] - self.min_vals['y_surface'])
        canvas = np.ones((final_h, final_w), dtype=np.float32)
        for letters_set in self.letters_sets_list:
            for stroke in letters_set.strokes_list:
                stroke_x_list = stroke.get_x_coordinates_list()
                stroke_y_list = stroke.get_y_coordinates_list()
                normalized_x = [x - self.min_vals['x_surface'] for x in stroke_x_list]
                normalized_y = [y - self.min_vals['y_surface'] for y in stroke_y_list]
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
                        canvas[y, x] *= darkening_factor

        flip_img = cv2.flip(canvas, 0)
        negative_img = 1.0 - flip_img
        return negative_img


    def _rep_multichannel(self, base_img, min_thickness = 2, max_thickness = 10):
        final_w = int(self.max_vals['x_surface'] - self.min_vals['x_surface'])
        final_h = int(self.max_vals['y_surface'] - self.min_vals['y_surface'])
        canvases = {
            name: (
                base_img.copy() if name == 'stroke'
                else np.zeros((final_h, final_w), dtype=np.float32)
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
                        canvases['timestamp'][y, x] = normalized_timestamp[i]
                        canvases['azimuth'][y, x] = normalized_azimuths[i]
                        canvases['altitude'][y, x] = normalized_altitudes[i]
                        canvases['pressure'][y, x] = normalized_pressures[i]

        # Flip (igual que enhanced)
        for key in canvases:        
            # si es stroke y proviene del enhanced → NO flip
            if key == 'stroke' and base_img is not None:
                continue
            
            canvases[key] = cv2.flip(canvases[key], 0)
        return canvases

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
    
    def getRepType(self):
        return self.rep_type

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