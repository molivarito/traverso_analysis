import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from openwind import Player, ImpedanceComputation, InstrumentGeometry

from notion_utils import get_json_files_from_notion

import logging

logger = logging.getLogger(__name__)

FLUTE_PARTS_ORDER = ["headjoint", "left", "right", "foot"]

# Default acoustic parameters for holes if not specified elsewhere
DEFAULT_CHIMNEY_HEIGHT = 3e-3 # meters
DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT = 5e-3 # meters
DEFAULT_HOLE_RADIUS_OUT_FACTOR = 1.2 # Factor to estimate outer radius from inner if not given

# Intenta construir una ruta más robusta para el archivo de digitaciones por defecto.
# Esto asume que data_json está en el directorio padre del directorio donde reside este script (flute_data.py)
# o al mismo nivel si flute_data.py está en el directorio raíz del proyecto.
# Para una aplicación empaquetada, se necesitarían otros métodos (ej. importlib.resources).
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FING_CHART_PATH = SCRIPT_DIR.parent / "data_json" / "traverso_fingerchart.txt"
if not DEFAULT_FING_CHART_PATH.exists(): # Fallback si la estructura anterior no se cumple
    DEFAULT_FING_CHART_PATH = Path("data_json") / "traverso_fingerchart.txt"
class FluteData:
    def __init__(self, source: str, notion_token: str = None, database_id: str = None,
                 fing_chart_file: str = str(DEFAULT_FING_CHART_PATH), temperature: float = 20,
                 la_frequency: float = 415) -> None:
        """
        Constructor de la clase FluteData.

        Args:
            source (str): Ruta base para archivos JSON o nombre en Notion.
            notion_token (str, opcional): Token de Notion.
            database_id (str, opcional): ID de la base de datos en Notion.
            fing_chart_file (str): Ruta al archivo de digitaciones.
            temperature (float): Temperatura en °C.
            la_frequency (float): Frecuencia de referencia del LA (default 415 Hz).
        """
        self.data: Dict[str, Any] = {}
        self.acoustic_analysis: Dict[str, Any] = {}
        self.instrument: Dict[str, Any] = {}
        self.combined_measurements: List[Dict[str, float]] = []
        self.la_frequency: float = la_frequency

        self.flute_model = Path(source).name if Path(source).is_dir() else source

        try:
            if notion_token and database_id:
                self._read_json_data_from_notion(notion_token, database_id, source)
            else:
                self._read_json_data_from_files(source)
            self.data["Flute Model"] = self.flute_model

            # Calcular las frecuencias de digitación a partir del archivo finger chart y la frecuencia del LA.
            try:
                with Path(fing_chart_file).open("r") as f:
                    header_line = f.readline().strip()
                tokens = header_line.split()
                # Se asume que la primera palabra es "label" y las siguientes son las notas.
                note_names = tokens[1:]
                # Mapeo de semitonos relativo al LA: D:-7, E:-5, Fs:-3, G:-2, A:0, B:2, Cs:4.
                semitone_mapping = {"D": -7, "E": -5, "Fs": -3, "G": -2, "A": 0, "B": 2, "Cs": 4}
                self.finger_frequencies: Dict[str, float] = {}
                for note in note_names:
                    n = semitone_mapping.get(note)
                    if n is not None:
                        self.finger_frequencies[note] = self.la_frequency * (2 ** (n / 12))
                    else:
                        self.finger_frequencies[note] = None
            except Exception as e:
                logger.error("Error al leer o procesar el archivo de digitaciones: %s", e)
                self.finger_frequencies = {}

            self.combined_measurements = self.combine_measurements()
            self.compute_acoustic_analysis(fing_chart_file, temperature)
        except Exception as e:
            logger.exception("Error al inicializar FluteData: %s", e)
            raise ValueError(f"Error al procesar los datos de la flauta: {e}")

    def _read_json_data_from_files(self, base_dir: str) -> None:
        """Carga datos desde archivos JSON locales."""
        for part in FLUTE_PARTS_ORDER:
            json_path = Path(base_dir) / f"{part}.json"
            try:
                with json_path.open('r') as file:
                    self.data[part] = json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"No se encontró el archivo: {json_path}")
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"Error al decodificar JSON: {json_path}", doc=str(e), pos=0)

    def _read_json_data_from_notion(self, notion_token: str, database_id: str, flute_name: str) -> None:
        """
        Carga datos desde Notion y los guarda temporalmente.

        Args:
            notion_token (str): Token de Notion.
            database_id (str): ID de la base de datos.
            flute_name (str): Nombre de la flauta.

        Raises:
            ValueError: Si los datos están incompletos.
        """
        try:
            headjoint_data, left_data, right_data, foot_data = get_json_files_from_notion(
                notion_token, database_id, flute_name
            )
            if not (headjoint_data and left_data and right_data and foot_data):
                raise ValueError(f"No se pudieron recuperar todos los datos desde Notion para '{flute_name}'.")
            # Los datos ya están en memoria, no es necesario guardarlos en archivos temporales.
            self.data = {
                "headjoint": headjoint_data,
                "left": left_data,
                "right": right_data,
                "foot": foot_data,
            }
        except Exception as e:
            logger.exception("Error al obtener datos de Notion: %s", e)
            raise ValueError(f"Error al obtener datos de Notion: {e}")

    def combine_measurements(self) -> List[Dict[str, float]]:
        """
        Combina mediciones de todas las partes ajustando las posiciones.

        Returns:
            list: Lista de mediciones combinadas.
        """
        combined_measurements = []
        current_position = 0
        # Longitud efectiva de cada parte (longitud total menos la espiga que se inserta)
        effective_lengths = {}
        for part_name in FLUTE_PARTS_ORDER:
            part_data = self.data.get(part_name, {})
            total_length = part_data.get("Total length", 0)
            # La mortaja de la parte anterior determina cuánto se inserta la espiga de la parte actual.
            # Para el headjoint, su "Mortise length" es la espiga que se inserta en el cuerpo.
            # Para las otras partes, su "Mortise length" es la espiga que ellas insertan en la parte anterior.
            # La lógica original parece restar la "Mortise length" de la *propia* parte para calcular el avance.
            # Esto es correcto si "Mortise length" se interpreta como la longitud de la espiga de *esa* parte.
            mortise_length = part_data.get("Mortise length", 0)
            effective_lengths[part_name] = total_length - mortise_length

        for i, part in enumerate(FLUTE_PARTS_ORDER):
            if i > 0:
                # La posición actual es la suma de las longitudes efectivas de las partes anteriores.
                # La lógica original tenía una forma más compleja de calcular esto,
                # esta es una simplificación asumiendo que "Mortise length" es la espiga de la parte actual.
                # Revalidar esta lógica con el significado exacto de "Mortise length" en los JSON.
                # El cálculo original para current_position se mantiene por ahora, pero podría simplificarse si
                # "Mortise length" consistentemente significa "longitud de la espiga de esta parte".
                # Para mantener la funcionalidad original exacta:
                if part == "left":
                    current_position = self.data["headjoint"].get("Total length", 0) - self.data["headjoint"].get("Mortise length", 0)
                elif part == "right":
                    current_position = (self.data["headjoint"].get("Total length", 0) - self.data["headjoint"].get("Mortise length", 0) +
                                        self.data["left"].get("Total length", 0) - self.data["right"].get("Mortise length", 0)) # Originalmente usaba mortise de right
                elif part == "foot":
                    current_position = (self.data["headjoint"].get("Total length", 0) - self.data["headjoint"].get("Mortise length", 0) +
                                        self.data["left"].get("Total length", 0) + self.data["right"].get("Total length", 0) -
                                        self.data["right"].get("Mortise length", 0) - self.data["foot"].get("Mortise length", 0)) # Originalmente usaba mortise de right y foot
            
            positions = [item["position"] for item in self.data[part]["measurements"]]
            diameters = [item["diameter"] for item in self.data[part]["measurements"]]
            mortise_length = self.data[part].get("Mortise length", 0)
            total_length = self.data[part].get("Total length", 0)

            for pos, diam in zip(positions, diameters):
                adjusted_pos = pos + current_position
                if part == "headjoint" and pos >= total_length - mortise_length:
                    continue
                if part in ["right", "foot"] and pos <= mortise_length:
                    continue
                combined_measurements.append({"position": adjusted_pos, "diameter": diam})
        return combined_measurements

    def compute_acoustic_analysis(self, fing_chart_file: str, temperature: float) -> None:
        """
        Realiza un análisis acústico para las notas en la tabla de digitaciones.

        Args:
            fing_chart_file (str): Ruta al archivo de digitaciones.
            temperature (float): Temperatura en °C.
        """
        try:
            combined_measurements = self.combine_measurements()
            geom = [[m["position"] / 1000, m["diameter"] / 2000] for m in combined_measurements]
            side_holes = [['label', 'position', 'chimney', 'radius', 'radius_out']]
            emb_label = "embouchure"
            embouchure_hole = self.data["headjoint"].get("Holes position", [])[0]
            embouchure_radius_mm = self.data["headjoint"].get("Holes diameter", [])[0] / 2
            embouchure_chimney = self.data["headjoint"].get("Holes chimney", [DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT * 1000])[0] / 1000 # Convert mm to m
            embouchure_radius_out_mm = self.data["headjoint"].get("Holes diameter_out", [embouchure_radius_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR])[0] / 2
            side_holes.append([emb_label, embouchure_hole / 1000, embouchure_chimney, embouchure_radius_mm / 1000, embouchure_radius_out_mm / 1000])

            carrier_left = self.data["headjoint"].get("Total length", 0) - self.data["headjoint"].get("Mortise length", 0)
            for i, hole_pos in enumerate(self.data["left"].get("Holes position", [])):
                hole_radius_mm = self.data["left"].get("Holes diameter", [])[i] / 2
                hole_chimney = self.data["left"].get("Holes chimney", [DEFAULT_CHIMNEY_HEIGHT * 1000]*len(self.data["left"]["Holes position"]))[i] / 1000
                hole_radius_out_mm = self.data["left"].get("Holes diameter_out", [hole_radius_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR]*len(self.data["left"]["Holes position"]))[i] / 2
                side_holes.append([f"hole{i+1}", (carrier_left + hole_pos) / 1000, hole_chimney, hole_radius_mm / 1000, hole_radius_out_mm / 1000])

            carrier_right = carrier_left + self.data["left"].get("Total length", 0) - self.data["right"].get("Mortise length", 0)
            for i, hole_pos in enumerate(self.data["right"].get("Holes position", [])):
                hole_radius_mm = self.data["right"].get("Holes diameter", [])[i] / 2
                hole_chimney = self.data["right"].get("Holes chimney", [DEFAULT_CHIMNEY_HEIGHT * 1000]*len(self.data["right"]["Holes position"]))[i] / 1000
                hole_radius_out_mm = self.data["right"].get("Holes diameter_out", [hole_radius_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR]*len(self.data["right"]["Holes position"]))[i] / 2
                side_holes.append([f"hole{i+4}", (hole_pos + carrier_right) / 1000, hole_chimney, hole_radius_mm / 1000, hole_radius_out_mm / 1000])

            carrier_foot = carrier_right + self.data["right"].get("Total length", 0) - self.data["foot"].get("Mortise length", 0)
            foot_hole_pos = self.data["foot"].get("Holes position", [])[0]
            foot_radius_mm = self.data["foot"].get("Holes diameter", [])[0] / 2
            foot_chimney = self.data["foot"].get("Holes chimney", [DEFAULT_CHIMNEY_HEIGHT * 1000])[0] / 1000
            foot_radius_out_mm = self.data["foot"].get("Holes diameter_out", [foot_radius_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR])[0] / 2
            side_holes.append([f"hole7", (foot_hole_pos + carrier_foot) / 1000, foot_chimney, foot_radius_mm / 1000, foot_radius_out_mm / 1000])
            Rw = embouchure_radius_mm / 1000 # Convert mm to m for player section
            freq = np.arange(100, 3000, 2)
            player_trans = Player("FLUTE")
            player_trans.update_curve("radiation_category", "infinite_flanged")
            player_trans.update_curve("section", np.pi * Rw**2)
            with Path(fing_chart_file).open("r") as f:
                lines = f.readlines()
            fing_chart = [line.strip().split() for line in lines]
            notes = fing_chart[0][1:]
            for note in notes:
                self.acoustic_analysis[note] = ImpedanceComputation(
                    freq, geom, side_holes, fing_chart,
                    player=player_trans,
                    note=note,
                    temperature=temperature,
                    interp=True,
                    source_location=emb_label
                )
        except Exception as e:
            logger.exception("Error en compute_acoustic_analysis: %s", e)