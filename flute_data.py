import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from openwind import Player, ImpedanceComputation, InstrumentGeometry

from notion_utils import get_json_files_from_notion

import logging

logger = logging.getLogger(__name__)

class FluteData:
    def __init__(self, source: str, notion_token: str = None, database_id: str = None,
                 fing_chart_file: str = "data_json/traverso_fingerchart.txt", temperature: float = 20,
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
        parts = ["headjoint", "left", "right", "foot"]
        for part in parts:
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
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as headjoint_file, \
                 tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as left_file, \
                 tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as right_file, \
                 tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as foot_file:
                json.dump(headjoint_data, headjoint_file, indent=4)
                json.dump(left_data, left_file, indent=4)
                json.dump(right_data, right_file, indent=4)
                json.dump(foot_data, foot_file, indent=4)
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
        part_order = ["headjoint", "left", "right", "foot"]

        for part in part_order:
            if part == "left":
                current_position = self.data["headjoint"].get("Total length", 0) - self.data["headjoint"].get("Mortise length", 0)
            elif part == "right":
                current_position = (self.data["headjoint"].get("Total length", 0) - self.data["headjoint"].get("Mortise length", 0) +
                                    self.data["left"].get("Total length", 0) - self.data["right"].get("Mortise length", 0))
            elif part == "foot":
                current_position = (self.data["headjoint"].get("Total length", 0) - self.data["headjoint"].get("Mortise length", 0) +
                                    self.data["left"].get("Total length", 0) + self.data["right"].get("Total length", 0) -
                                    self.data["right"].get("Mortise length", 0) - self.data["foot"].get("Mortise length", 0))

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
            embouchure_diameter = self.data["headjoint"].get("Holes diameter", [])[0]
            side_holes.append([emb_label, embouchure_hole / 1000, 5e-3, embouchure_diameter / 2000, 5e-3])
            carrier_left = self.data["headjoint"].get("Total length", 0) - self.data["headjoint"].get("Mortise length", 0)
            for i, hole_pos in enumerate(self.data["left"].get("Holes position", [])):
                hole_diam = self.data["left"].get("Holes diameter", [])[i]
                side_holes.append([f"hole{i+1}", (carrier_left + hole_pos) / 1000, 3e-3, hole_diam / 2000, 4e-3])
            carrier_right = carrier_left + self.data["left"].get("Total length", 0) - self.data["right"].get("Mortise length", 0)
            for i, hole_pos in enumerate(self.data["right"].get("Holes position", [])):
                hole_diam = self.data["right"].get("Holes diameter", [])[i]
                side_holes.append([f"hole{i+4}", (hole_pos + carrier_right) / 1000, 3e-3, hole_diam / 2000, 4e-3])
            carrier_foot = carrier_right + self.data["right"].get("Total length", 0) - self.data["foot"].get("Mortise length", 0)
            foot_hole = self.data["foot"].get("Holes position", [])[0]
            foot_diameter = self.data["foot"].get("Holes diameter", [])[0]
            side_holes.append([f"hole7", (foot_hole + carrier_foot) / 1000, 3e-3, foot_diameter / 2000, 4e-3])
            Rw = embouchure_diameter / 2
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