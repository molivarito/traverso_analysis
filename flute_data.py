import json
import tempfile # Not currently used, but kept for potential future use
from pathlib import Path
from typing import Any, Dict, List, Union, Optional # Added Union, Optional
import numpy as np
from openwind import Player, ImpedanceComputation, InstrumentGeometry # type: ignore

# Assuming notion_utils.py is in the same directory or PYTHONPATH
from notion_utils import get_json_files_from_notion
# Assuming constants.py is in the same directory or PYTHONPATH
from constants import (
    FLUTE_PARTS_ORDER, DEFAULT_CHIMNEY_HEIGHT,
    DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT, DEFAULT_HOLE_RADIUS_OUT_FACTOR
)

import logging

logger = logging.getLogger(__name__)

# Default fingering chart path logic
SCRIPT_DIR_FLUTE_DATA = Path(__file__).resolve().parent
# Try to find data_json relative to this script's parent, or directly if that fails
DEFAULT_FING_CHART_RELATIVE_PATH = Path("data_json") / "traverso_fingerchart.txt"
DEFAULT_FING_CHART_PATH_OPTION1 = SCRIPT_DIR_FLUTE_DATA.parent / DEFAULT_FING_CHART_RELATIVE_PATH
DEFAULT_FING_CHART_PATH_OPTION2 = SCRIPT_DIR_FLUTE_DATA / DEFAULT_FING_CHART_RELATIVE_PATH # If constants.py is not in same dir
DEFAULT_FING_CHART_PATH_OPTION3 = DEFAULT_FING_CHART_RELATIVE_PATH # Current working directory

if DEFAULT_FING_CHART_PATH_OPTION1.exists():
    DEFAULT_FING_CHART_PATH = str(DEFAULT_FING_CHART_PATH_OPTION1)
elif DEFAULT_FING_CHART_PATH_OPTION2.exists():
    DEFAULT_FING_CHART_PATH = str(DEFAULT_FING_CHART_PATH_OPTION2)
else:
    DEFAULT_FING_CHART_PATH = str(DEFAULT_FING_CHART_PATH_OPTION3)


class FluteData:
    def __init__(self,
                 source: Union[str, Dict[str, Any]],
                 source_name: Optional[str] = None,
                 notion_token: Optional[str] = None,
                 database_id: Optional[str] = None,
                 fing_chart_file: str = DEFAULT_FING_CHART_PATH,
                 temperature: float = 20,
                 la_frequency: float = 415.0,
                 skip_acoustic_analysis: bool = False) -> None: # Nuevo parámetro

        self.data: Dict[str, Any] = {}
        self.acoustic_analysis: Dict[str, ImpedanceComputation] = {} # More specific type
        self.instrument: Dict[str, Any] = {}
        self.combined_measurements: List[Dict[str, float]] = []
        self.la_frequency: float = la_frequency
        self.flute_model: str = "UnknownFlute"
        self.temperature = temperature
        self.fing_chart_file_path = fing_chart_file
        self._skip_acoustic_analysis = skip_acoustic_analysis # Guardar el estado

        try:
            if isinstance(source, str): # Source is a directory path string
                self.flute_model = Path(source).name if Path(source).is_dir() else source
                if source_name: # If source_name is provided, it overrides the path-derived name
                    self.flute_model = source_name

                if notion_token and database_id: # Assuming source (path) is used as filter name for Notion
                    self._read_json_data_from_notion(notion_token, database_id, source)
                else:
                    self._read_json_data_from_files(source)
                # Ensure "Flute Model" is in data after reading from files/Notion
                if "Flute Model" not in self.data or not self.data.get("Flute Model"):
                    self.data["Flute Model"] = self.flute_model
                else: # If loaded data has a "Flute Model", ensure self.flute_model matches it.
                    self.flute_model = self.data["Flute Model"]

            elif isinstance(source, dict): # Source is a pre-loaded data dictionary
                self.data = source # Assign directly
                # Determine flute_model: use source_name, then "Flute Model" in dict, then default
                if source_name:
                    self.flute_model = source_name
                elif "Flute Model" in self.data and self.data["Flute Model"]:
                    self.flute_model = str(self.data["Flute Model"])
                else:
                    self.flute_model = "InMemoryFlute"

                # Ensure "Flute Model" key exists and is consistent in self.data
                self.data["Flute Model"] = self.flute_model
                logger.info(f"FluteData initialized from dictionary. Model: {self.flute_model}")

            else:
                raise ValueError("La fuente (source) para FluteData debe ser una ruta de directorio (str) o un diccionario de datos (dict).")

            # Common initialization steps
            try:
                # Ensure fing_chart_file path is valid before opening
                fing_chart_path_obj = Path(fing_chart_file)
                if not fing_chart_path_obj.is_file():
                    logger.error(f"Archivo de digitaciones no es un archivo válido o no existe: {fing_chart_file} para {self.flute_model}")
                    raise FileNotFoundError(f"Archivo de digitaciones no encontrado o inválido: {fing_chart_file}")

                with fing_chart_path_obj.open("r", encoding='utf-8') as f:
                    header_line = f.readline().strip()
                tokens = header_line.split()
                note_names = tokens[1:] if len(tokens) > 1 else []
                semitone_mapping = {"D": -7, "E": -5, "Fs": -3, "G": -2, "A": 0, "B": 2, "Cs": 4}
                self.finger_frequencies: Dict[str, float] = {}
                for note in note_names:
                    n = semitone_mapping.get(note)
                    if n is not None:
                        self.finger_frequencies[note] = self.la_frequency * (2 ** (n / 12.0))
                    else:
                        logger.warning(f"Nota '{note}' del archivo de digitación no encontrada en semitone_mapping para {self.flute_model}.")
            except FileNotFoundError as e_fnf:
                logger.error(f"Archivo de digitaciones no encontrado en: {fing_chart_file} para {self.flute_model}. Error: {e_fnf}")
                self.finger_frequencies = {} # Ensure it's initialized
            except Exception as e_fc:
                logger.error(f"Error al leer o procesar el archivo de digitaciones '{fing_chart_file}' para {self.flute_model}: {e_fc}")
                self.finger_frequencies = {}

            self.combined_measurements = self.combine_measurements()
            if not self._skip_acoustic_analysis: # Condición para ejecutar el análisis
                self.compute_acoustic_analysis(self.fing_chart_file_path, self.temperature)

        except Exception as e_init:
            logger.exception(f"Error al inicializar FluteData para '{self.flute_model}': {e_init}")
            raise ValueError(f"Error al procesar los datos de la flauta '{self.flute_model}': {e_init}")


    def _read_json_data_from_files(self, base_dir: str) -> None:
        logger.info(f"Leyendo datos JSON desde archivos en: {base_dir} para {self.flute_model}")
        loaded_data_for_parts: Dict[str, Any] = {}
        for part in FLUTE_PARTS_ORDER:
            json_path = Path(base_dir) / f"{part}.json"
            try:
                with json_path.open('r', encoding='utf-8') as file:
                    loaded_data_for_parts[part] = json.load(file)
                logger.debug(f"Cargado {json_path}")
            except FileNotFoundError:
                logger.error(f"No se encontró el archivo JSON para la parte '{part}': {json_path}")
                raise FileNotFoundError(f"No se encontró el archivo: {json_path} para la flauta {self.flute_model}")
            except json.JSONDecodeError as e:
                logger.error(f"Error al decodificar JSON en '{json_path}': {e.msg} (línea {e.lineno}, col {e.colno})")
                raise json.JSONDecodeError(f"Error al decodificar JSON: {json_path} - {e.msg}", e.doc, e.pos)
            except Exception as e_gen:
                logger.error(f"Error inesperado cargando '{json_path}': {e_gen}")
                raise
        self.data.update(loaded_data_for_parts) # Update self.data with loaded parts

    def _read_json_data_from_notion(self, notion_token: str, database_id: str, flute_name_filter: str) -> None:
        logger.info(f"Leyendo datos JSON desde Notion para la flauta: {flute_name_filter}")
        try:
            retrieved_data_map = get_json_files_from_notion(
                notion_token, database_id, flute_name_filter
            )
            missing_parts = [part for part in FLUTE_PARTS_ORDER if part not in retrieved_data_map]
            if missing_parts:
                err_msg = f"No se pudieron recuperar todas las partes desde Notion para '{flute_name_filter}'. Faltan: {', '.join(missing_parts)}"
                logger.error(err_msg)
                raise ValueError(err_msg)

            loaded_data_for_parts: Dict[str, Any] = {}
            for part_name in FLUTE_PARTS_ORDER:
                loaded_data_for_parts[part_name] = retrieved_data_map[part_name]
                logger.debug(f"Datos de Notion cargados para la parte: {part_name}")
            self.data.update(loaded_data_for_parts)

        except Exception as e:
            logger.exception(f"Error al obtener datos de Notion para '{flute_name_filter}': {e}")
            raise ValueError(f"Error al obtener datos de Notion: {e}")


    def combine_measurements(self) -> List[Dict[str, float]]:
        logger.debug(f"Combinando mediciones para {self.flute_model}")
        combined_measurements = []
        current_position = 0.0

        # Ensure all parts are at least an empty dict in part_data_map if not in self.data
        part_data_map = {part_name: self.data.get(part_name, {}) for part_name in FLUTE_PARTS_ORDER}

        for i, part_name in enumerate(FLUTE_PARTS_ORDER):
            part_specific_data = part_data_map.get(part_name)
            if not part_specific_data:
                logger.warning(f"No hay datos para la parte '{part_name}' en {self.flute_model} al combinar mediciones. Saltando esta parte.")
                continue

            if i > 0: # Adjust current_position for parts after headjoint
                headjoint_data = part_data_map.get(FLUTE_PARTS_ORDER[0], {})
                headjoint_total_length = headjoint_data.get("Total length", 0.0)
                headjoint_mortise = headjoint_data.get("Mortise length", 0.0)

                left_data = part_data_map.get(FLUTE_PARTS_ORDER[1], {})
                left_total_length = left_data.get("Total length", 0.0)
                left_mortise = left_data.get("Mortise length", 0.0) # Tenon of the left part

                right_data = part_data_map.get(FLUTE_PARTS_ORDER[2], {})
                right_total_length = right_data.get("Total length", 0.0)
                right_mortise = right_data.get("Mortise length", 0.0) # Tenon of the right part

                # foot_data and foot_mortise_val are not directly needed here if calculated sequentially

                if part_name == FLUTE_PARTS_ORDER[1]: # left
                    current_position = headjoint_total_length - headjoint_mortise
                elif part_name == FLUTE_PARTS_ORDER[2]: # right
                    # Start of right part is end of headjoint body + length of left part body
                    current_position = (headjoint_total_length - headjoint_mortise) + \
                                       (left_total_length - left_mortise)
                elif part_name == FLUTE_PARTS_ORDER[3]: # foot
                    # Start of foot part is end of headjoint body + length of left part body + length of right part body
                    current_position = (headjoint_total_length - headjoint_mortise) + \
                                       (left_total_length - left_mortise) + \
                                       (right_total_length - right_mortise)


            measurements_list = part_specific_data.get("measurements", [])
            if not isinstance(measurements_list, list): # Ensure it's a list
                logger.warning(f"Mediciones para la parte '{part_name}' no es una lista. Saltando mediciones de esta parte.")
                measurements_list = []

            positions = [item.get("position", 0.0) for item in measurements_list]
            diameters = [item.get("diameter", 0.0) for item in measurements_list]

            part_mortise_length = part_specific_data.get("Mortise length", 0.0)
            part_total_length = part_specific_data.get("Total length", 0.0)

            for pos, diam in zip(positions, diameters):
                adjusted_pos = pos + current_position
                # Filter conditions
                # Ensure part_total_length - part_mortise_length is positive or handle zero case if mortise can be >= total_length
                filter_threshold = part_total_length - part_mortise_length
                if part_name == FLUTE_PARTS_ORDER[0] and filter_threshold > 0 and pos >= filter_threshold:
                    continue
                elif part_name == FLUTE_PARTS_ORDER[0] and filter_threshold <= 0: # Mortise too large or total_length zero
                    logger.debug(f"Headjoint {self.flute_model}: mortise {part_mortise_length} >= total_length {part_total_length}. Todas las mediciones se incluirán.")
                    # No filtrar si el umbral no es válido

                if part_name in [FLUTE_PARTS_ORDER[2], FLUTE_PARTS_ORDER[3]] and pos <= part_mortise_length:
                    continue
                combined_measurements.append({"position": adjusted_pos, "diameter": diam})

        logger.debug(f"Mediciones combinadas generadas para {self.flute_model} con {len(combined_measurements)} puntos.")
        if not combined_measurements and any(part_data_map.values()): # Si hay datos de partes pero no mediciones combinadas
            logger.warning(f"La lista de mediciones combinadas está vacía para {self.flute_model}, aunque hay datos de partes. Verifique la lógica de combinación y los datos de entrada.")
        return combined_measurements

    def compute_acoustic_analysis(self, fing_chart_file: str, temperature: float) -> None:
        logger.debug(f"Calculando análisis acústico para {self.flute_model} a {temperature}°C.")
        self.acoustic_analysis = {} # Reset before filling

        try:
            if not self.combined_measurements:
                logger.warning(f"No hay mediciones combinadas para {self.flute_model}, saltando análisis acústico.")
                return

            geom = [[m["position"] / 1000.0, m["diameter"] / 2000.0] for m in self.combined_measurements]

            side_holes_data = []
            Rw = 0.006 # Default embouchure radius in meters

            headjoint_data = self.data.get(FLUTE_PARTS_ORDER[0], {})
            emb_hole_positions = headjoint_data.get("Holes position", [])
            emb_hole_diameters = headjoint_data.get("Holes diameter", [])
            # Use .get with defaults for chimney and diameter_out to avoid errors if keys are missing
            emb_hole_chimneys_list = headjoint_data.get("Holes chimney", [])
            emb_hole_diam_out_list = headjoint_data.get("Holes diameter_out", [])


            if emb_hole_positions and emb_hole_diameters:
                emb_pos_mm = emb_hole_positions[0]
                emb_diam_mm = emb_hole_diameters[0]
                # Provide defaults if chimney/diam_out lists are shorter or missing
                emb_chim_mm = emb_hole_chimneys_list[0] if emb_hole_chimneys_list else DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT * 1000
                emb_diam_o_mm = emb_hole_diam_out_list[0] if emb_hole_diam_out_list else emb_diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR

                side_holes_data.append([
                    "embouchure",
                    emb_pos_mm / 1000.0,
                    emb_chim_mm / 1000.0,
                    (emb_diam_mm / 2.0) / 1000.0,
                    (emb_diam_o_mm / 2.0) / 1000.0
                ])
                Rw = (emb_diam_mm / 2.0) / 1000.0
            else:
                logger.warning(f"Datos de embocadura insuficientes o ausentes para {self.flute_model}. Usando Rw por defecto ({Rw}m).")

            # Corrected logic for current_offset_mm based on the structure of combine_measurements
            current_offset_mm = 0.0 # This will be the start of the current part being processed for holes
            part_idx_offset = 0     # For global hole numbering (hole1, hole2...)

            # Headjoint holes are already handled (embouchure) if they were part of side_holes_data.
            # Here, we calculate offsets for body parts and then add their holes.
            # The first offset (start of 'left' part)
            if FLUTE_PARTS_ORDER[0] in self.data and self.data[FLUTE_PARTS_ORDER[0]]:
                 hj_total = self.data[FLUTE_PARTS_ORDER[0]].get("Total length", 0.0)
                 hj_mortise = self.data[FLUTE_PARTS_ORDER[0]].get("Mortise length", 0.0)
                 offset_after_headjoint = hj_total - hj_mortise
            else:
                 offset_after_headjoint = 0.0


            for part_enum_idx, part_name in enumerate(FLUTE_PARTS_ORDER):
                if part_name == FLUTE_PARTS_ORDER[0]: # Headjoint (embouchure already handled)
                    continue # Skip to next part for tone holes

                part_data = self.data.get(part_name, {})
                if not part_data:
                    logger.warning(f"No data for part '{part_name}' when calculating hole positions for {self.flute_model}.")
                    continue
                
                # Determine the absolute starting position of the current part
                if part_name == FLUTE_PARTS_ORDER[1]: # 'left'
                    current_offset_mm = offset_after_headjoint
                elif part_name == FLUTE_PARTS_ORDER[2]: # 'right'
                    l_data = self.data.get(FLUTE_PARTS_ORDER[1], {})
                    r_data_mortise = part_data.get("Mortise length", 0.0) # Mortise of the current 'right' part
                    current_offset_mm = offset_after_headjoint + l_data.get("Total length", 0.0) - r_data_mortise
                elif part_name == FLUTE_PARTS_ORDER[3]: # 'foot'
                    l_data = self.data.get(FLUTE_PARTS_ORDER[1], {})
                    r_data = self.data.get(FLUTE_PARTS_ORDER[2], {})
                    f_data_mortise = part_data.get("Mortise length", 0.0) # Mortise of the current 'foot' part
                    current_offset_mm = (offset_after_headjoint +
                                         l_data.get("Total length", 0.0) +
                                         r_data.get("Total length", 0.0) - r_data.get("Mortise length", 0.0) -
                                         f_data_mortise)
                else: # Should not happen with FLUTE_PARTS_ORDER
                    current_offset_mm = 0.0


                holes_pos = part_data.get("Holes position", [])
                num_holes_in_part = len(holes_pos)
                holes_diam = part_data.get("Holes diameter", [7.0]*num_holes_in_part)
                holes_chimney_list = part_data.get("Holes chimney", [])
                holes_diam_out_list = part_data.get("Holes diameter_out", [])

                for i, hole_pos_mm in enumerate(holes_pos):
                    diam_mm = holes_diam[i] if i < len(holes_diam) else 7.0
                    chimney_mm = holes_chimney_list[i] if i < len(holes_chimney_list) else DEFAULT_CHIMNEY_HEIGHT * 1000
                    diam_out_mm = holes_diam_out_list[i] if i < len(holes_diam_out_list) else diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR

                    side_holes_data.append([
                        f"hole{part_idx_offset + i + 1}",
                        (current_offset_mm + hole_pos_mm) / 1000.0,
                        chimney_mm / 1000.0,
                        (diam_mm / 2.0) / 1000.0,
                        (diam_out_mm / 2.0) / 1000.0
                    ])
                part_idx_offset += num_holes_in_part

            side_holes_header = [['label', 'position', 'chimney', 'radius', 'radius_out']]
            side_holes_for_openwind = side_holes_header + side_holes_data
            logger.debug(f"Side holes data para openwind ({self.flute_model}): {len(side_holes_data)} agujeros tónicos definidos.")

            freq_range = np.arange(100, 3000, 2.0)
            player = Player("FLUTE")
            player.update_curve("radiation_category", "infinite_flanged")
            player.update_curve("section", np.pi * Rw**2)

            fing_chart_path_obj = Path(fing_chart_file)
            if not fing_chart_path_obj.is_file():
                logger.error(f"Archivo de digitaciones '{fing_chart_file}' no encontrado o inválido para análisis acústico de {self.flute_model}.")
                return

            with fing_chart_path_obj.open("r", encoding='utf-8') as f:
                lines = f.readlines()
            fing_chart_parsed = [line.strip().split() for line in lines if line.strip()]

            if not fing_chart_parsed or not fing_chart_parsed[0] or len(fing_chart_parsed[0]) <=1 :
                logger.error(f"Archivo de digitaciones '{fing_chart_file}' está vacío o malformado para {self.flute_model}.")
                return

            notes_from_chart = fing_chart_parsed[0][1:]
            logger.debug(f"Notas del archivo de digitación para {self.flute_model}: {notes_from_chart}")

            for note in notes_from_chart:
                if not note: continue
                logger.debug(f"Calculando impedancia para nota: {note} en {self.flute_model}")
                try:
                    self.acoustic_analysis[note] = ImpedanceComputation(
                        freq_range, geom, side_holes_for_openwind, fing_chart_parsed,
                        player=player,
                        note=note,
                        temperature=temperature,
                        interp=True,
                        source_location="embouchure"
                    )
                    logger.info(f"Análisis acústico completado para nota {note} en {self.flute_model}")
                except Exception as e_imp:
                    logger.error(f"Error en ImpedanceComputation para nota '{note}' en {self.flute_model}. Datos de entrada: "
                                 f"Geom (primeros 5): {geom[:5]}, SideHoles (primeros 5): {side_holes_for_openwind[:5]}. Error: {e_imp}")

        except Exception as e_main:
            logger.exception(f"Error mayor en compute_acoustic_analysis para {self.flute_model}: {e_main}")
            self.acoustic_analysis = {} # Ensure it's empty on major failure