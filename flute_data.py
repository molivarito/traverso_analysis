import json
import tempfile # Not currently used, but kept for potential future use
from pathlib import Path
from typing import Any, Dict, List
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
DEFAULT_FING_CHART_PATH = SCRIPT_DIR_FLUTE_DATA.parent / "data_json" / "traverso_fingerchart.txt"
if not DEFAULT_FING_CHART_PATH.exists(): # Fallback if the parent/data_json structure isn't found
    DEFAULT_FING_CHART_PATH = Path("data_json") / "traverso_fingerchart.txt"


class FluteData:
    def __init__(self, source: str, notion_token: str = None, database_id: str = None,
                 fing_chart_file: str = str(DEFAULT_FING_CHART_PATH), temperature: float = 20,
                 la_frequency: float = 415) -> None:
        self.data: Dict[str, Any] = {}
        self.acoustic_analysis: Dict[str, ImpedanceComputation] = {} # More specific type
        self.instrument: Dict[str, Any] = {} # Could be more specific if structure is known
        self.combined_measurements: List[Dict[str, float]] = []
        self.la_frequency: float = la_frequency
        self.flute_model = Path(source).name if Path(source).is_dir() else source
        self.temperature = temperature # Store temperature for potential later use (e.g. speed of sound)
        self.fing_chart_file_path = fing_chart_file # Store for reference

        try:
            if notion_token and database_id:
                self._read_json_data_from_notion(notion_token, database_id, source)
            else:
                self._read_json_data_from_files(source)
            self.data["Flute Model"] = self.flute_model

            try:
                with Path(fing_chart_file).open("r") as f:
                    header_line = f.readline().strip()
                tokens = header_line.split()
                note_names = tokens[1:]
                semitone_mapping = {"D": -7, "E": -5, "Fs": -3, "G": -2, "A": 0, "B": 2, "Cs": 4} # Example, adjust as needed
                self.finger_frequencies: Dict[str, float] = {}
                for note in note_names:
                    n = semitone_mapping.get(note)
                    if n is not None:
                        self.finger_frequencies[note] = self.la_frequency * (2 ** (n / 12.0))
                    else:
                        # self.finger_frequencies[note] = None # type: ignore # Or log warning/error
                        logger.warning(f"Nota '{note}' del archivo de digitación no encontrada en semitone_mapping.")
            except FileNotFoundError:
                logger.error(f"Archivo de digitaciones no encontrado en: {fing_chart_file}")
                self.finger_frequencies = {}
            except Exception as e:
                logger.error(f"Error al leer o procesar el archivo de digitaciones '{fing_chart_file}': {e}")
                self.finger_frequencies = {}

            self.combined_measurements = self.combine_measurements()
            self.compute_acoustic_analysis(fing_chart_file, temperature)
        except Exception as e:
            logger.exception(f"Error al inicializar FluteData para '{source}': {e}")
            # Re-raise para que la GUI pueda capturarlo y mostrar un mensaje
            raise ValueError(f"Error al procesar los datos de la flauta '{self.flute_model}': {e}")


    def _read_json_data_from_files(self, base_dir: str) -> None:
        logger.info(f"Leyendo datos JSON desde archivos en: {base_dir}")
        for part in FLUTE_PARTS_ORDER:
            json_path = Path(base_dir) / f"{part}.json"
            try:
                with json_path.open('r', encoding='utf-8') as file: # Especificar encoding
                    self.data[part] = json.load(file)
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

    def _read_json_data_from_notion(self, notion_token: str, database_id: str, flute_name_filter: str) -> None:
        logger.info(f"Leyendo datos JSON desde Notion para la flauta: {flute_name_filter}")
        try:
            # Asumiendo que get_json_files_from_notion devuelve un diccionario
            # donde las claves son los nombres de las partes (headjoint, left, etc.)
            # y los valores son los datos JSON deserializados (diccionarios Python).
            retrieved_data_map = get_json_files_from_notion(
                notion_token, database_id, flute_name_filter
            )
            
            # Verificar que se obtuvieron todos los datos necesarios según FLUTE_PARTS_ORDER
            missing_parts = [part for part in FLUTE_PARTS_ORDER if part not in retrieved_data_map]
            if missing_parts:
                logger.error(f"No se pudieron recuperar todas las partes desde Notion para '{flute_name_filter}'. Faltan: {', '.join(missing_parts)}")
                raise ValueError(f"Datos incompletos desde Notion para '{flute_name_filter}'. Faltan: {', '.join(missing_parts)}")

            # Asignar los datos recuperados a self.data
            for part_name in FLUTE_PARTS_ORDER:
                self.data[part_name] = retrieved_data_map[part_name]
                logger.debug(f"Datos de Notion cargados para la parte: {part_name}")

        except Exception as e:
            logger.exception(f"Error al obtener datos de Notion para '{flute_name_filter}': {e}")
            raise ValueError(f"Error al obtener datos de Notion: {e}")


    def combine_measurements(self) -> List[Dict[str, float]]:
        logger.debug(f"Combinando mediciones para {self.flute_model}")
        combined_measurements = []
        current_position = 0.0
        
        part_data_map = {part_name: self.data.get(part_name, {}) for part_name in FLUTE_PARTS_ORDER}

        for i, part_name in enumerate(FLUTE_PARTS_ORDER):
            part_specific_data = part_data_map[part_name]
            if not part_specific_data:
                logger.warning(f"No hay datos para la parte '{part_name}' en {self.flute_model} al combinar mediciones.")
                continue # Saltar esta parte si no hay datos

            if i > 0:
                # Lógica original de ajuste de current_position
                headjoint_total_length = part_data_map[FLUTE_PARTS_ORDER[0]].get("Total length", 0.0)
                headjoint_mortise = part_data_map[FLUTE_PARTS_ORDER[0]].get("Mortise length", 0.0) # CORREGIDO EL TYPO AQUÍ
                
                left_data = part_data_map.get(FLUTE_PARTS_ORDER[1], {})
                left_total_length = left_data.get("Total length", 0.0)
                
                right_data = part_data_map.get(FLUTE_PARTS_ORDER[2], {})
                right_total_length = right_data.get("Total length", 0.0)
                right_mortise = right_data.get("Mortise length", 0.0)
                
                foot_data = part_data_map.get(FLUTE_PARTS_ORDER[3], {})
                foot_mortise = foot_data.get("Mortise length", 0.0)

                if part_name == FLUTE_PARTS_ORDER[1]: # left
                    current_position = headjoint_total_length - headjoint_mortise
                elif part_name == FLUTE_PARTS_ORDER[2]: # right
                    current_position = (headjoint_total_length - headjoint_mortise +
                                        left_total_length - right_mortise)
                elif part_name == FLUTE_PARTS_ORDER[3]: # foot
                    current_position = (headjoint_total_length - headjoint_mortise +
                                        left_total_length + right_total_length -
                                        right_mortise - foot_mortise)
            
            positions = [item.get("position", 0.0) for item in part_specific_data.get("measurements", [])]
            diameters = [item.get("diameter", 0.0) for item in part_specific_data.get("measurements", [])]
            
            part_mortise_length = part_specific_data.get("Mortise length", 0.0)
            part_total_length = part_specific_data.get("Total length", 0.0)

            for pos, diam in zip(positions, diameters):
                adjusted_pos = pos + current_position
                if part_name == FLUTE_PARTS_ORDER[0] and pos >= part_total_length - part_mortise_length:
                    continue
                if part_name in [FLUTE_PARTS_ORDER[2], FLUTE_PARTS_ORDER[3]] and pos <= part_mortise_length:
                    continue
                combined_measurements.append({"position": adjusted_pos, "diameter": diam})
        
        logger.debug(f"Mediciones combinadas generadas con {len(combined_measurements)} puntos.")
        return combined_measurements

    def compute_acoustic_analysis(self, fing_chart_file: str, temperature: float) -> None:
        logger.debug(f"Calculando análisis acústico para {self.flute_model} a {temperature}°C.")
        try:
            if not self.combined_measurements:
                logger.warning(f"No hay mediciones combinadas para {self.flute_model}, saltando análisis acústico.")
                return
            
            geom = [[m["position"] / 1000.0, m["diameter"] / 2000.0] for m in self.combined_measurements]
            
            side_holes_data = []
            Rw = 0.006 # Default embouchure radius in meters if not found

            # Embouchure (del headjoint)
            headjoint_data = self.data.get(FLUTE_PARTS_ORDER[0], {})
            emb_hole_info_list = headjoint_data.get("Holes position", []) # Es una lista
            if emb_hole_info_list:
                embouchure_pos_mm = emb_hole_info_list[0]
                embouchure_diam_mm = headjoint_data.get("Holes diameter", [12.0])[0] # Default si no está
                embouchure_chimney_mm = headjoint_data.get("Holes chimney", [DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT * 1000])[0]
                embouchure_diam_out_mm = headjoint_data.get("Holes diameter_out", [embouchure_diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR])[0]
                
                side_holes_data.append([
                    "embouchure",
                    embouchure_pos_mm / 1000.0,
                    embouchure_chimney_mm / 1000.0,
                    (embouchure_diam_mm / 2.0) / 1000.0,
                    (embouchure_diam_out_mm / 2.0) / 1000.0
                ])
                Rw = (embouchure_diam_mm / 2.0) / 1000.0
            else:
                logger.warning(f"No se encontraron datos de agujero de embocadura para {self.flute_model}. Usando Rw por defecto.")

            # Cálculo de offsets para las partes (debe ser consistente con combine_measurements)
            current_offset_mm = 0.0
            if FLUTE_PARTS_ORDER[0] in self.data and self.data[FLUTE_PARTS_ORDER[0]]:
                 hj_total = self.data[FLUTE_PARTS_ORDER[0]].get("Total length", 0.0)
                 hj_mortise = self.data[FLUTE_PARTS_ORDER[0]].get("Mortise length", 0.0)
                 current_offset_mm = hj_total - hj_mortise

            # Agujeros de 'left'
            part_idx_offset = 0 # Para numeración de agujeros hole1, hole2...
            left_data = self.data.get(FLUTE_PARTS_ORDER[1], {})
            left_holes_pos = left_data.get("Holes position", [])
            left_holes_diam = left_data.get("Holes diameter", [0]*len(left_holes_pos))
            left_holes_chimney = left_data.get("Holes chimney", [DEFAULT_CHIMNEY_HEIGHT * 1000]*len(left_holes_pos))
            left_holes_diam_out = left_data.get("Holes diameter_out", [d * DEFAULT_HOLE_RADIUS_OUT_FACTOR for d in left_holes_diam])

            for i, hole_pos_mm in enumerate(left_holes_pos):
                side_holes_data.append([
                    f"hole{part_idx_offset + i + 1}",
                    (current_offset_mm + hole_pos_mm) / 1000.0,
                    left_holes_chimney[i] / 1000.0,
                    (left_holes_diam[i] / 2.0) / 1000.0,
                    (left_holes_diam_out[i] / 2.0) / 1000.0
                ])
            part_idx_offset += len(left_holes_pos)

            if FLUTE_PARTS_ORDER[1] in self.data and self.data[FLUTE_PARTS_ORDER[1]]:
                left_total = self.data[FLUTE_PARTS_ORDER[1]].get("Total length", 0.0)
                right_mortise_for_offset = self.data.get(FLUTE_PARTS_ORDER[2], {}).get("Mortise length", 0.0)
                current_offset_mm += left_total - right_mortise_for_offset

            # Agujeros de 'right'
            right_data = self.data.get(FLUTE_PARTS_ORDER[2], {})
            right_holes_pos = right_data.get("Holes position", [])
            right_holes_diam = right_data.get("Holes diameter", [0]*len(right_holes_pos))
            right_holes_chimney = right_data.get("Holes chimney", [DEFAULT_CHIMNEY_HEIGHT * 1000]*len(right_holes_pos))
            right_holes_diam_out = right_data.get("Holes diameter_out", [d * DEFAULT_HOLE_RADIUS_OUT_FACTOR for d in right_holes_diam])
            
            for i, hole_pos_mm in enumerate(right_holes_pos):
                side_holes_data.append([
                    f"hole{part_idx_offset + i + 1}",
                    (current_offset_mm + hole_pos_mm) / 1000.0,
                    right_holes_chimney[i] / 1000.0,
                    (right_holes_diam[i] / 2.0) / 1000.0,
                    (right_holes_diam_out[i] / 2.0) / 1000.0
                ])
            part_idx_offset += len(right_holes_pos)

            if FLUTE_PARTS_ORDER[2] in self.data and self.data[FLUTE_PARTS_ORDER[2]]:
                right_total = self.data[FLUTE_PARTS_ORDER[2]].get("Total length", 0.0)
                foot_mortise_for_offset = self.data.get(FLUTE_PARTS_ORDER[3], {}).get("Mortise length", 0.0)
                current_offset_mm += right_total - foot_mortise_for_offset
            
            # Agujeros de 'foot'
            foot_data = self.data.get(FLUTE_PARTS_ORDER[3], {})
            foot_holes_pos = foot_data.get("Holes position", [])
            if foot_holes_pos: # Asumiendo que si hay 'Holes position', los otros campos también estarán (o tendrán defaults)
                foot_holes_diam = foot_data.get("Holes diameter", [0]*len(foot_holes_pos))
                foot_holes_chimney = foot_data.get("Holes chimney", [DEFAULT_CHIMNEY_HEIGHT * 1000]*len(foot_holes_pos))
                foot_holes_diam_out = foot_data.get("Holes diameter_out", [d * DEFAULT_HOLE_RADIUS_OUT_FACTOR for d in foot_holes_diam])
                for i, hole_pos_mm in enumerate(foot_holes_pos): # Iterar por si hay múltiples agujeros en el pie
                    side_holes_data.append([
                        f"hole{part_idx_offset + i + 1}",
                        (current_offset_mm + hole_pos_mm) / 1000.0,
                        foot_holes_chimney[i] / 1000.0,
                        (foot_holes_diam[i] / 2.0) / 1000.0,
                        (foot_holes_diam_out[i] / 2.0) / 1000.0
                    ])
            
            side_holes_header = [['label', 'position', 'chimney', 'radius', 'radius_out']]
            side_holes_for_openwind = side_holes_header + side_holes_data
            logger.debug(f"Side holes data para openwind: {side_holes_for_openwind}")

            freq_range = np.arange(100, 3000, 2)
            player = Player("FLUTE") # Asegúrate que "FLUTE" es un tipo de Player válido en tu openwind
            player.update_curve("radiation_category", "infinite_flanged")
            player.update_curve("section", np.pi * Rw**2)
            
            if not Path(fing_chart_file).exists():
                logger.error(f"Archivo de digitaciones '{fing_chart_file}' no encontrado para análisis acústico de {self.flute_model}.")
                return

            with Path(fing_chart_file).open("r") as f:
                lines = f.readlines()
            fing_chart_parsed = [line.strip().split() for line in lines if line.strip()] # Ignorar líneas vacías
            
            if not fing_chart_parsed or not fing_chart_parsed[0]:
                logger.error(f"Archivo de digitaciones '{fing_chart_file}' está vacío o malformado para {self.flute_model}.")
                return

            notes_from_chart = fing_chart_parsed[0][1:]
            logger.debug(f"Notas del archivo de digitación: {notes_from_chart}")
            
            for note in notes_from_chart:
                if not note: continue
                logger.debug(f"Calculando impedancia para nota: {note} en {self.flute_model}")
                try:
                    self.acoustic_analysis[note] = ImpedanceComputation(
                        freq_range, geom, side_holes_for_openwind, fing_chart_parsed,
                        player=player,
                        note=note,
                        temperature=temperature,
                        interp=True, # Interpolar para geometrías más suaves
                        source_location="embouchure"
                    )
                    logger.info(f"Análisis acústico completado para nota {note} en {self.flute_model}")
                except Exception as e_imp:
                    logger.error(f"Error en ImpedanceComputation para nota '{note}' en {self.flute_model}: {e_imp}")
                    # Continuar con otras notas si una falla
        
        except Exception as e_main:
            logger.exception(f"Error mayor en compute_acoustic_analysis para {self.flute_model}: {e_main}")

