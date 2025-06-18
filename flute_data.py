import json
import tempfile # Not currently used, but kept for potential future use
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple # Added Union, Optional, Tuple
import numpy as np
from openwind import Player, ImpedanceComputation, InstrumentGeometry # type: ignore

# Assuming notion_utils.py is in the same directory or PYTHONPATH
from notion_utils import get_json_files_from_notion
# Assuming constants.py is in the same directory or PYTHONPATH
from constants import ( # type: ignore
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

class FluteDataInitializationError(ValueError):
    """Custom exception for errors during FluteData initialization, after initial JSON parsing."""
    pass

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
        self.validation_errors: List[Dict[str, Any]] = [] # Cambiado a List[Dict]
        self.validation_warnings: List[Dict[str, Any]] = [] # Cambiado a List[Dict]


        try:
            if isinstance(source, str): # Source is a directory path string
                self.flute_model = Path(source).name if Path(source).is_dir() else source
                if source_name: # If source_name is provided, it overrides the path-derived name
                    self.flute_model = source_name

                if notion_token and database_id: # Assuming source (path) is used as filter name for Notion
                    self._read_json_data_from_notion(notion_token, database_id, source)
                else:
                    self._read_json_data_from_files(source)

                if "Flute Model" not in self.data or not self.data.get("Flute Model"):
                    # If the main data dict doesn't have a model name (e.g., from Notion where it's per-part)
                    # or if individual parts don't set it, use the derived/provided name.
                    # We'll try to get it from headjoint first if available.
                    hj_data = self.data.get(FLUTE_PARTS_ORDER[0], {})
                    model_from_hj = hj_data.get("Flute Model")
                    if model_from_hj:
                        self.flute_model = model_from_hj
                    # Ensure self.data has a top-level "Flute Model" for consistency
                    self.data["Flute Model"] = self.flute_model
                else: # If self.data already has a "Flute Model" (e.g. from a combined JSON)
                    self.flute_model = self.data["Flute Model"]


            elif isinstance(source, dict): # Source is a pre-loaded data dictionary
                self.data = source 
                if source_name:
                    self.flute_model = source_name
                elif "Flute Model" in self.data and self.data["Flute Model"]:
                    self.flute_model = str(self.data["Flute Model"])
                else: # Try to get from headjoint if it's a multi-part dict
                    hj_data_dict = self.data.get(FLUTE_PARTS_ORDER[0], {})
                    model_from_hj_dict = hj_data_dict.get("Flute Model")
                    if model_from_hj_dict:
                        self.flute_model = model_from_hj_dict
                    else:
                        self.flute_model = "InMemoryFlute"
                self.data["Flute Model"] = self.flute_model # Ensure top-level model name
                logger.info(f"FluteData initialized from dictionary. Model: {self.flute_model}")
            else:
                raise ValueError("La fuente (source) para FluteData debe ser una ruta de directorio (str) o un diccionario de datos (dict).")

            try:
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
                self.finger_frequencies = {} 
            except Exception as e_fc:
                logger.error(f"Error al leer o procesar el archivo de digitaciones '{fing_chart_file}' para {self.flute_model}: {e_fc}")
                self.finger_frequencies = {}

            self._validate_loaded_data()

            if self.validation_errors:
                logger.error(f"Errores de validación encontrados para {self.flute_model}: {self.validation_errors}")
                self.combined_measurements = []
                self.acoustic_analysis = {}
            else:
                if self.validation_warnings:
                    logger.warning(f"Advertencias de validación para {self.flute_model}: {self.validation_warnings}")
                self.combined_measurements = self.combine_measurements()
                
                if not self._skip_acoustic_analysis and not self.validation_errors: 
                    try:
                        self.compute_acoustic_analysis(self.fing_chart_file_path, self.temperature)
                    except Exception as e_acoustic:
                        err_msg_acoustic = f"Error al calcular análisis acústico: {e_acoustic}. Verifique la consistencia de la geometría."
                        logger.error(f"Error durante compute_acoustic_analysis para {self.flute_model}: {err_msg_acoustic}", exc_info=True)
                        self.validation_errors.append({'message': err_msg_acoustic})
                        self.acoustic_analysis = {} 
                        raise FluteDataInitializationError(f"Fallo en análisis acústico para {self.flute_model}: {err_msg_acoustic}") from e_acoustic

        except (FileNotFoundError, json.JSONDecodeError) as e_file_json:
            logger.error(f"Error de archivo/JSON durante la inicialización de FluteData para '{self.flute_model}': {e_file_json}")
        except Exception as e_init:
            logger.exception(f"Error al inicializar FluteData para '{self.flute_model}': {e_init}")
            if not self.validation_errors: 
                    self.validation_errors.append({'message': f"Error general al inicializar FluteData: {e_init}"})

    def get_openwind_geometry_inputs(self) -> Tuple[List[List[Union[float, str]]], List[List[Any]], List[List[str]]]:
        if self.validation_errors:
            logger.error(f"No se pueden generar inputs de OpenWind para {self.flute_model} debido a errores de validación previos.")
            return [], [], []

        logger.info(f"Generando inputs de geometría OpenWind para {self.flute_model}...")
        if not self.combined_measurements:
            logger.warning(f"No hay mediciones combinadas para {self.flute_model}, no se pueden generar inputs de geometría.")
            return [], [], []

        logger.debug(f"Creando geometría del bore para {self.flute_model}...")
        bore_segments_m_radius: List[List[Union[float, str]]] = []
        if len(self.combined_measurements) >= 2:
            # Obtener la posición del corcho para normalizar las coordenadas del bore
            # Asumimos que combined_measurements[0]['position'] es la posición absoluta del corcho
            # si el perfil acústico comienza allí.
            # Una forma más robusta es obtenerlo de headjoint_data.
            headjoint_data_for_stopper = self.data.get(FLUTE_PARTS_ORDER[0], {})
            stopper_offset_for_bore_m = headjoint_data_for_stopper.get('_calculated_stopper_absolute_position_mm', 0.0) / 1000.0

            for i in range(len(self.combined_measurements) - 1):
                p1 = self.combined_measurements[i]; p2 = self.combined_measurements[i+1]
                # Normalizar posiciones al corcho
                x_start_m = (p1["position"] / 1000.0) - stopper_offset_for_bore_m
                r_start_m = (p1["diameter"] / 2.0) / 1000.0
                x_end_m = (p2["position"] / 1000.0) - stopper_offset_for_bore_m
                r_end_m = (p2["diameter"] / 2.0) / 1000.0
                # Si el segmento tiene longitud cero o negativa, no lo añadas y registra un error.
                # OpenWind no puede manejar esto.
                if x_end_m <= x_start_m + 1e-7: # Usar una tolerancia pequeña
                    error_detail = (
                        f"Segmento de tubo retrocede o tiene longitud cero/negativa entre puntos combinados:\n"
                        f"  Punto A: AbsPos={p1['position']:.2f}mm (Origen: {p1.get('source_part_name','N/A')}, PosRel={p1.get('source_relative_position','N/A')}mm) -> NormPos={x_start_m:.4f}m\n"
                        f"  Punto B: AbsPos={p2['position']:.2f}mm (Origen: {p2.get('source_part_name','N/A')}, PosRel={p2.get('source_relative_position','N/A')}mm) -> NormPos={x_end_m:.4f}m\n"
                        f"Causa probable: 'Mortise length' incorrectos, mediciones desordenadas o superposición de partes."
                    )
                    # No añadir a self.validation_errors para este caso específico,
                    # ya que compute_acoustic_analysis lo trataría como fatal.
                    # El logger.error es suficiente. OpenWind fallará si la geometría es inutilizable.
                    logger.error(f"Error de geometría para {self.flute_model}: {error_detail}")
                else: # Segmento válido
                    bore_segments_m_radius.append([x_start_m, x_end_m, r_start_m, r_end_m, 'linear'])
                    logger.debug(f"  Bore segment: X=[{x_start_m:.4f}, {x_end_m:.4f}], R=[{r_start_m:.5f}, {r_end_m:.5f}]")
        elif self.combined_measurements: 
            logger.warning(f"Solo una medición combinada para {self.flute_model}. No se puede generar un segmento de taladro OpenWind con un solo punto.")
        else:
            logger.warning(f"No hay mediciones combinadas para {self.flute_model}. No se puede generar la geometría del taladro.")

        if any("Segmento de tubo retrocede" in err.get('message','') or "Superposición de segmentos" in err.get('message','') for err in self.validation_errors):
            logger.error(f"Errores críticos en la geometría del bore para {self.flute_model}. Abortando generación de inputs de OpenWind.")
            return [], [], []

        logger.debug(f"Creando agujeros laterales para {self.flute_model}...")
        side_holes_for_openwind: List[List[Any]] = []
        side_holes_for_openwind.append(['label', 'position', 'chimney', 'radius', 'radius_out']) 

        embouchure_label_const = "embouchure" # Definir la etiqueta constante
        headjoint_data = self.data.get(FLUTE_PARTS_ORDER[0], {})
        emb_hole_positions = headjoint_data.get("Holes position", [])
        emb_hole_diameters = headjoint_data.get("Holes diameter", [])
        emb_hole_chimneys_list = headjoint_data.get("Holes chimney", [])
        emb_hole_diam_out_list = headjoint_data.get("Holes diameter_out", [])

        if emb_hole_positions and emb_hole_diameters:
            emb_pos_mm = emb_hole_positions[0]
            emb_diam_mm = emb_hole_diameters[0]
            emb_chim_mm = emb_hole_chimneys_list[0] if emb_hole_chimneys_list else DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT * 1000
            emb_diam_o_mm = emb_hole_diam_out_list[0] if emb_hole_diam_out_list and emb_hole_diam_out_list[0] > 1e-9 else emb_diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR
            if emb_diam_mm > 1e-9: 
                if emb_chim_mm < 1e-9: emb_chim_mm = DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT * 1000
                if emb_diam_o_mm < 1e-9 or emb_diam_o_mm <= emb_diam_mm: emb_diam_o_mm = emb_diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR
            
            # La posición de la embocadura para OpenWind debe ser relativa al inicio del tubo acústico (corcho)
            stopper_abs_pos_mm = headjoint_data.get('_calculated_stopper_absolute_position_mm', 0.0)
            emb_pos_for_openwind_m = (emb_pos_mm - stopper_abs_pos_mm) / 1000.0

            side_holes_for_openwind.append([
                embouchure_label_const, emb_pos_for_openwind_m, emb_chim_mm / 1000.0,
                (emb_diam_mm / 2.0) / 1000.0, (emb_diam_o_mm / 2.0) / 1000.0
            ])
            logger.debug(f"  Embouchure '{embouchure_label_const}' added: pos_rel_corcho={emb_pos_for_openwind_m:.4f}m")
        else:
            logger.warning(f"  Datos de embocadura insuficientes para {self.flute_model}. '{embouchure_label_const}' no se añadirá geométricamente.")
        
        # Lógica para calcular el inicio físico de cada parte, consistente con combine_measurements
        # current_physical_connection_point_abs_for_holes: Punto donde la parte ANTERIOR termina físicamente y la ACTUAL se une.
        current_physical_connection_point_abs_for_holes = 0.0
        part_physical_starts_map_for_holes: Dict[str, float] = {}

        for idx_part_calc, part_name_calc in enumerate(FLUTE_PARTS_ORDER):
            part_data_calc = self.data.get(part_name_calc, {})
            part_total_length_calc = part_data_calc.get("Total length", 0.0)
            part_mortise_length_calc = part_data_calc.get("Mortise length", 0.0)

            if idx_part_calc == 0: # Headjoint
                part_physical_starts_map_for_holes[part_name_calc] = 0.0
                current_physical_connection_point_abs_for_holes = part_total_length_calc - part_mortise_length_calc
            elif idx_part_calc == 1: # Left
                part_physical_starts_map_for_holes[part_name_calc] = current_physical_connection_point_abs_for_holes
                current_physical_connection_point_abs_for_holes += part_total_length_calc # El final físico de Left es donde se une Right
            else: # Right, Foot
                # El inicio físico de Right/Foot es el punto de conexión anterior MENOS su propio mortise (socket)
                part_physical_starts_map_for_holes[part_name_calc] = current_physical_connection_point_abs_for_holes - part_mortise_length_calc
                current_physical_connection_point_abs_for_holes = part_physical_starts_map_for_holes[part_name_calc] + part_total_length_calc
        
        tone_hole_counter = 0 
        for part_name in FLUTE_PARTS_ORDER:
            if part_name == FLUTE_PARTS_ORDER[0]: continue 
            part_data = self.data.get(part_name, {})
            current_part_physical_start_abs_mm = part_physical_starts_map_for_holes.get(part_name, 0.0)
            stopper_abs_pos_mm = self.data.get(FLUTE_PARTS_ORDER[0], {}).get('_calculated_stopper_absolute_position_mm', 0.0)

            holes_pos_rel_list = part_data.get("Holes position", [])
            holes_diam_rel_list = part_data.get("Holes diameter", [])
            holes_chimney_rel_list = part_data.get("Holes chimney", [])
            holes_diam_out_rel_list = part_data.get("Holes diameter_out", [])
            for i, hole_pos_rel_mm in enumerate(holes_pos_rel_list):
                diam_mm = holes_diam_rel_list[i] if i < len(holes_diam_rel_list) else 7.0 
                chimney_mm = holes_chimney_rel_list[i] if i < len(holes_chimney_rel_list) else DEFAULT_CHIMNEY_HEIGHT * 1000
                diam_out_mm = holes_diam_out_rel_list[i] if i < len(holes_diam_out_rel_list) and holes_diam_out_rel_list[i] > 1e-9 else diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR
                if diam_mm > 1e-9: 
                    if chimney_mm < 1e-9: chimney_mm = DEFAULT_CHIMNEY_HEIGHT * 1000
                    if diam_out_mm < 1e-9 or diam_out_mm <= diam_mm: diam_out_mm = diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR
                tone_hole_counter += 1
                hole_label = f"hole{tone_hole_counter}"
                # Posición absoluta del agujero desde el inicio FÍSICO del headjoint
                hole_abs_physical_pos_mm = current_part_physical_start_abs_mm + hole_pos_rel_mm
                # Posición del agujero para OpenWind (relativa al corcho)
                hole_pos_for_openwind_m = (hole_abs_physical_pos_mm - stopper_abs_pos_mm) / 1000.0

                side_holes_for_openwind.append([
                    hole_label, hole_pos_for_openwind_m, chimney_mm / 1000.0,
                    (diam_mm / 2.0) / 1000.0, (diam_out_mm / 2.0) / 1000.0
                ])
                logger.debug(f"  Tone hole '{hole_label}' added: part='{part_name}', pos_rel_part={hole_pos_rel_mm}mm, pos_abs_phys={hole_abs_physical_pos_mm:.2f}mm, pos_rel_corcho={hole_pos_for_openwind_m:.4f}m")
        logger.debug(f"  Total de agujeros geométricos definidos (incl. embocadura si se añadió): {len(side_holes_for_openwind) -1}")

        logger.debug(f"Creando tabla de digitaciones para {self.flute_model} desde {self.fing_chart_file_path}...")
        fing_chart_parsed: List[List[str]] = []
        try:
            with open(self.fing_chart_file_path, 'r', encoding='utf-8') as f:
                # Leer y filtrar la embocadura si está presente en el archivo
                raw_lines = [line.strip().split() for line in f if line.strip() and not line.startswith('#')]
                if raw_lines:
                    # Mantener el encabezado
                    fing_chart_parsed.append(raw_lines[0])
                    # Añadir filas de datos, excluyendo la embocadura
                    for row in raw_lines[1:]:
                        if row and row[0].lower() != embouchure_label_const:
                            fing_chart_parsed.append(row)
            if not fing_chart_parsed or len(fing_chart_parsed) <=1 : # Si solo queda el encabezado o está vacío
                 logger.warning(f"El archivo de digitaciones {self.fing_chart_file_path} está vacío o solo contiene embocadura. Creando dummy chart.")
                 fing_chart_parsed = [['label', 'D_dummy']] # Crear un chart dummy
        except FileNotFoundError: logger.error(f"Archivo de digitaciones no encontrado: {self.fing_chart_file_path}")
        except Exception as e_chart: logger.error(f"Error al procesar el archivo de digitaciones: {e_chart}")

        if side_holes_for_openwind and len(side_holes_for_openwind) > 1: 
            geom_hole_labels_set = {str(hole_entry[0]) for hole_entry in side_holes_for_openwind[1:]} 
            
            # Asegurar que el chart dummy tenga un encabezado válido si se creó arriba
            if not fing_chart_parsed or not fing_chart_parsed[0] or fing_chart_parsed[0][0].lower() != 'label' or len(fing_chart_parsed[0]) < 2:
                logger.warning(f"Tabla de digitaciones para {self.flute_model} está vacía o no tiene encabezado 'label'. Creando una dummy.")
                fing_chart_parsed = [['label', 'D_dummy']] 
            
            chart_header_row = fing_chart_parsed[0]
            num_notes_in_chart = len(chart_header_row) - 1
            chart_file_labels_set = {str(row[0]) for row in fing_chart_parsed[1:]}

            for geom_label in geom_hole_labels_set:
                if geom_label.lower() == embouchure_label_const: # No añadir la embocadura al chart
                    continue
                if geom_label not in chart_file_labels_set and num_notes_in_chart > 0:
                    logger.info(f"Etiqueta de agujero de tono '{geom_label}' (de geometría) no encontrada en tabla de digitaciones. Añadiendo como 'abierto'.")
                    new_row = [geom_label] + ['o'] * num_notes_in_chart
                    fing_chart_parsed.append(new_row)
        else:
            logger.warning(f"No hay agujeros geométricos (aparte de posible embocadura) definidos en side_holes_for_openwind para {self.flute_model}, o el chart está vacío.")
            if not fing_chart_parsed or len(fing_chart_parsed) <=1: fing_chart_parsed = [['label', 'D_dummy']]

        logger.debug(f"Inputs de geometría para OpenWind ({self.flute_model}): Bore segments: {len(bore_segments_m_radius)}, Holes: {len(side_holes_for_openwind)-1 if side_holes_for_openwind else 0}, Chart rows: {len(fing_chart_parsed)-1 if fing_chart_parsed else 0}")
        logger.debug(f"  Bore (primeros 2): {bore_segments_m_radius[:2]}")
        logger.debug(f"  Holes (primeros 3): {side_holes_for_openwind[:3]}")
        logger.debug(f"  Chart (primeras 3 filas): {fing_chart_parsed[:3]}")
        return bore_segments_m_radius, side_holes_for_openwind, fing_chart_parsed

    def _calculate_part_absolute_start_position_mm(self, part_name: str) -> float:
        """
        Calcula la posición absoluta de inicio FÍSICO de una parte,
        relativa al inicio FÍSICO del headjoint (que es 0).
        Esta función es crucial para calcular las posiciones absolutas de los agujeros de tono.
        """
        if self.validation_errors:
            logger.error(f"Saltando _calculate_part_absolute_start_position_mm para '{part_name}' debido a errores de validación.")
            return 0.0

        if part_name == FLUTE_PARTS_ORDER[0]: # Headjoint
            return 0.0

        current_physical_offset_mm = 0.0
        try:
            part_index = FLUTE_PARTS_ORDER.index(part_name)
        except ValueError:
            logger.error(f"Nombre de parte desconocido '{part_name}' en _calculate_part_absolute_start_position_mm.")
            return 0.0

        # Acumular las longitudes de las partes anteriores según cómo se ensamblan físicamente
        for i in range(part_index):
            prev_part_name = FLUTE_PARTS_ORDER[i]
            prev_part_data = self.data.get(prev_part_name, {})
            prev_total_length = prev_part_data.get("Total length", 0.0)
            
            length_to_add_for_physical_offset = 0.0
            if prev_part_name == FLUTE_PARTS_ORDER[0]: # Headjoint
                # Left se inserta en el socket de Headjoint.
                # El inicio físico de Left es el final del cuerpo de Headjoint.
                prev_mortise_length = prev_part_data.get("Mortise length", 0.0) # Profundidad del socket de HJ
                length_to_add_for_physical_offset = prev_total_length - prev_mortise_length
            elif prev_part_name == FLUTE_PARTS_ORDER[1]: # Left
                # Right se inserta en el tenon de Left.
                # El inicio físico de Right (su socket) es el final del cuerpo de Left.
                length_to_add_for_physical_offset = prev_total_length # Left no tiene socket que reste a su contribución física
            elif prev_part_name == FLUTE_PARTS_ORDER[2]: # Right
                # Foot se inserta en el tenon de Right.
                # El inicio físico de Foot (su socket) es el final del cuerpo de Right.
                length_to_add_for_physical_offset = prev_total_length # Right no tiene socket que reste a su contribución física
            
            current_physical_offset_mm += length_to_add_for_physical_offset
        
        logger.debug(f"Posición de inicio físico absoluta calculada para '{part_name}': {current_physical_offset_mm:.2f} mm")
        return current_physical_offset_mm


    def _read_json_data_from_files(self, base_dir: str) -> None:
        logger.info(f"Leyendo datos JSON desde archivos en: {base_dir} para {self.flute_model}")
        loaded_data_for_parts: Dict[str, Any] = {}
        for part in FLUTE_PARTS_ORDER:
            json_path = Path(base_dir) / f"{part}.json"
            try:
                with json_path.open('r', encoding='utf-8') as file:
                    part_json_content = json.load(file)
                    # Asegurar que el "Flute Model" de la parte coincida con el general si es posible
                    if "Flute Model" not in part_json_content or not part_json_content["Flute Model"]:
                        part_json_content["Flute Model"] = self.flute_model
                    elif part_json_content["Flute Model"] != self.flute_model and part == FLUTE_PARTS_ORDER[0]:
                        # Si el headjoint tiene un nombre de modelo diferente, podría ser el principal
                        logger.warning(f"El 'Flute Model' en {part}.json ('{part_json_content['Flute Model']}') "
                                       f"difiere del modelo general ('{self.flute_model}'). "
                                       f"Usando el del headjoint como principal.")
                        self.flute_model = part_json_content["Flute Model"]
                    loaded_data_for_parts[part] = part_json_content
                logger.debug(f"Cargado {json_path}")
            except FileNotFoundError as e_fnf: 
                err_msg_fnf = f"No se encontró el archivo JSON para la parte '{part}': {json_path}"
                logger.error(err_msg_fnf)
                self.validation_errors.append({'part': part, 'message': err_msg_fnf})
            except json.JSONDecodeError as e: 
                err_msg_json = f"Error al decodificar JSON en '{json_path}': {e.msg} (línea {e.lineno}, col {e.colno})"
                logger.error(err_msg_json)
                self.validation_errors.append({'part': part, 'message': err_msg_json})
            except Exception as e_gen: 
                err_msg_gen = f"Error inesperado cargando '{json_path}': {e_gen}"
                logger.error(err_msg_gen, exc_info=True)
                self.validation_errors.append({'part': part, 'message': err_msg_gen})
        
        self.data.update(loaded_data_for_parts)
        # Asegurar que el flute_model de la instancia sea el correcto después de cargar todas las partes
        if FLUTE_PARTS_ORDER[0] in self.data and "Flute Model" in self.data[FLUTE_PARTS_ORDER[0]]:
            self.flute_model = self.data[FLUTE_PARTS_ORDER[0]]["Flute Model"]
        self.data["Flute Model"] = self.flute_model # Actualizar el top-level


    def _read_json_data_from_notion(self, notion_token: str, database_id: str, flute_name_filter: str) -> None:
        logger.info(f"Leyendo datos JSON desde Notion para la flauta: {flute_name_filter}")
        try:
            retrieved_data_map = get_json_files_from_notion(notion_token, database_id, flute_name_filter)
            missing_parts = [part for part in FLUTE_PARTS_ORDER if part not in retrieved_data_map]
            if missing_parts:
                err_msg = f"No se pudieron recuperar todas las partes desde Notion para '{flute_name_filter}'. Faltan: {', '.join(missing_parts)}"
                logger.error(err_msg)
                raise ValueError(err_msg)
            loaded_data_for_parts: Dict[str, Any] = {}
            # El nombre de la flauta de Notion debería ser el principal
            # Asumimos que el nombre del modelo está en los datos de la headjoint o es el filtro.
            if FLUTE_PARTS_ORDER[0] in retrieved_data_map and "Flute Model" in retrieved_data_map[FLUTE_PARTS_ORDER[0]]:
                self.flute_model = retrieved_data_map[FLUTE_PARTS_ORDER[0]]["Flute Model"]
            else:
                self.flute_model = flute_name_filter # Usar el filtro como nombre de modelo

            for part_name in FLUTE_PARTS_ORDER:
                part_json_content = retrieved_data_map[part_name]
                # Asegurar que cada parte tenga el nombre del modelo
                part_json_content["Flute Model"] = self.flute_model
                loaded_data_for_parts[part_name] = part_json_content
                logger.debug(f"Datos de Notion cargados para la parte: {part_name}")
            
            self.data.update(loaded_data_for_parts)
            self.data["Flute Model"] = self.flute_model # Actualizar el top-level

        except Exception as e:
            logger.exception(f"Error al obtener datos de Notion para '{flute_name_filter}': {e}")
            raise ValueError(f"Error al obtener datos de Notion: {e}")

    def _get_diameter_from_measurements_at_pos(self, measurements_list: List[Dict[str, float]], target_pos_mm: float) -> float:
        if not measurements_list:
            logger.warning(f"No hay mediciones en measurements_list para interpolar diámetro en {self.flute_model} para pos {target_pos_mm}.")
            return 0.0
        try:
            sorted_measurements = sorted(measurements_list, key=lambda m: m.get("position", float('inf')))
            positions = np.array([item.get("position", 0.0) for item in sorted_measurements])
            diameters = np.array([item.get("diameter", 0.0) for item in sorted_measurements])
            if not positions.size or not diameters.size:
                 logger.warning(f"Array de posiciones o diámetros vacío después de procesar measurements_list en {self.flute_model}.")
                 return 0.0
            if target_pos_mm <= positions[0]: return diameters[0]
            if target_pos_mm >= positions[-1]: return diameters[-1]
            return float(np.interp(target_pos_mm, positions, diameters))
        except Exception as e:
            logger.error(f"Error durante la interpolación del diámetro en la lista de mediciones en posición {target_pos_mm:.2f}mm para {self.flute_model}: {e}")
            return 0.0

    def _validate_loaded_data(self):
        self.validation_errors.clear()
        self.validation_warnings.clear()
        if not self.data:
            self.validation_errors.append({'message': "No se cargaron datos para ninguna parte de la flauta."})
            return

        for part_name in FLUTE_PARTS_ORDER:
            part_data = self.data.get(part_name)
            is_headjoint = (part_name == FLUTE_PARTS_ORDER[0])
            if not isinstance(part_data, dict):
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': Los datos faltan o no son un diccionario."})
                continue 
            total_length = part_data.get("Total length")
            if not isinstance(total_length, (int, float)):
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': 'Total length' falta o es inválido ({total_length})."})
                total_length = 0 
            elif total_length <= 0:
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': 'Total length' ({total_length}) debe ser positivo."})
            mortise_length = part_data.get("Mortise length")
            if not isinstance(mortise_length, (int, float)):
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': 'Mortise length' falta o es inválido ({mortise_length})."})
            elif mortise_length < 0:
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': 'Mortise length' ({mortise_length}) no puede ser negativo."})
            elif total_length > 0 and isinstance(mortise_length, (int, float)) and mortise_length > total_length:
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': 'Mortise length' ({mortise_length}) no puede ser mayor que 'Total length' ({total_length})."})
            measurements = part_data.get("measurements")
            if not isinstance(measurements, list):
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': 'measurements' falta o no es una lista."})
            else:
                try:
                    original_positions = [m.get("position") for m in measurements if isinstance(m, dict)]
                    current_measurements = [dict(m) for m in measurements if isinstance(m, dict)] 
                    current_measurements.sort(key=lambda m: m.get("position", float('inf')))
                    if len(original_positions) == len(current_measurements) and \
                       any(orig_pos != sorted_m.get("position") for orig_pos, sorted_m in zip(original_positions, current_measurements) if orig_pos is not None):
                        self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}': 'measurements' no estaban ordenados por posición. Se han ordenado automáticamente."})
                        part_data["measurements"] = current_measurements
                        measurements = current_measurements
                except Exception as e_sort:
                    self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}': No se pudieron ordenar 'measurements': {e_sort}"})
                for i, m_item in enumerate(measurements):
                    if not isinstance(m_item, dict):
                        self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}', medición {i+1}: El item no es un diccionario."})
                        continue
                    m_pos = m_item.get("position")
                    m_diam = m_item.get("diameter")
                    if not isinstance(m_pos, (int, float)):
                        self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}', medición {i+1}: 'position' falta o es inválida ({m_pos})."})
                    elif m_pos < 0:
                        self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}', medición {i+1}: 'position' ({m_pos}) es negativa. Se usará abs({m_pos})."})
                        m_item["position"] = abs(m_pos)
                    elif total_length > 0 and m_pos > total_length + 1e-6:
                        self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}', medición {i+1}: 'position' ({m_pos}) excede 'Total length' ({total_length})."})
                    if not isinstance(m_diam, (int, float)):
                        self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}', medición {i+1}: 'diameter' falta o es inválido ({m_diam})."})
                    elif m_diam <= 0:
                        self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}', medición {i+1}: 'diameter' ({m_diam}) debe ser positivo."})
            holes_pos = part_data.get("Holes position")
            holes_diam = part_data.get("Holes diameter")
            if not isinstance(holes_pos, list) or not isinstance(holes_diam, list):
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': 'Holes position' o 'Holes diameter' falta o no es una lista."})
            elif len(holes_pos) != len(holes_diam):
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': Discrepancia en longitud de 'Holes position' ({len(holes_pos)}) y 'Holes diameter' ({len(holes_diam)})."})
            else:
                num_holes = len(holes_pos)
                for prop_name, default_val_factor, is_emb_specific in [
                    ("Holes chimney", DEFAULT_CHIMNEY_HEIGHT * 1000, True),
                    ("Holes diameter_out", DEFAULT_HOLE_RADIUS_OUT_FACTOR, False)
                ]:
                    prop_list = part_data.get(prop_name)
                    if prop_list is not None:
                        if not isinstance(prop_list, list):
                            self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}': '{prop_name}' no es una lista. Se usarán/generarán defaults."})
                            prop_list = [None] * num_holes 
                        corrected_list = list(prop_list) 
                        while len(corrected_list) < num_holes: corrected_list.append(None) 
                        corrected_list = corrected_list[:num_holes] 
                        for i_h in range(num_holes):
                            if corrected_list[i_h] is None or (isinstance(corrected_list[i_h], (int,float)) and corrected_list[i_h] < 0):
                                self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}', agujero {i_h+1}: '{prop_name}' inválido o ausente. Se generará default."})
                                if prop_name == "Holes chimney":
                                    default_h_val = (DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT if is_emb_specific and part_name == FLUTE_PARTS_ORDER[0] and i_h == 0 else DEFAULT_CHIMNEY_HEIGHT) * 1000
                                    corrected_list[i_h] = default_h_val
                                elif prop_name == "Holes diameter_out" and isinstance(holes_diam[i_h], (int,float)) and holes_diam[i_h] > 0:
                                    corrected_list[i_h] = holes_diam[i_h] * default_val_factor
                                else: 
                                    corrected_list[i_h] = 0.1 
                        part_data[prop_name] = corrected_list
                for i, h_pos_val in enumerate(holes_pos):
                    if not isinstance(h_pos_val, (int, float)) or h_pos_val < 0 or (total_length > 0 and h_pos_val > total_length + 1e-6):
                        self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}', agujero {i+1}: 'position' ({h_pos_val:.2f}) inválida, negativa o excede 'Total length' ({total_length:.2f})."})
                    current_hole_diam_spec = holes_diam[i] if i < len(holes_diam) else None
                    effective_hole_diam = None
                    if isinstance(current_hole_diam_spec, (list, tuple)) and len(current_hole_diam_spec) == 2 and all(isinstance(axis, (int, float)) for axis in current_hole_diam_spec):
                        major_axis, minor_axis = float(current_hole_diam_spec[0]), float(current_hole_diam_spec[1])
                        if major_axis > 0 and minor_axis > 0:
                            ellipse_area = np.pi * (major_axis / 2.0) * (minor_axis / 2.0)
                            effective_hole_diam = 2.0 * np.sqrt(ellipse_area / np.pi)
                            self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}', agujero {i+1}: Diámetro elíptico ({major_axis:.2f}x{minor_axis:.2f}mm) interpretado. Usando diámetro circular equivalente: {effective_hole_diam:.2f}mm."})
                            part_data["Holes diameter"][i] = effective_hole_diam 
                        else:
                            self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}', agujero {i+1}: Ejes de elipse ({major_axis}x{minor_axis}) deben ser positivos."})
                    elif isinstance(current_hole_diam_spec, (int, float)):
                        effective_hole_diam = float(current_hole_diam_spec)
                    if effective_hole_diam is None or effective_hole_diam <= 0:
                        self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}', agujero {i+1}: 'diameter' ({current_hole_diam_spec}) inválido, no numérico, o no positivo después de conversión."})

            if is_headjoint:
                stopper_rel_pos_key = "Stopper Position Relative to Embouchure Center (mm)"
                stopper_rel_pos_val = part_data.get(stopper_rel_pos_key)
                calculated_stopper_abs_pos_mm: Optional[float] = None
                emb_hole_positions_for_stopper = part_data.get("Holes position", [])

                if emb_hole_positions_for_stopper: 
                    emb_center_pos_mm = emb_hole_positions_for_stopper[0] 
                    if stopper_rel_pos_val is not None and isinstance(stopper_rel_pos_val, (int, float)):
                        calculated_stopper_abs_pos_mm = emb_center_pos_mm + float(stopper_rel_pos_val)
                        logger.info(f"{self.flute_model}: Usando '{stopper_rel_pos_key}' ({stopper_rel_pos_val}mm) provisto. Posición absoluta del corcho: {calculated_stopper_abs_pos_mm:.2f}mm.")
                    else:
                        headjoint_physical_measurements = part_data.get("measurements", [])
                        bore_diam_at_emb_center_mm = self._get_diameter_from_measurements_at_pos(headjoint_physical_measurements, emb_center_pos_mm)
                        if bore_diam_at_emb_center_mm > 0:
                            default_stopper_rel_offset = -bore_diam_at_emb_center_mm
                            calculated_stopper_abs_pos_mm = emb_center_pos_mm + default_stopper_rel_offset
                            self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}': '{stopper_rel_pos_key}' no especificado o inválido. Calculando por defecto: {default_stopper_rel_offset:.2f}mm (un diámetro de tubo en embocadura) desde el centro de la embocadura. Posición absoluta del corcho: {calculated_stopper_abs_pos_mm:.2f}mm."})
                        else:
                            self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': No se pudo calcular la posición del corcho por defecto debido a un diámetro de tubo inválido en la embocadura."})
                    if calculated_stopper_abs_pos_mm is not None:
                        if calculated_stopper_abs_pos_mm < 0:
                            self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': Posición absoluta del corcho calculada ({calculated_stopper_abs_pos_mm:.2f}mm) es negativa. Revisar datos."})
                        elif total_length > 0 and calculated_stopper_abs_pos_mm >= total_length:
                             self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': Posición absoluta del corcho calculada ({calculated_stopper_abs_pos_mm:.2f}mm) excede o iguala la longitud total del headjoint ({total_length:.2f}mm)."})
                        part_data['_calculated_stopper_absolute_position_mm'] = calculated_stopper_abs_pos_mm
                elif stopper_rel_pos_val is not None: 
                     self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}': '{stopper_rel_pos_key}' especificado, pero no hay datos de embocadura para calcular su posición absoluta."})

    def combine_measurements(self) -> List[Dict[str, float]]:
        if self.validation_errors:
            logger.error(f"Saltando combine_measurements para {self.flute_model} debido a errores de validación.")
            return []
        logger.info(f"Combinando mediciones para {self.flute_model} según la lógica de ensamblaje acústico.")
        combined_measurements: List[Dict[str, float]] = []
        
        # current_physical_connection_point_abs: Punto donde la parte ANTERIOR termina físicamente y la ACTUAL se une.
        current_physical_connection_point_abs = 0.0

        for i, part_name in enumerate(FLUTE_PARTS_ORDER):
            part_specific_data = self.data.get(part_name, {})
            if not part_specific_data:
                logger.warning(f"No hay datos para la parte '{part_name}' en {self.flute_model}. Saltando.")
                continue

            is_headjoint = (part_name == FLUTE_PARTS_ORDER[0])
            measurements_list = part_specific_data.get("measurements", [])
            # Asegurar que las mediciones de la parte estén ordenadas por posición
            sorted_part_measurements = sorted(
                [m for m in measurements_list if isinstance(m, dict) and "position" in m and "diameter" in m],
                key=lambda m: m["position"]
            )

            part_json_mortise_length = part_specific_data.get("Mortise length", 0.0) # Profundidad del socket de esta parte
            part_total_length = part_specific_data.get("Total length", 0.0)

            # part_acoustic_body_starts_here_abs: Posición absoluta donde comienza el cuerpo acústico de ESTA parte.
            part_acoustic_body_starts_here_abs: float
            # part_physical_start_for_this_part_abs: Posición absoluta donde las mediciones relativas de esta parte (pos:0) se mapean.
            part_physical_start_for_this_part_abs: float

            if is_headjoint:
                part_physical_start_for_this_part_abs = 0.0 # Headjoint siempre empieza en 0
                stopper_abs_pos_mm = part_specific_data.get('_calculated_stopper_absolute_position_mm')
                if stopper_abs_pos_mm is None:
                    logger.critical(f"Posición del corcho no calculada para headjoint de {self.flute_model}. No se puede combinar.")
                    self.validation_errors.append({'part': part_name, 'message': "Posición del corcho no calculada."})
                    return []
                part_acoustic_body_starts_here_abs = stopper_abs_pos_mm
                
                # Añadir el punto del corcho
                diam_at_stopper = self._get_diameter_from_measurements_at_pos(sorted_part_measurements, stopper_abs_pos_mm)
                combined_measurements.append({
                    "position": stopper_abs_pos_mm, "diameter": diam_at_stopper,
                    "source_part_name": part_name, "source_relative_position": stopper_abs_pos_mm
                })
            
            elif part_name == FLUTE_PARTS_ORDER[1]: # Left
                # Left se une donde termina el cuerpo acústico del Headjoint (inicio del socket del HJ)
                part_physical_start_for_this_part_abs = current_physical_connection_point_abs
                part_acoustic_body_starts_here_abs = part_physical_start_for_this_part_abs # Cuerpo acústico de Left comienza en su inicio físico

                # Determinar el diámetro de Left en su inicio acústico (posición relativa 0.0)
                if sorted_part_measurements and abs(sorted_part_measurements[0]['position'] - 0.0) < 1e-6:
                    diam_at_left_start = sorted_part_measurements[0]['diameter']
                else: # Interpolar o tomar el primer diámetro si no hay punto en 0.0
                    diam_at_left_start = self._get_diameter_from_measurements_at_pos(sorted_part_measurements, 0.0)

                new_part_start_point = {
                    "position": part_acoustic_body_starts_here_abs, "diameter": diam_at_left_start,
                    "source_part_name": part_name, "source_relative_position": 0.0
                }
                if combined_measurements and \
                   abs(combined_measurements[-1]['position'] - new_part_start_point['position']) < 1e-6 and \
                   combined_measurements[-1]['source_part_name'] == new_part_start_point['source_part_name']:
                    logger.debug(f"Actualizando punto de inicio existente para {part_name} en {new_part_start_point['position']:.2f}mm con diámetro {diam_at_left_start:.2f}mm.")
                    combined_measurements[-1]['diameter'] = diam_at_left_start
                else:
                    logger.debug(f"Añadiendo punto de inicio para {part_name} en {new_part_start_point['position']:.2f}mm, diámetro {diam_at_left_start:.2f}mm.")
                    combined_measurements.append(new_part_start_point)

            else: # Right, Foot
                part_physical_start_for_this_part_abs = current_physical_connection_point_abs - part_json_mortise_length
                part_acoustic_body_starts_here_abs = part_physical_start_for_this_part_abs + part_json_mortise_length
                
                # Determinar el diámetro de la parte actual (Right/Foot) en el inicio de su cuerpo acústico
                # (que es en su posición relativa part_json_mortise_length)
                diam_at_current_part_acoustic_start = self._get_diameter_from_measurements_at_pos(
                    sorted_part_measurements, part_json_mortise_length # Posición relativa del inicio acústico
                )
                new_part_start_point = {
                    "position": part_acoustic_body_starts_here_abs, # Posición absoluta
                    "diameter": diam_at_current_part_acoustic_start,
                    "source_part_name": part_name, # "right" o "foot"
                    "source_relative_position": part_json_mortise_length # Posición relativa
                }
                if combined_measurements and \
                   abs(combined_measurements[-1]['position'] - new_part_start_point['position']) < 1e-6 and \
                   combined_measurements[-1]['source_part_name'] == new_part_start_point['source_part_name']:
                    logger.debug(f"Actualizando punto de inicio existente para {part_name} en {new_part_start_point['position']:.2f}mm con diámetro {diam_at_current_part_acoustic_start:.2f}mm.")
                    combined_measurements[-1]['diameter'] = diam_at_current_part_acoustic_start
                else:
                    logger.debug(f"Añadiendo punto de inicio para {part_name} en {new_part_start_point['position']:.2f}mm, diámetro {diam_at_current_part_acoustic_start:.2f}mm.")
                    combined_measurements.append(new_part_start_point)


            # Procesar mediciones de la parte actual
            # Solo añadimos mediciones que están DENTRO del cuerpo acústico de esta parte.
            for meas_data in sorted_part_measurements:
                pos_rel_part_mm = meas_data["position"] 
                diam_part_mm = meas_data["diameter"]
                
                # Posición absoluta de la medición actual, relativa al inicio FÍSICO de la parte actual
                pos_abs_flute_mm = part_physical_start_for_this_part_abs + pos_rel_part_mm

                # Filtrar mediciones según el cuerpo acústico de la parte actual
                # El cuerpo acústico de la headjoint va de stopper_abs_pos_mm a (part_total_length - part_json_mortise_length)
                # El cuerpo acústico de 'left' va de part_physical_start_for_this_part_abs a (part_physical_start_for_this_part_abs + part_total_length)
                # El cuerpo acústico de 'right'/'foot' va de (part_physical_start_for_this_part_abs + part_json_mortise_length) a (part_physical_start_for_this_part_abs + part_total_length)
                
                current_part_acoustic_body_physical_start_rel_part: float # Inicio del cuerpo acústico relativo al inicio físico de la parte
                current_part_acoustic_body_physical_end_rel_part: float   # Fin del cuerpo acústico relativo al inicio físico de la parte

                if is_headjoint:
                    current_part_acoustic_body_physical_start_rel_part = stopper_abs_pos_mm
                    current_part_acoustic_body_physical_end_rel_part = part_total_length - part_json_mortise_length
                elif part_name == FLUTE_PARTS_ORDER[1]: # Left
                    current_part_acoustic_body_physical_start_rel_part = 0.0 # Left starts acoustically at its beginning.
                    # Left contributes its full total length to the acoustic profile.
                    current_part_acoustic_body_physical_end_rel_part = part_total_length
                else: # Right, Foot
                    current_part_acoustic_body_physical_start_rel_part = part_json_mortise_length # Cuerpo empieza después del socket
                    current_part_acoustic_body_physical_end_rel_part = part_total_length
                
                # Solo añadir puntos que estén estrictamente después del inicio del cuerpo acústico
                # (o en el inicio si es el primer punto del cuerpo)
                # y antes o en el final del cuerpo acústico.
                if pos_rel_part_mm < current_part_acoustic_body_physical_start_rel_part - 1e-6:
                    continue 
                if pos_rel_part_mm > current_part_acoustic_body_physical_end_rel_part + 1e-6:
                    continue
                
                # Evitar añadir puntos antes del último punto añadido si no es el punto de inicio del cuerpo acústico
                # Esto previene problemas si hay mediciones en la parte que caen dentro del socket de la parte actual
                # pero antes del final del socket.
                if combined_measurements and pos_abs_flute_mm < combined_measurements[-1]["position"] + 1e-6 and \
                   abs(pos_abs_flute_mm - part_acoustic_body_starts_here_abs) > 1e-6 : # No es el punto de inicio exacto del cuerpo acústico
                    continue

                # Manejo de puntos en la misma posición absoluta
                if combined_measurements and abs(combined_measurements[-1]["position"] - pos_abs_flute_mm) < 1e-6:
                    # Si el punto anterior es de la MISMA parte, es un duplicado o un error de datos.
                    # Se podría actualizar el diámetro o simplemente ignorar. Por ahora, se ignora para evitar alterar el perfil.
                    if combined_measurements[-1]["source_part_name"] == part_name:
                        logger.debug(f"Skipping measurement for {part_name} at {pos_abs_flute_mm:.2f}mm as it's at the same position as the previous point from the same part.")
                        continue
                    # Si el punto anterior es de una parte DIFERENTE (es una unión),
                    # se procederá a añadir este nuevo punto. La limpieza final se encargará.
                    # Esto es importante para que plot_combined_flute_data pueda dibujar el escalón y cambiar de color.
                    else: # combined_measurements[-1]["source_part_name"] != part_name
                        logger.debug(f"Join point at {pos_abs_flute_mm:.2f}mm. Previous part: {combined_measurements[-1]['source_part_name']}, Current part: {part_name}. Adding new point.")
                        # No hacer 'continue', permitir que el punto se añada abajo.
                        pass


                combined_measurements.append({
                    "position": pos_abs_flute_mm, "diameter": diam_part_mm,
                    "source_part_name": part_name, "source_relative_position": pos_rel_part_mm
                })

            # --- Añadir el punto final acústico de la parte actual ---
            part_acoustic_end_abs_val: float
            part_acoustic_end_rel_pos_for_diam_lookup: float

            if is_headjoint:
                # El final acústico del Headjoint es donde se une el Left (antes del socket del Headjoint)
                part_acoustic_end_abs_val = (part_physical_start_for_this_part_abs + part_total_length) - part_json_mortise_length
                part_acoustic_end_rel_pos_for_diam_lookup = part_total_length - part_json_mortise_length
            else: # Left, Right, Foot
                # El final acústico es el final físico de la parte
                part_acoustic_end_abs_val = part_physical_start_for_this_part_abs + part_total_length
                part_acoustic_end_rel_pos_for_diam_lookup = part_total_length
            
            diam_at_part_acoustic_end = self._get_diameter_from_measurements_at_pos(
                sorted_part_measurements, 
                part_acoustic_end_rel_pos_for_diam_lookup
            )

            can_add_acoustic_end_point = True
            if combined_measurements:
                last_added_point = combined_measurements[-1]
                # Si el último punto añadido es de la MISMA parte y en la MISMA posición
                if last_added_point.get("source_part_name") == part_name and \
                   abs(last_added_point["position"] - part_acoustic_end_abs_val) < 1e-6:
                    if abs(last_added_point["diameter"] - diam_at_part_acoustic_end) > 1e-3: # Si el diámetro es diferente, actualizar
                        logger.debug(f"Updating diameter at acoustic end of {part_name} (pos {part_acoustic_end_abs_val:.2f}mm) from {last_added_point['diameter']:.2f} to {diam_at_part_acoustic_end:.2f}mm.")
                        last_added_point["diameter"] = diam_at_part_acoustic_end
                    can_add_acoustic_end_point = False
                # Si el punto final acústico haría retroceder el bore y el último punto es de la misma parte
                elif part_acoustic_end_abs_val < last_added_point["position"] + 1e-6 and \
                     last_added_point.get("source_part_name") == part_name:
                    logger.debug(f"Skipping acoustic end for {part_name} at {part_acoustic_end_abs_val:.2f}mm; would recede from {last_added_point['position']:.2f}mm (same part).")
                    can_add_acoustic_end_point = False
            
            if can_add_acoustic_end_point:
                logger.debug(f"Adding acoustic end point for {part_name} at {part_acoustic_end_abs_val:.2f}mm, diameter {diam_at_part_acoustic_end:.2f}mm.")
                combined_measurements.append({
                    "position": part_acoustic_end_abs_val, "diameter": diam_at_part_acoustic_end,
                    "source_part_name": part_name, "source_relative_position": part_acoustic_end_rel_pos_for_diam_lookup
                })

            # Actualizar current_physical_connection_point_abs para la siguiente iteración.
            # Este es el punto donde la SIGUIENTE parte se unirá FÍSICAMENTE.
            if is_headjoint:
                # Headjoint's physical start is 0. Its acoustic end is before its socket.
                # Left se une al inicio del socket del Headjoint.
                current_physical_connection_point_abs = part_physical_start_for_this_part_abs + (part_total_length - part_json_mortise_length)
            elif part_name == FLUTE_PARTS_ORDER[1]: # Left part
                # Right se une al final FÍSICO de Left (que es su longitud total).
                current_physical_connection_point_abs = part_physical_start_for_this_part_abs + part_total_length
            else: # Right, Foot
                # La siguiente parte (o el final de la flauta) se considera después del final FÍSICO de la parte actual.
                # part_physical_start_for_this_part_abs ya tuvo en cuenta el "retroceso" del socket.
                current_physical_connection_point_abs = part_physical_start_for_this_part_abs + part_total_length

        # Limpieza final: ordenar y eliminar duplicados exactos o fusionar puntos muy cercanos
        if combined_measurements:
            combined_measurements.sort(key=lambda m: m["position"])
            unique_combined_measurements: List[Dict[str, float]] = []
            if not combined_measurements: return []

            unique_combined_measurements.append(combined_measurements[0])
            for k in range(1, len(combined_measurements)):
                current_m = combined_measurements[k]
                prev_m = unique_combined_measurements[-1]
                # Solo fusionar si la posición es la misma Y la parte de origen es la misma.
                # Si la posición es la misma pero la parte de origen es diferente (ej. unión de partes),
                # se deben mantener ambos puntos para permitir un escalón y cambio de color correcto.
                if abs(current_m["position"] - prev_m["position"]) < 1e-6 and \
                   current_m.get("source_part_name") == prev_m.get("source_part_name"):
                    # Si los diámetros son diferentes, esto indica un problema de lógica o datos.
                    # Por ahora, se prioriza el último (o se podría promediar, o tomar el de la parte "más interna")
                    if abs(current_m["diameter"] - prev_m["diameter"]) > 1e-3:
                         logger.warning(f"Conflicto de diámetro en posición {current_m['position']:.3f}mm: {prev_m['diameter']:.3f} (de {prev_m['source_part_name']}) vs {current_m['diameter']:.3f} (de {current_m['source_part_name']}). Usando el último.")
                    prev_m["diameter"] = current_m["diameter"] # Actualizar al último diámetro en caso de colisión
                    prev_m["source_part_name"] = current_m["source_part_name"]
                    prev_m["source_relative_position"] = current_m["source_relative_position"]
                else: # Diferente posición O diferente parte de origen (incluso si es la misma posición)
                    unique_combined_measurements.append(current_m)
            combined_measurements = unique_combined_measurements

        logger.debug(f"Mediciones combinadas generadas para {self.flute_model} con {len(combined_measurements)} puntos.")
        if not combined_measurements and any(self.data.values()):
            logger.warning(f"La lista de mediciones combinadas está vacía para {self.flute_model}, aunque hay datos de partes.")
        return combined_measurements

    def _find_holes_outside_bore(self) -> str:
        """
        Identifica los agujeros que están posicionados fuera del tubo principal calculado
        tal como se preparó para OpenWind.
        Devuelve una cadena que detalla los agujeros problemáticos, o una cadena vacía si no se encuentran
        o los datos son insuficientes.
        """
        try:
            # Obtener las entradas de geometría tal como OpenWind las vería
            bore_segments_m_radius, side_holes_for_openwind, _ = self.get_openwind_geometry_inputs()

            if not bore_segments_m_radius:
                return "Datos de segmentos del bore para OpenWind no disponibles."
            if not side_holes_for_openwind or len(side_holes_for_openwind) <= 1: # Encabezado + datos
                return "Datos de agujeros laterales para OpenWind no disponibles."

            min_bore_pos_m = bore_segments_m_radius[0][0]  # Ya relativo al corcho
            max_bore_pos_m = bore_segments_m_radius[-1][1] # Ya relativo al corcho
            
            problematic_holes = []
            tolerance = 1e-6 # Pequeña tolerancia para comparaciones de punto flotante

            for hole_entry in side_holes_for_openwind[1:]: # Saltar encabezado
                hole_label = str(hole_entry[0])
                # La posición en side_holes_for_openwind[1] ya es relativa al corcho
                hole_pos_m_rel_cork = float(hole_entry[1]) 

                if hole_pos_m_rel_cork < min_bore_pos_m - tolerance or \
                   hole_pos_m_rel_cork > max_bore_pos_m + tolerance:
                    problematic_holes.append(
                        f"Agujero '{hole_label}' en {hole_pos_m_rel_cork*1000:.2f}mm (pos. OpenWind, rel. al corcho) "
                        f"está fuera del rango del bore [{min_bore_pos_m*1000:.2f}mm, {max_bore_pos_m*1000:.2f}mm]."
                    )
            
            return "; ".join(problematic_holes) if problematic_holes else "No se identificaron agujeros específicos fuera del bore mediante esta verificación."
        except Exception as e_check:
            logger.error(f"Error interno al verificar agujeros fuera del bore: {e_check}")
            return "No se pudo completar la verificación de agujeros fuera del bore."

    def compute_acoustic_analysis(self, fing_chart_file: str, temperature: float) -> None:
        if self.validation_errors: 
            logger.error(f"Saltando compute_acoustic_analysis para {self.flute_model} debido a errores de validación.")
            return
        logger.debug(f"Calculando análisis acústico para {self.flute_model} a {temperature}°C.")
        self.acoustic_analysis = {} 
        try:
            if not self.combined_measurements:
                logger.warning(f"No hay mediciones combinadas para {self.flute_model}, saltando análisis acústico.")
                return
            bore_segments_m_radius, side_holes_for_openwind, fing_chart_parsed = self.get_openwind_geometry_inputs()
            if self.validation_errors: 
                logger.error(f"Errores de validación de geometría impiden el análisis acústico para {self.flute_model}.")
                raise ValueError(f"Errores de validación de geometría: {self.validation_errors[0]['message']}")

            # Normalizar las posiciones de combined_measurements para que el corcho sea x=0
            headjoint_data_for_stopper_ic = self.data.get(FLUTE_PARTS_ORDER[0], {})
            stopper_offset_for_ic_m = headjoint_data_for_stopper_ic.get('_calculated_stopper_absolute_position_mm', 0.0) / 1000.0
            
            geom_for_ic = [[(m["position"] / 1000.0) - stopper_offset_for_ic_m, m["diameter"] / 2000.0] for m in self.combined_measurements]
            
            Rw = 0.006 
            headjoint_data = self.data.get(FLUTE_PARTS_ORDER[0], {})
            emb_hole_diameters = headjoint_data.get("Holes diameter", [])
            if emb_hole_diameters and emb_hole_diameters[0] > 0: Rw = (emb_hole_diameters[0] / 2.0) / 1000.0
            else: logger.warning(f"No se encontró diámetro de embocadura válido para {self.flute_model}, usando Rw por defecto {Rw}m.")

            freq_range = np.arange(100, 3000, 2.0)
            player = Player("FLUTE")
            player.update_curve("radiation_category", "infinite_flanged")
            player.update_curve("section", np.pi * Rw**2)

            fing_chart_path_obj = Path(fing_chart_file)
            if not fing_chart_path_obj.is_file():
                logger.error(f"Archivo de digitaciones '{fing_chart_file}' no encontrado o inválido para análisis acústico de {self.flute_model}.")
                return
            if not fing_chart_parsed or not fing_chart_parsed[0] or len(fing_chart_parsed[0]) <=1 :
                logger.error(f"Archivo de digitaciones '{fing_chart_file}' está vacío o malformado para {self.flute_model}.")
                return
            notes_from_chart = fing_chart_parsed[0][1:]
            logger.debug(f"Notas del archivo de digitación para {self.flute_model}: {notes_from_chart}")

            for note in notes_from_chart:
                if not note or not note.strip(): continue 
                logger.debug(f"Calculando impedancia para nota: {note} en {self.flute_model}")
                try:
                    self.acoustic_analysis[note] = ImpedanceComputation(
                        freq_range, geom_for_ic, side_holes_for_openwind, fing_chart_parsed,
                        player=player, note=note, temperature=temperature,
                        interp=True, source_location="embouchure"
                    )
                    logger.info(f"Análisis acústico completado para nota {note} en {self.flute_model}")
                except Exception as e_imp:
                    error_msg_detail = str(e_imp)
                    if "One hole is placed outside the main bore" in error_msg_detail:
                        holes_info_str = self._find_holes_outside_bore()
                        if holes_info_str and "No se identificaron" not in holes_info_str and "no disponibles" not in holes_info_str:
                            error_msg_detail += f"\n  Detalle de agujero(s) problemático(s): {holes_info_str}"
                        else:
                            error_msg_detail += f"\n  (Verificación automática de agujeros: {holes_info_str})"
                    
                    logger.error(f"Error en ImpedanceComputation para nota '{note}' en {self.flute_model}. {error_msg_detail}", exc_info=True)
                    raise ValueError(f"Fallo en ImpedanceComputation para nota '{note}': {error_msg_detail}") from e_imp
        except Exception as e_main:
            logger.exception(f"Error mayor en compute_acoustic_analysis para {self.flute_model}: {e_main}")
            self.acoustic_analysis = {} 
            raise ValueError(f"Error en la configuración del análisis acústico: {e_main}") from e_main
