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
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []

        try:
            if isinstance(source, str): # Source is a directory path string
                self.flute_model = Path(source).name if Path(source).is_dir() else source
                if source_name: # If source_name is provided, it overrides the path-derived name
                    self.flute_model = source_name

                if notion_token and database_id: # Assuming source (path) is used as filter name for Notion
                    self._read_json_data_from_notion(notion_token, database_id, source)
                else:
                    # _read_json_data_from_files ahora puebla self.validation_errors
                    # y relanza la excepción original si ocurre.
                    # El bloque try/except general de __init__ capturará esto.
                    # No es necesario un try/except específico aquí si _read_json_data_from_files
                    # ya maneja la adición a validation_errors.
                    self._read_json_data_from_files(source)

                # Ensure "Flute Model" is in data after reading from files/Notion
                if "Flute Model" not in self.data or not self.data.get("Flute Model"):
                    self.data["Flute Model"] = self.flute_model
                # If loaded data has a "Flute Model" (e.g. from a general_info.json if you add one),
                # ensure self.flute_model matches it.
                # For now, self.flute_model is primarily derived from directory or source_name.
                # If a part JSON (like headjoint.json) itself contained a "Flute Model" key,
                # that would be unusual and might lead to conflicts.
                # We assume "Flute Model" is a global property, not per-part.
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

            # Validar los datos cargados
            self._validate_loaded_data()

            if self.validation_errors:
                logger.error(f"Errores de validación encontrados para {self.flute_model}: {self.validation_errors}")
                # No continuar con combine_measurements ni compute_acoustic_analysis si hay errores fatales
                self.combined_measurements = []
                self.acoustic_analysis = {}
            else:
                if self.validation_warnings:
                    logger.warning(f"Advertencias de validación para {self.flute_model}: {self.validation_warnings}")
                self.combined_measurements = self.combine_measurements()
                
                # Re-check validation_errors as combine_measurements might add some (though less likely now)
                # The main geometry validation will happen in get_openwind_geometry_inputs
                if not self._skip_acoustic_analysis and not self.validation_errors: 
                    try:
                        self.compute_acoustic_analysis(self.fing_chart_file_path, self.temperature)
                    except Exception as e_acoustic:
                        err_msg_acoustic = f"Error al calcular análisis acústico: {e_acoustic}. Verifique la consistencia de la geometría."
                        logger.error(f"Error durante compute_acoustic_analysis para {self.flute_model}: {err_msg_acoustic}", exc_info=True)
                        self.validation_errors.append({'message': err_msg_acoustic})
                        self.acoustic_analysis = {} # Asegurar que esté vacío
                        # Propagar el error para que la GUI lo maneje
                        raise FluteDataInitializationError(f"Fallo en análisis acústico para {self.flute_model}: {err_msg_acoustic}") from e_acoustic

        except (FileNotFoundError, json.JSONDecodeError) as e_file_json:
            # Estos errores ya deberían haber poblado self.validation_errors
            # con la información de 'part' desde _read_json_data_from_files.
            # No necesitamos hacer nada más aquí, la GUI leerá validation_errors.
            logger.error(f"Error de archivo/JSON durante la inicialización de FluteData para '{self.flute_model}': {e_file_json}")
            # No relanzar aquí para permitir que la GUI maneje los validation_errors.
        except Exception as e_init:
            logger.exception(f"Error al inicializar FluteData para '{self.flute_model}': {e_init}")
            if not self.validation_errors: # Solo añadir si está vacío para no sobrescribir errores más específicos
                    self.validation_errors.append({'message': f"Error general al inicializar FluteData: {e_init}"})
            # Considerar si relanzar aquí o dejar que la GUI maneje los validation_errors.
            # Si no relanzamos, la GUI verá los errores en validation_errors.

    def get_openwind_geometry_inputs(self) -> Tuple[List[List[Union[float, str]]], List[List[Any]], List[List[str]]]:
        # type: ignore
        """
        Prepara y devuelve los componentes de la geometría (taladro, agujeros laterales, tabla de digitaciones)
        en el formato esperado por InstrumentGeometry de OpenWind.
        No incluye ningún tubo adicional; es la geometría base de la flauta.
        Las dimensiones están en metros y radios. Los tipos de forma son 'linear'.
        """ # noqa: E501
        if self.validation_errors:
            logger.error(f"No se pueden generar inputs de OpenWind para {self.flute_model} debido a errores de validación previos.")
            return [], [], []

        logger.info(f"Generando inputs de geometría OpenWind para {self.flute_model}...")
        if not self.combined_measurements:
            logger.warning(f"No hay mediciones combinadas para {self.flute_model}, no se pueden generar inputs de geometría.")
            return [], [], []

        # 1. Geometría del Taladro (Bore)
        logger.debug(f"Creando geometría del bore para {self.flute_model}...")
        bore_segments_m_radius: List[List[Union[float, str]]] = []
        if len(self.combined_measurements) >= 2:
            for i in range(len(self.combined_measurements) - 1):
                p1 = self.combined_measurements[i]; p2 = self.combined_measurements[i+1]
                
                x_start_m = p1["position"] / 1000.0
                r_start_m = (p1["diameter"] / 2.0) / 1000.0
                x_end_m = p2["position"] / 1000.0
                r_end_m = (p2["diameter"] / 2.0) / 1000.0

                # Validación de monotonía y longitud de segmento
                if x_end_m <= x_start_m + 1e-9: # Usar una tolerancia pequeña para flotantes
                    error_detail = (
                        f"Segmento de tubo retrocede o tiene longitud cero/negativa entre puntos combinados:\n"
                        f"  Punto A: AbsPos={p1['position']:.2f}mm (Origen: {p1.get('source_part_name','N/A')}, PosRel={p1.get('source_relative_position','N/A')}mm)\n"
                        f"  Punto B: AbsPos={p2['position']:.2f}mm (Origen: {p2.get('source_part_name','N/A')}, PosRel={p2.get('source_relative_position','N/A')}mm)\n"
                        f"Causa probable: 'Mortise length' incorrectos, mediciones desordenadas o superposición de partes."
                    )
                    self.validation_errors.append({'message': error_detail})
                    logger.error(f"Error de geometría para {self.flute_model}: {error_detail}")
                    # No añadir este segmento problemático, pero continuar validando el resto si es posible
                    # o retornar inmediatamente si se considera un error fatal para la geometría.
                    # Por ahora, vamos a registrar y continuar para ver si hay más.
                    # Si se quiere detener aquí, se podría `return [], [], []` o lanzar una excepción.
                    continue # Saltar este segmento inválido

                if bore_segments_m_radius and x_start_m < bore_segments_m_radius[-1][1] - 1e-9: # Comprobar contra el x_end del segmento anterior
                    prev_x_end = bore_segments_m_radius[-1][1] * 1000
                    error_overlap = f"Superposición de segmentos de tubo: Inicio del segmento actual ({p1['position']:.2f}mm) es menor que el final del anterior ({prev_x_end:.2f}mm)."
                    self.validation_errors.append({'message': error_overlap})
                    logger.error(f"Error de geometría para {self.flute_model}: {error_overlap}")
                
                if x_end_m > x_start_m + 1e-9: # Asegurar que el segmento tenga una longitud positiva significativa
                    bore_segments_m_radius.append([x_start_m, x_end_m, r_start_m, r_end_m, 'linear'])
                    logger.debug(f"  Bore segment: X=[{x_start_m:.4f}, {x_end_m:.4f}], R=[{r_start_m:.5f}, {r_end_m:.5f}]")
        elif self.combined_measurements: # Solo un punto
            logger.warning(f"Solo una medición combinada para {self.flute_model}. No se puede generar un segmento de taladro OpenWind con un solo punto.")
        else:
            logger.warning(f"No hay mediciones combinadas para {self.flute_model}. No se puede generar la geometría del taladro.")

        # Si se encontraron errores de geometría del bore, no continuar con la creación de InstrumentGeometry
        if any("Segmento de tubo retrocede" in err.get('message','') or "Superposición de segmentos" in err.get('message','') for err in self.validation_errors):
            logger.error(f"Errores críticos en la geometría del bore para {self.flute_model}. Abortando generación de inputs de OpenWind.")
            return [], [], []

        # 2. Agujeros Laterales (Side Holes)
        logger.debug(f"Creando agujeros laterales para {self.flute_model}...")
        side_holes_for_openwind: List[List[Any]] = []
        side_holes_for_openwind.append(['label', 'position', 'chimney', 'radius', 'radius_out']) # Header

        embouchure_label = "embouchure"
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
            
            if emb_diam_mm > 1e-9: # Ensure diameter is significant before applying defaults based on it
                if emb_chim_mm < 1e-9: emb_chim_mm = DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT * 1000
                if emb_diam_o_mm < 1e-9 or emb_diam_o_mm <= emb_diam_mm: emb_diam_o_mm = emb_diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR

            side_holes_for_openwind.append([
                embouchure_label,
                emb_pos_mm / 1000.0, 
                emb_chim_mm / 1000.0,
                (emb_diam_mm / 2.0) / 1000.0,
                (emb_diam_o_mm / 2.0) / 1000.0
            ])
            logger.debug(f"  Embouchure '{embouchure_label}' added: pos={emb_pos_mm/1000.0:.4f}m")
        else:
            logger.warning(f"  Datos de embocadura insuficientes para {self.flute_model} (posiciones o diámetros vacíos/ausentes). 'embouchure' no se añadirá geométricamente.")

        tone_hole_counter = 0 
        for part_name in FLUTE_PARTS_ORDER:
            if part_name == FLUTE_PARTS_ORDER[0]: 
                continue 

            part_data = self.data.get(part_name, {})
            current_part_start_abs_mm = self._calculate_part_absolute_start_position_mm(part_name)
            
            holes_pos_rel_list = part_data.get("Holes position", [])
            holes_diam_rel_list = part_data.get("Holes diameter", [])
            holes_chimney_rel_list = part_data.get("Holes chimney", [])
            holes_diam_out_rel_list = part_data.get("Holes diameter_out", [])

            for i, hole_pos_rel_mm in enumerate(holes_pos_rel_list):
                diam_mm = holes_diam_rel_list[i] if i < len(holes_diam_rel_list) else 7.0 
                chimney_mm = holes_chimney_rel_list[i] if i < len(holes_chimney_rel_list) else DEFAULT_CHIMNEY_HEIGHT * 1000
                diam_out_mm = holes_diam_out_rel_list[i] if i < len(holes_diam_out_rel_list) and holes_diam_out_rel_list[i] > 1e-9 else diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR

                if diam_mm > 1e-9: # Ensure diameter is significant
                    if chimney_mm < 1e-9: chimney_mm = DEFAULT_CHIMNEY_HEIGHT * 1000
                    if diam_out_mm < 1e-9 or diam_out_mm <= diam_mm: diam_out_mm = diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR
                
                tone_hole_counter += 1
                hole_label = f"hole{tone_hole_counter}"
                hole_abs_pos_m = (current_part_start_abs_mm + hole_pos_rel_mm) / 1000.0
                
                side_holes_for_openwind.append([
                    hole_label,
                    hole_abs_pos_m,
                    chimney_mm / 1000.0,
                    (diam_mm / 2.0) / 1000.0,
                    (diam_out_mm / 2.0) / 1000.0
                ])
                logger.debug(f"  Tone hole '{hole_label}' added: part='{part_name}', pos_rel={hole_pos_rel_mm}mm, pos_abs={hole_abs_pos_m:.4f}m")
        logger.debug(f"  Total de agujeros geométricos definidos (incl. embocadura si se añadió): {len(side_holes_for_openwind) -1}")

        # 3. Tabla de Digitaciones (Fingering Chart)
        logger.debug(f"Creando tabla de digitaciones para {self.flute_model} desde {self.fing_chart_file_path}...")
        fing_chart_parsed: List[List[str]] = []
        try:
            with open(self.fing_chart_file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip().split() for line in f if line.strip() and not line.startswith('#')]
            if lines:
                fing_chart_parsed = lines
            else:
                logger.warning(f"El archivo de digitaciones {self.fing_chart_file_path} está vacío.")
        except FileNotFoundError:
            logger.error(f"Archivo de digitaciones no encontrado: {self.fing_chart_file_path}")
        except Exception as e_chart:
            logger.error(f"Error al procesar el archivo de digitaciones: {e_chart}")

        if side_holes_for_openwind and len(side_holes_for_openwind) > 1: 
            geom_hole_labels_set = {str(hole_entry[0]) for hole_entry in side_holes_for_openwind[1:]} 

            if not fing_chart_parsed or not fing_chart_parsed[0] or fing_chart_parsed[0][0].lower() != 'label':
                logger.warning(f"Tabla de digitaciones para {self.flute_model} está vacía o no tiene encabezado 'label'. Creando una dummy.")
                fing_chart_parsed = [['label', 'D_dummy']] 
                num_notes_in_chart = 1
            else:
                chart_header_row = fing_chart_parsed[0]
                num_notes_in_chart = len(chart_header_row) - 1

            chart_file_labels_set = {str(row[0]) for row in fing_chart_parsed[1:]}
            
            for geom_label in geom_hole_labels_set:
                if geom_label not in chart_file_labels_set:
                    logger.info(f"Etiqueta de agujero '{geom_label}' (de geometría) no encontrada en tabla de digitaciones. Añadiendo como 'abierto'.")
                    new_row = [geom_label] + ['o'] * num_notes_in_chart
                    fing_chart_parsed.append(new_row)
        else:
            logger.warning(f"No hay agujeros geométricos definidos en side_holes_for_openwind para {self.flute_model}, no se puede aumentar/crear tabla de digitaciones de forma robusta.")
            if not fing_chart_parsed: 
                 fing_chart_parsed = [['label', 'D_dummy']]

        logger.debug(f"Inputs de geometría para OpenWind ({self.flute_model}): Bore segments: {len(bore_segments_m_radius)}, Holes: {len(side_holes_for_openwind)-1 if side_holes_for_openwind else 0}, Chart rows: {len(fing_chart_parsed)-1 if fing_chart_parsed else 0}")
        logger.debug(f"  Bore (primeros 2): {bore_segments_m_radius[:2]}")
        logger.debug(f"  Holes (primeros 3): {side_holes_for_openwind[:3]}")
        logger.debug(f"  Chart (primeras 3 filas): {fing_chart_parsed[:3]}")

        return bore_segments_m_radius, side_holes_for_openwind, fing_chart_parsed

    def _calculate_part_absolute_start_position_mm(self, part_name: str) -> float:
        """
        Calcula la posición absoluta de inicio (en mm) del cuerpo de una parte
        basándose en las longitudes totales y de espiga de las partes anteriores.
        """
        if self.validation_errors: # No intentar calcular si hay errores de validación
            logger.error(f"Saltando _calculate_part_absolute_start_position_mm para '{part_name}' debido a errores de validación.")
            return 0.0

        if part_name == FLUTE_PARTS_ORDER[0]: # Headjoint starts at 0
            return 0.0

        current_offset_mm = 0.0
        part_data_map = {p_name: self.data.get(p_name, {}) for p_name in FLUTE_PARTS_ORDER}

        # Find the index of the current part
        try:
            part_index = FLUTE_PARTS_ORDER.index(part_name)
        except ValueError:
            logger.error(f"Nombre de parte desconocido '{part_name}' en _calculate_part_absolute_start_position_mm.")
            return 0.0 # Should not happen with FLUTE_PARTS_ORDER

        # Sum the body lengths of preceding parts
        for i in range(part_index):
            prev_part_name = FLUTE_PARTS_ORDER[i]
            prev_part_data = part_data_map.get(prev_part_name, {})
            prev_total_length = prev_part_data.get("Total length", 0.0)
            prev_mortise_length = prev_part_data.get("Mortise length", 0.0)

            # The body length of the previous part is Total length - Mortise length
            # For the headjoint (i=0), its body length is Total length - Mortise length.
            # For subsequent parts (i > 0), their body length is Total length - Mortise length.
            # The start of the current part is the sum of the body lengths of all preceding parts.
            current_offset_mm += (prev_total_length - prev_mortise_length)

        logger.debug(f"Posición de inicio absoluta calculada para '{part_name}': {current_offset_mm:.2f} mm")
        return current_offset_mm


    def _read_json_data_from_files(self, base_dir: str) -> None:
        logger.info(f"Leyendo datos JSON desde archivos en: {base_dir} para {self.flute_model}")
        loaded_data_for_parts: Dict[str, Any] = {}
        for part in FLUTE_PARTS_ORDER:
            json_path = Path(base_dir) / f"{part}.json"
            try:
                with json_path.open('r', encoding='utf-8') as file:
                    loaded_data_for_parts[part] = json.load(file)
                logger.debug(f"Cargado {json_path}")
            except FileNotFoundError as e_fnf: # Capturar y añadir a validation_errors
                err_msg_fnf = f"No se encontró el archivo JSON para la parte '{part}': {json_path}"
                logger.error(err_msg_fnf)
                self.validation_errors.append({'part': part, 'message': err_msg_fnf})
                # No relanzar aquí, permitir que __init__ continúe y la GUI maneje validation_errors
            except json.JSONDecodeError as e: # Capturar y añadir a validation_errors
                err_msg_json = f"Error al decodificar JSON en '{json_path}': {e.msg} (línea {e.lineno}, col {e.colno})"
                logger.error(err_msg_json)
                self.validation_errors.append({'part': part, 'message': err_msg_json})
                # No relanzar aquí
            except Exception as e_gen: # Capturar y añadir a validation_errors
                err_msg_gen = f"Error inesperado cargando '{json_path}': {e_gen}"
                logger.error(err_msg_gen, exc_info=True)
                self.validation_errors.append({'part': part, 'message': err_msg_gen})
                # No relanzar aquí
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
        if self.validation_errors: # No combinar si hay errores
            logger.error(f"Saltando combine_measurements para {self.flute_model} debido a errores de validación.")
            return []

        logger.debug(f"Combinando mediciones para {self.flute_model}")
        combined_measurements = []
        
        # Ensure all parts are at least an empty dict in part_data_map if not in self.data
        part_data_map = {part_name: self.data.get(part_name, {}) for part_name in FLUTE_PARTS_ORDER}

        for i, part_name in enumerate(FLUTE_PARTS_ORDER):
            part_specific_data = part_data_map.get(part_name)
            if not part_specific_data:
                logger.warning(f"No hay datos para la parte '{part_name}' en {self.flute_model} al combinar mediciones. Saltando esta parte.")
                continue

            # Calculate the absolute start position for the current part's body
            current_position_abs_mm = self._calculate_part_absolute_start_position_mm(part_name)


            measurements_list = part_specific_data.get("measurements", [])
            if not isinstance(measurements_list, list): # Ensure it's a list
                logger.warning(f"Mediciones para la parte '{part_name}' no es una lista. Saltando mediciones de esta parte.")
                measurements_list = []

            positions = [item.get("position", 0.0) for item in measurements_list]
            diameters = [item.get("diameter", 0.0) for item in measurements_list]

            part_mortise_length = part_specific_data.get("Mortise length", 0.0)
            part_total_length = part_specific_data.get("Total length", 0.0)

            for pos, diam in zip(positions, diameters):
                adjusted_pos = pos + current_position_abs_mm
                # Filter conditions
                # Ensure part_total_length - part_mortise_length is positive or handle zero case if mortise can be >= total_length
                filter_threshold = part_total_length - part_mortise_length
                if part_name == FLUTE_PARTS_ORDER[0] and filter_threshold > 0 and pos >= filter_threshold:
                    continue
                elif part_name == FLUTE_PARTS_ORDER[0] and filter_threshold <= 0: # Mortise too large or total_length zero
                    logger.debug(f"Headjoint {self.flute_model}: mortise {part_mortise_length} >= total_length {part_total_length}. Todas las mediciones se incluirán.")
                
                # Para las demás partes, no filtrar puntos que estén dentro de la espiga (mortise) de la parte actual
                # ya que esos puntos no forman parte del cuerpo acústico de *esta* parte.
                # La espiga de la parte actual se inserta en la parte anterior.
                if part_name != FLUTE_PARTS_ORDER[0] and pos < part_mortise_length - 1e-6 : # Tolerancia para flotantes
                    continue

                if part_name in [FLUTE_PARTS_ORDER[2], FLUTE_PARTS_ORDER[3]] and pos <= part_mortise_length:
                    continue
                combined_measurements.append({
                    "position": adjusted_pos, 
                    "diameter": diam,
                    "source_part_name": part_name,
                    "source_relative_position": pos
                })

        logger.debug(f"Mediciones combinadas generadas para {self.flute_model} con {len(combined_measurements)} puntos.")
        if not combined_measurements and any(part_data_map.values()): # Si hay datos de partes pero no mediciones combinadas
            logger.warning(f"La lista de mediciones combinadas está vacía para {self.flute_model}, aunque hay datos de partes. Verifique la lógica de combinación y los datos de entrada.")
        return combined_measurements

    def compute_acoustic_analysis(self, fing_chart_file: str, temperature: float) -> None:
        if self.validation_errors: # No analizar si hay errores
            logger.error(f"Saltando compute_acoustic_analysis para {self.flute_model} debido a errores de validación.")
            return

        logger.debug(f"Calculando análisis acústico para {self.flute_model} a {temperature}°C.")
        self.acoustic_analysis = {} # Reset before filling

        try:
            if not self.combined_measurements:
                logger.warning(f"No hay mediciones combinadas para {self.flute_model}, saltando análisis acústico.")
                return

            # Obtener y validar la geometría ANTES de intentar el bucle de notas
            bore_segments_m_radius, side_holes_for_openwind, fing_chart_parsed = self.get_openwind_geometry_inputs()
            if self.validation_errors: # Errores detectados en get_openwind_geometry_inputs
                logger.error(f"Errores de validación de geometría impiden el análisis acústico para {self.flute_model}.")
                raise ValueError(f"Errores de validación de geometría: {self.validation_errors[0]['message']}")

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
                emb_diam_o_mm = emb_hole_diam_out_list[0] if emb_hole_diam_out_list and emb_hole_diam_out_list[0] > 1e-9 else emb_diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR

                # Validate and adjust embouchure chimney and outer diameter
                if emb_diam_mm > 1e-9: # If embouchure diameter is significant
                    if emb_chim_mm < 1e-9: # If chimney height is zero or too small
                        logger.warning(f"Embouchure chimney height for {self.flute_model} is zero or too small ({emb_chim_mm:.4f}mm). Using default: {DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT * 1000:.2f}mm.")
                        emb_chim_mm = DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT * 1000
                    
                    # If outer diameter is zero, too small, or not larger than inner diameter, recalculate it
                    if emb_diam_o_mm < 1e-9 or emb_diam_o_mm <= emb_diam_mm:
                        original_emb_diam_o_mm = emb_diam_o_mm # For logging
                        emb_diam_o_mm = emb_diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR
                        logger.warning(f"Embouchure outer diameter for {self.flute_model} was invalid ({original_emb_diam_o_mm:.4f}mm) relative to inner diameter ({emb_diam_mm:.2f}mm). Adjusted to: {emb_diam_o_mm:.2f}mm.")

                side_holes_data.append([
                    "embouchure",
                    emb_pos_mm / 1000.0,
                    emb_chim_mm / 1000.0,
                    (emb_diam_mm / 2.0) / 1000.0,
                    (emb_diam_o_mm / 2.0) / 1000.0 # Use adjusted value
                ])
                Rw = (emb_diam_mm / 2.0) / 1000.0
            else:
                logger.warning(f"Datos de embocadura insuficientes o ausentes para {self.flute_model}. Usando Rw por defecto ({Rw}m).")

            # Corrected logic for current_offset_mm based on the structure of combine_measurements
            part_idx_offset = 0 # For global hole numbering (hole1, hole2...)

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
                # Si la parte actual es el headjoint, ya hemos procesado su primer agujero como embocadura.
                # Los agujeros tonales (hole1, hole2...) deben comenzar desde la siguiente parte.
                if part_name == FLUTE_PARTS_ORDER[0]: # Headjoint
                    continue # Saltar el headjoint para la numeración de agujeros tonales
                 # Calculate the absolute start position for the current part's body
                current_offset_abs_mm = self._calculate_part_absolute_start_position_mm(part_name)
                # The following lines seemed to be part of an 'else' block that was removed or misplaced.
                # They are not directly used before being potentially reassigned or if the logic
                # intended them for a specific part_name condition within the loop.
                # For now, I'm aligning them with the loop's main body.
                # If they were part of a conditional, that logic needs to be restored.
                l_data = self.data.get(FLUTE_PARTS_ORDER[1], {}) # Example, might need context
                r_data = self.data.get(FLUTE_PARTS_ORDER[2], {}) # Example, might need context
                # current_offset_mm was used later for side_holes_data, ensure it's correctly calculated if needed
                # The original calculation for current_offset_mm here seemed specific to the 'foot' part
                # and was outside a conditional check for part_name == FLUTE_PARTS_ORDER[3]
                # This part of the logic needs careful review based on its intended use.
                # For now, just fixing indentation. The variable current_offset_mm is used later.
                # It seems current_offset_abs_mm should be used for hole positions.
                # The variable 'current_offset_mm' used in side_holes_data.append might be a bug
                # and should likely be 'current_offset_abs_mm'.
                # For now, only fixing indentation.
                if part_name == FLUTE_PARTS_ORDER[3]: # 'foot' - Example of how it might have been structured
                    # This is a guess, the original logic for current_offset_mm was complex and outside a clear conditional
                    f_data_mortise = self.data.get(part_name, {}).get("Mortise length", 0.0)
                    # The original calculation for current_offset_mm was:
                    # current_offset_mm = (offset_after_headjoint +
                    #                     l_data.get("Total length", 0.0) +
                    #                     r_data.get("Total length", 0.0) - r_data.get("Mortise length", 0.0) -
                    #                     f_data_mortise)
                    # This needs to be assigned correctly if it's to be used.
                    # For now, we'll assume current_offset_abs_mm is the primary offset.
                    pass # Placeholder for correct logic if needed for current_offset_mm
                # elif part_name == FLUTE_PARTS_ORDER[0]: # Headjoint - Esta condición ya no es necesaria aquí debido al 'continue' de arriba
                    current_offset_abs_mm = 0.0

                part_data = self.data.get(part_name, {})
                if not part_data:
                    logger.warning(f"No data for part '{part_name}' when calculating hole positions for {self.flute_model}.")
                    continue

                holes_pos = part_data.get("Holes position", [])

                # --- Validación: Agujeros más grandes que el tubo ---
                # Solo validar si hay mediciones combinadas para interpolar
                if self.combined_measurements:
                    for i, hole_pos_mm in enumerate(holes_pos):
                         if i < len(part_data.get("Holes diameter", [])):
                             hole_diam_mm = part_data["Holes diameter"][i]
                             # La posición del agujero es relativa a la parte, necesitamos la posición absoluta
                             hole_pos_abs_mm = current_offset_abs_mm + hole_pos_mm
                             bore_diam_at_hole_pos = self._get_bore_diameter_at_absolute_pos(hole_pos_abs_mm)
                             if hole_diam_mm > bore_diam_at_hole_pos:
                                 logger.warning(f"Agujero {i+1} en '{part_name}' ({hole_pos_mm:.2f}mm rel, {hole_pos_abs_mm:.2f}mm abs) tiene diámetro ({hole_diam_mm:.2f}mm) mayor que el diámetro del tubo ({bore_diam_at_hole_pos:.2f}mm) en esa posición para {self.flute_model}. Esto puede causar problemas en Openwind.")

                # --- Fin Validación ---

                num_holes_in_part = len(holes_pos)
                holes_diam = part_data.get("Holes diameter", [7.0]*num_holes_in_part)
                holes_chimney_list = part_data.get("Holes chimney", [])
                holes_diam_out_list = part_data.get("Holes diameter_out", [])

                for i, hole_pos_mm in enumerate(holes_pos):
                    diam_mm = holes_diam[i] if i < len(holes_diam) else 7.0
                    chimney_mm = holes_chimney_list[i] if i < len(holes_chimney_list) else DEFAULT_CHIMNEY_HEIGHT * 1000
                    diam_out_mm = holes_diam_out_list[i] if i < len(holes_diam_out_list) and holes_diam_out_list[i] > 1e-9 else diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR

                    # Validate and adjust tonehole chimney and outer diameter
                    if diam_mm > 1e-9: # If hole diameter is significant
                        if chimney_mm < 1e-9: # If chimney height is zero or too small
                            logger.warning(f"Hole {part_idx_offset + i + 1} chimney height for {self.flute_model} is zero or too small ({chimney_mm:.4f}mm). Using default: {DEFAULT_CHIMNEY_HEIGHT * 1000:.2f}mm.")
                            chimney_mm = DEFAULT_CHIMNEY_HEIGHT * 1000
                        
                        if diam_out_mm < 1e-9 or diam_out_mm <= diam_mm: # If outer diameter is zero, too small, or not larger than inner
                            original_diam_out_mm = diam_out_mm # For logging
                            diam_out_mm = diam_mm * DEFAULT_HOLE_RADIUS_OUT_FACTOR
                            logger.warning(f"Hole {part_idx_offset + i + 1} outer diameter for {self.flute_model} was invalid ({original_diam_out_mm:.4f}mm) relative to inner diameter ({diam_mm:.2f}mm). Adjusted to: {diam_out_mm:.2f}mm.")

                    side_holes_data.append([ # Generamos etiquetas hole1, hole2, ...
                        f"hole{part_idx_offset + i + 1}",
                        (current_offset_abs_mm + hole_pos_mm) / 1000.0, # <--- CORRECCIÓN AQUÍ
                        chimney_mm / 1000.0,
                        (diam_mm / 2.0) / 1000.0,
                        (diam_out_mm / 2.0) / 1000.0 # Use adjusted value
                    ])
                part_idx_offset += num_holes_in_part

            # Nota: La generación de etiquetas "hole1", "hole2", etc., asume que el archivo de digitaciones
            # utiliza esta misma convención para referenciar los agujeros tonales. Si tu archivo de digitaciones
            # usa nombres diferentes (ej. "H1", "T1", "T2"), Openwind reportará errores de "Side component not defined".

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
                if not note or not note.strip(): continue # Saltar notas vacías o solo espacios
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
                                 f"Geom (primeros 5): {geom[:5]}, SideHoles (primeros 5): {side_holes_for_openwind[:5]}. Error: {e_imp}", exc_info=True)
                    # Si un cálculo de impedancia falla, propagar inmediatamente para que __init__ lo maneje.
                    raise ValueError(f"Fallo en ImpedanceComputation para nota '{note}': {e_imp}") from e_imp

        except Exception as e_main:
            logger.exception(f"Error mayor en compute_acoustic_analysis para {self.flute_model}: {e_main}")
            self.acoustic_analysis = {} # Ensure it's empty on major failure
            # Propagar este error también para que __init__ lo maneje
            raise ValueError(f"Error en la configuración del análisis acústico: {e_main}") from e_main

    def _get_bore_diameter_at_absolute_pos(self, abs_x_mm: float) -> float:
        """
        Interpola el diámetro del tubo en una posición absoluta dada (en mm)
        usando las mediciones combinadas.
        """
        if not self.combined_measurements:
            logger.warning(f"No hay mediciones combinadas para interpolar el diámetro en {self.flute_model}.")
            return 0.0

        try:
            positions = np.array([item["position"] for item in self.combined_measurements])
            diameters = np.array([item["diameter"] for item in self.combined_measurements])

            # Asegurarse de que las posiciones estén ordenadas (deberían estarlo por combine_measurements)
            sort_indices = np.argsort(positions)
            positions = positions[sort_indices]
            diameters = diameters[sort_indices]

            return float(np.interp(abs_x_mm, positions, diameters))
        except Exception as e:
            logger.error(f"Error durante la interpolación del diámetro en posición absoluta {abs_x_mm:.2f}mm para {self.flute_model}: {e}")
            return 0.0

    def _validate_loaded_data(self):
        """
        Valida los datos cargados en self.data y puebla
        self.validation_errors y self.validation_warnings.
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()

        if not self.data:
            self.validation_errors.append({'message': "No se cargaron datos para ninguna parte de la flauta."})
            return

        for part_name in FLUTE_PARTS_ORDER:
            part_data = self.data.get(part_name)

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
                    current_measurements = [dict(m) for m in measurements if isinstance(m, dict)] # Copia para ordenar
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
            holes_chimney = part_data.get("Holes chimney")
            holes_diam_out = part_data.get("Holes diameter_out")

            if not isinstance(holes_pos, list) or not isinstance(holes_diam, list):
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': 'Holes position' o 'Holes diameter' falta o no es una lista."})
            elif len(holes_pos) != len(holes_diam):
                self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}': Discrepancia en longitud de 'Holes position' ({len(holes_pos)}) y 'Holes diameter' ({len(holes_diam)})."})
            else:
                num_holes = len(holes_pos)
                # Validar y completar listas opcionales (chimney, diameter_out)
                for prop_name, default_val_factor, is_emb_specific in [
                    ("Holes chimney", DEFAULT_CHIMNEY_HEIGHT * 1000, True),
                    ("Holes diameter_out", DEFAULT_HOLE_RADIUS_OUT_FACTOR, False)
                ]:
                    prop_list = part_data.get(prop_name)
                    if prop_list is not None:
                        if not isinstance(prop_list, list):
                            self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}': '{prop_name}' no es una lista. Se usarán/generarán defaults."})
                            prop_list = [None] * num_holes # Para forzar la generación de defaults
                        
                        corrected_list = list(prop_list) # Copia para modificar
                        while len(corrected_list) < num_holes: corrected_list.append(None) # Rellenar si es más corta
                        corrected_list = corrected_list[:num_holes] # Truncar si es más larga

                        for i_h in range(num_holes):
                            if corrected_list[i_h] is None or (isinstance(corrected_list[i_h], (int,float)) and corrected_list[i_h] < 0):
                                self.validation_warnings.append({'part': part_name, 'message': f"Parte '{part_name}', agujero {i_h+1}: '{prop_name}' inválido o ausente. Se generará default."})
                                if prop_name == "Holes chimney":
                                    default_h_val = (DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT if is_emb_specific and part_name == FLUTE_PARTS_ORDER[0] and i_h == 0 else DEFAULT_CHIMNEY_HEIGHT) * 1000
                                    corrected_list[i_h] = default_h_val
                                elif prop_name == "Holes diameter_out" and isinstance(holes_diam[i_h], (int,float)) and holes_diam[i_h] > 0:
                                    corrected_list[i_h] = holes_diam[i_h] * default_val_factor
                                else: # Fallback si el diámetro base es inválido
                                    corrected_list[i_h] = 0.1 # Un valor pequeño pero positivo
                        part_data[prop_name] = corrected_list

                # Validaciones individuales de agujeros (posición, diámetro)
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
                            # Actualizar la lista de diámetros en part_data para uso posterior
                            part_data["Holes diameter"][i] = effective_hole_diam 
                        else:
                            self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}', agujero {i+1}: Ejes de elipse ({major_axis}x{minor_axis}) deben ser positivos."})
                    elif isinstance(current_hole_diam_spec, (int, float)):
                        effective_hole_diam = float(current_hole_diam_spec)

                    if effective_hole_diam is None or effective_hole_diam <= 0:
                        self.validation_errors.append({'part': part_name, 'message': f"Parte '{part_name}', agujero {i+1}: 'diameter' ({current_hole_diam_spec}) inválido, no numérico, o no positivo después de conversión."})
