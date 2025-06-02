from pathlib import Path

# --- General Constants ---
FLUTE_PARTS_ORDER = ["headjoint", "left", "right", "foot"]

# --- Plotting Constants ---
# Paleta de colores base (ejemplo de Matplotlib)
BASE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
# Colormap para cuando hay muchas flautas (ej. > 6)
COLORMAP_LARGE_NUMBER_OF_FLUTES = 'tab10' # 'viridis', 'tab20'

LINESTYLES = ['-', '--', '-.', ':']

# --- Default Values & Factors ---
MM_TO_M_FACTOR = 1e-3
M_TO_MM_FACTOR = 1e3

# Default acoustic parameters for holes if not specified elsewhere
DEFAULT_CHIMNEY_HEIGHT = 3e-3 # meters
DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT = 5e-3 # meters
DEFAULT_HOLE_RADIUS_OUT_FACTOR = 1.2 # Factor to estimate outer radius from inner if not given

# --- Path Constants ---
# Es mejor que las rutas que dependen de la ubicación del script se definan en el script que las usa directamente
# o que se pasen como argumentos, pero si son verdaderamente globales:
# SCRIPT_DIR_FLUTE_DATA = Path(__file__).resolve().parent # Asumiendo que constants.py está en el mismo dir que flute_data.py
# DEFAULT_FING_CHART_PATH_ABSOLUTE = SCRIPT_DIR_FLUTE_DATA.parent / "data_json" / "traverso_fingerchart.txt"
# if not DEFAULT_FING_CHART_PATH_ABSOLUTE.exists():
#     DEFAULT_FING_CHART_PATH_ABSOLUTE = Path("data_json") / "traverso_fingerchart.txt"

# Para DEFAULT_FING_CHART_PATH, la lógica en flute_data.py es bastante buena para encontrarlo.
# Podría quedarse ahí o, si se centraliza, necesitaría una forma robusta de definir su base.
# Por ahora, dejaremos que flute_data.py maneje su propia ruta de digitación por defecto.

# DEFAULT_DATA_JSON_DIR para la GUI es mejor definirlo en gui.py usando Path(__file__)