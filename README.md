# traverso_analysis
Analisis y comparación de traversos a partir de su geometría en archivos json

Las applicaciones hasta hoy 14 de junio 2025:
gui.py: 
-Carga una o varias flautas y las analiza geométrica y acústicamente
-Muestra la geometría de las 4 partes así como su concatenación.
-Muestra admitancia por nota, así como la envolvente del flujo y la presión
-Muestra la inarmonicidad por nota
-Dos descriptores nuevos:
    -MOC y BI_ESPE


flute_experimenter.py:
-Carga una flauta y muestra un análisis simple: geometría, inarmonicidad, MOC y BI_ESPE
-Permite editarla para crear variaciones a partir de la flauta cargada y despliega el análisis de la flauta modificada.
-Esto es útil para estudiar tendencias o variaciones en la geometría


flute_optimizer_gui.py:
-Carga una flauta y calcula el largo que tendría que tener la embocadura para producir una afinación definida por el diapasón del la (tipicamente 415Hz)
-Muestra los largos de las chimeneas optimizadas además de las admitancias por nota.
-Muestra la geometría de las flautas optimizadas junto con la envolvente de flujo y presión.