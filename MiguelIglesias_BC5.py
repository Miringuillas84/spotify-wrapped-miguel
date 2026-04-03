# ============================================================
# CABECERA
# ============================================================
# Alumno: Miguel Iglesias
# URL Streamlit Cloud: https://spotify-wrapped-miguel.streamlit.app/
# URL GitHub: https://github.com/Miringuillas84/spotify-wrapped-miguel

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """ 

Eres un analista de datos experto en Spotify. Tu objetivo es generar código Python para visualizar el historial de escucha del usuario contenido en un DataFrame llamado 'df'.

DATOS DISPONIBLES:
El DataFrame 'df' tiene las siguientes columnas clave creadas para ti:
- 'ts': Timestamp de finalización.
- 'master_metadata_track_name': Nombre de la canción.
- 'master_metadata_album_artist_name': Artista.
- 'ms_played': Milisegundos escuchados.
- 'minutos_reproducidos', 'horas_reproducidas': Tiempo en minutos/horas.
- 'fecha', 'mes', 'mes_nombre', 'dia_semana', 'hora': Componentes temporales.
- 'es_fin_de_semana': Booleano (True para Sáb/Dom).
- 'estacion': 'Verano' (Jun-Ago), 'Invierno' (Dic-Feb), 'Otras'.
- 'skipped': True si se saltó la canción.
- 'shuffle': True si se usó modo aleatorio.
- 'platform': Dispositivo usado.

REGLAS DE SALIDA:
Debes responder ÚNICAMENTE con un objeto JSON válido que tenga esta estructura exacta:
{{
    "tipo": "grafico",
    "codigo": "Código Python usando plotly.express (almacena el resultado en la variable 'fig'). No incluyas st.write ni st.plotly_chart.",
    "interpretacion": "Breve resumen en lenguaje natural del hallazgo."
}}

Si la pregunta está fuera de contexto, usa:
{{
    "tipo": "error",
    "codigo": "",
    "interpretacion": "Lo siento, solo puedo responder preguntas relacionadas con tu historial de Spotify."
}}

DIRECTRICES CRÍTICAS:
- Para rankings (Top 10), agrupa por artista o canción y suma 'minutos_reproducidos' o cuenta registros.
- Para evolución temporal, usa 'mes' o 'fecha'.
- Para patrones de uso, usa 'hora' o 'dia_semana'.
- Para comparaciones estacionales, filtra por la columna 'estacion'.
- Los gráficos deben ser claros, con títulos y etiquetas de ejes en español.
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------
# >>> TU TRABAJO ESTÁ AQUÍ <<<

    # 1. Convertir timestamps
    df['ts'] = pd.to_datetime(df['ts'])
    
    # 2. Crear columnas derivadas para facilitar el análisis
    df['fecha'] = df['ts'].dt.date
    df['mes'] = df['ts'].dt.strftime('%Y-%m')
    df['mes_nombre'] = df['ts'].dt.month_name()
    df['dia_semana'] = df['ts'].dt.day_name()
    df['hora'] = df['ts'].dt.hour
    df['es_fin_de_semana'] = df['ts'].dt.dayofweek.isin([5, 6])
    
    # 3. Métricas de tiempo
    df['minutos_reproducidos'] = df['ms_played'] / 60000
    df['horas_reproducidas'] = df['minutos_reproducidos'] / 60
    
    # 4. Clasificación de estaciones (Pregunta E)
    def get_season(month):
        if month in [12, 1, 2]: return 'Invierno'
        if month in [6, 7, 8]: return 'Verano'
        return 'Otras'
    df['estacion'] = df['ts'].dt.month.apply(get_season)
    
    # 5. Limpieza de nulos en metadata (evitar errores en agrupaciones)
    df['master_metadata_track_name'] = df['master_metadata_track_name'].fillna('Desconocido')
    df['master_metadata_album_artist_name'] = df['master_metadata_album_artist_name'].fillna('Desconocido')

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error(f"Error detectado: {e}")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
# 
# Mi app usa una arquitectura donde el LLM actúa como programador, no como analista.
# El LLM recibe un "mapa" (el esquema de columnas y tipos) pero nunca ve los datos reales.
# Devuelve un objeto JSON con una cadena de texto (código Python) que se ejecuta en el 
# servidor local mediante exec(). No enviamos los datos al LLM por tres razones: 
# privacidad del usuario, ahorro masivo de tokens (coste) y evitar que el modelo 
# "alucine" con los números en lugar de calcularlos con precisión matemática.
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
# Proporciono al LLM el nombre exacto de las columnas pre-procesadas (como 'estacion' o 
# 'minutos_reproducidos') para asegurar que el código generado sea compatible. 
# Gracias a la instrucción sobre 'estacion', el modelo puede resolver: "Compara mi 
# verano vs invierno" filtrando correctamente meses específicos. Sin esta instrucción, 
# el modelo fallaría al intentar adivinar qué meses corresponden a cada estación en 
# el hemisferio norte, produciendo un error de código o un gráfico vacío.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
# El flujo es: 1. El usuario introduce una pregunta. 2. Streamlit envía la pregunta 
# junto con el System Prompt a GPT-4.1-mini. 3. El LLM genera una respuesta en JSON. 
# 4. La app extrae el string de la clave "codigo" y lo ejecuta con exec() sobre el 
# DataFrame 'df' cargado en memoria. 5. El código ejecutado crea un objeto 'fig' 
# de Plotly. 6. Streamlit captura ese objeto y lo renderiza junto con la interpretación 
# textual del JSON, completando la interfaz de usuario. 