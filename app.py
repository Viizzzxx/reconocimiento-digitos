"""
App web del Proyecto Integrador (Streamlit Community Cloud).
Ejecutar localmente:  streamlit run app.py
La lógica (modelos, preprocesamiento, predicción, gráficas) está en nucleo.py, generado desde el notebook.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Reconocimiento de dígitos", page_icon="🔢", layout="wide")


@st.cache_resource(show_spinner="Cargando los modelos entrenados…")
def cargar_nucleo():
    import nucleo
    return nucleo


@st.cache_resource
def hilo_de_prediccion():
    # Un único hilo permanente para todas las predicciones: las ejecuciones casi simultáneas que dispara el lienzo
    # en tiempo real no compiten entre sí por la CPU.
    return ThreadPoolExecutor(max_workers=1)


N = cargar_nucleo()
HILO = hilo_de_prediccion()
CEROS = np.zeros((28, 28), dtype="float32")


def en_caliente(funcion, *args):
    """Ejecuta funcion(*args) en el hilo de predicción tras una llamada de calentamiento a cada modelo.

    Tras unos segundos sin uso, los hilos internos de TensorFlow se duermen y la primera llamada tarda ~30 veces
    más (TL: ~250 ms frente a ~8 ms). Igual que medir_inferencia_ms en el notebook, el tiempo que se muestra es el
    del modelo ya caliente."""
    def tarea():
        for predecir in N.PREDICTORES.values():
            predecir(CEROS)
        return funcion(*args)
    return HILO.submit(tarea).result()

# Variables de color que el CSS del notebook toma del tema de Gradio
st.markdown(f"""<style>
:root {{ --border-color-primary: #E2E8F0; --block-background-fill: #FFFFFF; }}
{N.CSS}
</style>""", unsafe_allow_html=True)

TEXTO_PRIVACIDAD_GRADIO = ("- El enlace público de Gradio pasa por los servidores de túnel de Gradio. **No subas imágenes con datos "
                           "personales** (firmas, identificaciones, documentos).")
TEXTO_PRIVACIDAD_STREAMLIT = ("- La app está alojada en Streamlit Community Cloud: tus imágenes viajan a ese servidor para "
                              "procesarse. **No subas imágenes con datos personales** (firmas, identificaciones, documentos).")
TEXTO_ETICA = N.TEXTO_ETICA.replace(TEXTO_PRIVACIDAD_GRADIO, TEXTO_PRIVACIDAD_STREAMLIT)


def mostrar_html(html):
    # Streamlit trata como bloque de código las líneas con sangría; se compacta el HTML en una sola línea
    st.markdown("".join(linea.strip() for linea in html.splitlines()), unsafe_allow_html=True)


def lienzo(clave):
    resultado = st_canvas(stroke_width=18, stroke_color="#000000", background_color="#FFFFFF", height=280, width=280,
                          drawing_mode="freedraw", update_streamlit=True, return_image_data=True, key=clave)
    imagen = resultado.image_data if resultado is not None else None
    # Lienzo vacío → todo blanco: no hay nada que analizar
    if imagen is None or imagen[..., :3].min() > 245:
        return None
    return imagen


st.title("🔢 Reconocimiento de dígitos manuscritos")
st.caption("Clásico · CNN · RNN · Transfer Learning · Autoencoder — ciclo completo de un sistema inteligente")

tab_comp, tab_dash, tab_ae, tab_etica, tab_acerca = st.tabs(
    ["✏️ Comparador en vivo", "📊 Dashboard", "🧽 Autoencoder", "⚖️ Responsabilidad e impacto", "ℹ️ Acerca del proyecto"])

# ── Comparador en vivo ─────────────────────────────────────────────────────
with tab_comp:
    col_entrada, col_salida = st.columns([2, 3], gap="large")
    with col_entrada:
        modo = st.radio("Entrada", ["✏️ Dibujar", "📁 Subir imagen"], horizontal=True, label_visibility="collapsed")
        if modo == "✏️ Dibujar":
            st.caption("Dibuja un dígito grande y centrado con el ratón (o el dedo). Usa el ícono de la papelera para borrar.")
            entrada = lienzo("lienzo_comparador")
        else:
            archivo = st.file_uploader("Imagen de un dígito", type=["png", "jpg", "jpeg", "bmp", "webp"])
            entrada = np.asarray(Image.open(archivo).convert("RGB")) if archivo is not None else None
    vista, html, figura = en_caliente(N.comparar, entrada)
    with col_entrada:
        if vista is not None:
            st.image(vista, caption="Lo que reciben los modelos (28×28)", width=196)
    with col_salida:
        mostrar_html(html)
        if figura is not None:
            st.plotly_chart(figura, use_container_width=True)

# ── Dashboard ──────────────────────────────────────────────────────────────
with tab_dash:
    st.markdown("Métricas sobre el **conjunto de prueba** (10 000 imágenes que ningún modelo vio al entrenar).")
    st.dataframe(N.tabla_resumen(), hide_index=True, use_container_width=True)
    st.plotly_chart(N.fig_comparativa(), use_container_width=True)
    clave_por_nombre = {r["nombre"]: k for k, r in N.RESULTADOS.items()}
    nombre = st.selectbox("Modelo a inspeccionar", list(clave_por_nombre), index=1)
    st.plotly_chart(N.fig_curvas(clave_por_nombre[nombre]), use_container_width=True)
    st.plotly_chart(N.fig_matriz(clave_por_nombre[nombre]), use_container_width=True)

# ── Autoencoder ────────────────────────────────────────────────────────────
with tab_ae:
    st.markdown("El autoencoder recibe un trazo **con ruido** y lo **reconstruye**. Compara cómo cambia la predicción "
                "de la CNN antes y después de limpiarlo.\n\n" + N.TEXTO_METRICAS_AE)
    col_entrada, col_salida = st.columns([2, 3], gap="large")
    with col_entrada:
        dibujo_ae = lienzo("lienzo_autoencoder")
        intensidad = st.slider("Intensidad del ruido", 0.0, 0.8, 0.5, 0.05)
        if st.button("Ensuciar y reconstruir mi dibujo", type="primary", use_container_width=True):
            st.session_state["ae"] = en_caliente(N.autoencoder_desde_dibujo, dibujo_ae, intensidad)
        if st.button("Usar un ejemplo de MNIST", use_container_width=True):
            st.session_state["ae"] = en_caliente(N.autoencoder_ejemplo, intensidad)
    with col_salida:
        if "ae" in st.session_state:
            entrada_ae, sucia, recon, tabla = st.session_state["ae"]
            if entrada_ae is not None:
                c1, c2, c3 = st.columns(3)
                c1.image(entrada_ae, caption="Entrada", use_container_width=True)
                c2.image(sucia, caption="Con ruido", use_container_width=True)
                c3.image(recon, caption="Reconstruida", use_container_width=True)
            mostrar_html(tabla)
        st.plotly_chart(N.fig_curvas_autoencoder(), use_container_width=True)

# ── Responsabilidad e impacto / Acerca ─────────────────────────────────────
with tab_etica:
    st.markdown(TEXTO_ETICA)

with tab_acerca:
    st.markdown(N.texto_acerca())
