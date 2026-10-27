"""
Lógica del Proyecto Integrador (modelos, preprocesamiento, predicción y gráficas) para la app de Streamlit.
Generado a partir de proyecto_integrador.ipynb: es el mismo código del notebook, pero carga los modelos ya
entrenados en Colab (carpeta modelos/) y las métricas (metricas.json) en lugar de entrenar.
"""
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from PIL import Image
from plotly.subplots import make_subplots
from tensorflow import keras

BASE = Path(__file__).resolve().parent   # funciona sin importar desde qué carpeta se ejecute la app
UMBRAL_CONFIANZA = 50.0
rng = np.random.default_rng(42)

with open(BASE / "metricas.json", encoding="utf-8") as f:
    _metricas = json.load(f)
MODO_RAPIDO = _metricas["modo_rapido"]
RESULTADOS, RESULTADO_AE = _metricas["modelos"], _metricas["autoencoder"]

modelo_clasico = joblib.load(BASE / "modelos" / "clasico_regresion_logistica.joblib")
modelo_cnn = keras.models.load_model(BASE / "modelos" / "cnn.keras")
modelo_rnn = keras.models.load_model(BASE / "modelos" / "rnn_lstm.keras")
modelo_tl = keras.models.load_model(BASE / "modelos" / "transfer_mobilenetv2.keras")
autoencoder = keras.models.load_model(BASE / "modelos" / "autoencoder.keras")

# 200 imágenes del conjunto de prueba de MNIST para el botón "Usar un ejemplo de MNIST"
x_test = np.load(BASE / "ejemplos_mnist.npz")["x"].astype("float32") / 255.0



NOMBRES = {
    "clasico": "Clásico (Regresión Logística)",
    "cnn": "CNN",
    "rnn": "RNN (LSTM)",
    "tl": "Transfer Learning (MobileNetV2)",
}


COLORES = {"clasico": "#6B7280", "cnn": "#2563EB", "rnn": "#F59E0B", "tl": "#10B981"}


def predictor_keras(modelo, forma):
    """Predicción de UNA imagen compilada con tf.function: evita la sobrecarga de ejecutar capa por capa en Python
    (en MobileNetV2 baja de ~100 ms a ~8 ms), así los tiempos comparan modelos y no la forma de llamarlos."""
    compilada = tf.function(lambda x: modelo(x, training=False),
                            input_signature=[tf.TensorSpec((1, *forma), tf.float32)])
    return lambda x: compilada(tf.convert_to_tensor(x, tf.float32)).numpy()


def desplazar(img, dy, dx):
    """Mueve la imagen dy filas y dx columnas, rellenando con ceros."""
    salida = np.zeros_like(img)
    h, w = img.shape
    salida[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        img[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return salida


def preprocesar_entrada(imagen):
    """Convierte un dibujo o una foto al formato MNIST (28×28, [0, 1]). Devuelve None si no hay trazo."""
    arr = np.asarray(imagen)
    arr = arr.astype("float32") / 255.0 if np.issubdtype(arr.dtype, np.integer) else arr.astype("float32")

    # 1) Escala de grises (la transparencia se compone sobre fondo blanco)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            alfa = arr[..., 3:4]
            arr = arr[..., :3] * alfa + (1.0 - alfa)
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype="float32")

    # 2) Fondo claro → invertimos para tener trazo blanco sobre negro
    borde = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if np.median(borde) > 0.5:
        arr, borde = 1.0 - arr, 1.0 - borde

    # 3) Quitamos el fondo (útil en fotos con sombras) y recortamos el dígito
    fondo = np.median(borde)
    arr = np.clip((arr - fondo) / max(arr.max() - fondo, 1e-6), 0, 1)
    arr[arr < 0.2] = 0
    filas, cols = np.nonzero(arr)
    if len(filas) == 0:
        return None
    recorte = arr[filas.min():filas.max() + 1, cols.min():cols.max() + 1]

    # 4) Lado mayor = 20 px, conservando proporción
    h, w = recorte.shape
    escala = 20.0 / max(h, w)
    nh, nw = max(1, round(h * escala)), max(1, round(w * escala))
    recorte = Image.fromarray((recorte * 255).astype("uint8")).resize((nw, nh), Image.LANCZOS)
    recorte = np.asarray(recorte, dtype="float32") / 255.0

    # 5) Lienzo 28×28 centrado por centro de masa
    lienzo = np.zeros((28, 28), dtype="float32")
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = recorte
    total = lienzo.sum()
    cy = (lienzo.sum(axis=1) * np.arange(28)).sum() / total
    cx = (lienzo.sum(axis=0) * np.arange(28)).sum() / total
    lienzo = desplazar(lienzo, int(round(13.5 - cy)), int(round(13.5 - cx)))
    return np.clip(lienzo, 0, 1)


def ensuciar(x, intensidad):
    """Ruido gaussiano + 'sal' (píxeles blancos sueltos), recortado a [0, 1]."""
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    ruido = x + intensidad * tf.random.normal(tf.shape(x))
    sal = tf.cast(tf.random.uniform(tf.shape(x)) < intensidad * 0.08, tf.float32)
    return tf.clip_by_value(tf.maximum(ruido, sal), 0.0, 1.0)


def tabla_resumen():
    return pd.DataFrame([{
        "Modelo": r["nombre"],
        "Exactitud (%)": round(r["exactitud"] * 100, 2),
        "Pérdida (log-loss)": round(r["perdida"], 4),
        "Parámetros": r["parametros"],
        "Entrenamiento (s)": r["tiempo_entrenamiento_s"],
        "Inferencia (ms/img)": r["inferencia_ms"],
    } for r in RESULTADOS.values()])


# Todos los modelos reciben la misma imagen 28×28; solo cambia la forma que espera cada arquitectura
PREDICTORES = {
    "clasico": lambda img: modelo_clasico.predict_proba(img.reshape(1, 784))[0],
    "cnn": lambda img, f=predictor_keras(modelo_cnn, (28, 28, 1)): f(img.reshape(1, 28, 28, 1))[0],
    "rnn": lambda img, f=predictor_keras(modelo_rnn, (28, 28)): f(img.reshape(1, 28, 28))[0],
    "tl": lambda img, f=predictor_keras(modelo_tl, (28, 28, 1)): f(img.reshape(1, 28, 28, 1))[0],
}
for predecir in PREDICTORES.values():       # calentamiento: la primera llamada siempre es más lenta
    predecir(np.zeros((28, 28), dtype="float32"))

AVISO_INICIAL = '<div class="vacio">✏️ Dibuja un dígito del 0 al 9 (o sube una imagen) y los cuatro modelos lo analizarán al instante.</div>'
AVISO_SIN_TRAZO = '<div class="vacio">No se detectó ningún trazo. Dibuja un dígito más grande o con más contraste.</div>'


def extraer_imagen(valor):
    """El lienzo de Gradio entrega {'background', 'layers', 'composite'}; usamos 'composite'."""
    if isinstance(valor, dict):
        valor = valor.get("composite")
    return None if valor is None else np.asarray(valor)


def a_vista(img28, tam=196):
    """Amplía la imagen 28×28 sin suavizar, para que se vean los píxeles que recibe el modelo."""
    return np.asarray(Image.fromarray((np.clip(img28, 0, 1) * 255).astype("uint8")).resize((tam, tam), Image.NEAREST))


def html_resultados(filas):
    votos = Counter(f["pred"] for f in filas)
    digito, n = votos.most_common(1)[0]
    mejor = max(filas, key=lambda f: f["conf"])
    tarjetas = ""
    for f in filas:
        aviso = '<div class="aviso">⚠️ baja confianza</div>' if f["conf"] < UMBRAL_CONFIANZA else ""
        tarjetas += f"""
        <div class="tarjeta{' mejor' if f is mejor else ''}" style="--color:{COLORES[f['clave']]}">
          <div class="modelo">{f['nombre']}</div>
          <div class="digito">{f['pred']}</div>
          <div class="barra"><span style="width:{f['conf']:.0f}%"></span></div>
          <div class="meta"><b>{f['conf']:.1f}%</b> confianza · <b>{f['ms']:.1f} ms</b></div>
          {aviso}
        </div>"""
    return (f'<div class="consenso">Consenso: <b>{digito}</b> <span>({n} de {len(filas)} modelos)</span></div>'
            f'<div class="tarjetas">{tarjetas}</div>')


def figura_probabilidades(filas):
    fig = go.Figure()
    for f in filas:
        fig.add_bar(x=list(range(10)), y=f["proba"] * 100, name=f["nombre"], marker_color=COLORES[f["clave"]],
                    hovertemplate="Dígito %{x}: %{y:.1f}%<extra>" + f["nombre"] + "</extra>")
    fig.update_layout(barmode="group", template="plotly_white", height=380, margin=dict(l=40, r=10, t=40, b=110),
                      title="Probabilidad asignada a cada dígito", yaxis=dict(title="%", range=[0, 100]),
                      xaxis=dict(title="Dígito", dtick=1), legend=dict(orientation="h", y=-0.3))
    return fig


def comparar(valor):
    bruto = extraer_imagen(valor)
    if bruto is None:
        return None, AVISO_INICIAL, None
    img = preprocesar_entrada(bruto)
    if img is None:
        return None, AVISO_SIN_TRAZO, None
    filas = []
    for clave, predecir in PREDICTORES.items():
        t0 = time.perf_counter()
        p = predecir(img)
        ms = (time.perf_counter() - t0) * 1000
        filas.append({"clave": clave, "nombre": NOMBRES[clave], "proba": p,
                      "pred": int(p.argmax()), "conf": float(p.max() * 100), "ms": ms})
    return a_vista(img), html_resultados(filas), figura_probabilidades(filas)


def procesar_autoencoder(img, intensidad):
    sucia = ensuciar(img[None, ..., None], intensidad).numpy()[0, ..., 0] if intensidad > 0 else img
    recon = autoencoder(sucia[None, ..., None], training=False).numpy()[0, ..., 0]
    filas = ""
    for nombre, x in [("Entrada", img), ("Con ruido", sucia), ("Reconstruida", recon)]:
        p = PREDICTORES["cnn"](x)
        filas += f"<tr><td>{nombre}</td><td><b>{p.argmax()}</b></td><td>{p.max():.1%}</td></tr>"
    tabla = f'<table class="tabla-ae"><tr><th>Imagen</th><th>Predicción CNN</th><th>Confianza</th></tr>{filas}</table>'
    return a_vista(img), a_vista(sucia), a_vista(recon), tabla


def autoencoder_desde_dibujo(valor, intensidad):
    bruto = extraer_imagen(valor)
    img = preprocesar_entrada(bruto) if bruto is not None else None
    if img is None:
        return None, None, None, AVISO_SIN_TRAZO
    return procesar_autoencoder(img, intensidad)


def autoencoder_ejemplo(intensidad):
    return procesar_autoencoder(x_test[rng.integers(len(x_test))], intensidad)


def fig_comparativa():
    t = tabla_resumen()
    colores = [COLORES[k] for k in RESULTADOS]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Exactitud en prueba (%)", "Inferencia (ms por imagen)"),
                        shared_yaxes=True, horizontal_spacing=0.04)
    fig.add_bar(y=t["Modelo"], x=t["Exactitud (%)"], orientation="h", marker_color=colores,
                text=t["Exactitud (%)"].map(lambda v: f"{v:.2f}%"), textposition="auto", row=1, col=1)
    fig.add_bar(y=t["Modelo"], x=t["Inferencia (ms/img)"], orientation="h", marker_color=colores,
                text=t["Inferencia (ms/img)"].map(lambda v: f"{v:.2f} ms"), textposition="auto", row=1, col=2)
    fig.update_xaxes(range=[max(0, t["Exactitud (%)"].min() - 5), 100], row=1, col=1)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", height=300, showlegend=False, margin=dict(l=10, r=10, t=60, b=30))
    return fig


def fig_pesos_clasico():
    fig = make_subplots(rows=2, cols=5, subplot_titles=[f"Dígito {d}" for d in range(10)],
                        horizontal_spacing=0.02, vertical_spacing=0.12)
    limite = float(np.abs(modelo_clasico.coef_).max())
    for d in range(10):
        fig.add_heatmap(z=modelo_clasico.coef_[d].reshape(28, 28), colorscale="RdBu", zmin=-limite, zmax=limite,
                        showscale=(d == 0), row=d // 5 + 1, col=d % 5 + 1,
                        hovertemplate="peso %{z:.3f}<extra></extra>")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x")
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=60, b=10),
                      title="El modelo clásico no entrena por épocas: estos son sus pesos por dígito")
    return fig


def fig_curvas(clave):
    r = RESULTADOS[clave]
    if r["historial"] is None:
        return fig_pesos_clasico()
    h = r["historial"]
    epocas = list(range(1, len(h["loss"]) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pérdida", "Exactitud"))
    for col, metrica in [(1, "loss"), (2, "accuracy")]:
        fig.add_scatter(x=epocas, y=h[metrica], name="Entrenamiento", mode="lines+markers",
                        line_color=COLORES[clave], legendgroup="e", showlegend=(col == 1), row=1, col=col)
        fig.add_scatter(x=epocas, y=h["val_" + metrica], name="Validación", mode="lines+markers",
                        line=dict(color=COLORES[clave], dash="dot"), legendgroup="v", showlegend=(col == 1),
                        row=1, col=col)
    if "inicio_fine_tuning" in r:
        fig.add_vline(x=r["inicio_fine_tuning"] + 0.5, line_dash="dash", line_color="gray",
                      annotation_text="inicio fine-tuning", annotation_position="top left", row="all", col="all")
    fig.update_xaxes(title="Época", dtick=1)
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=40, r=10, t=60, b=40),
                      title=f"Curvas de entrenamiento · {r['nombre']}", legend=dict(orientation="h", y=-0.2))
    return fig


def fig_matriz(clave):
    r = RESULTADOS[clave]
    cm = np.array(r["matriz_confusion"])
    fig = go.Figure(go.Heatmap(z=cm, x=list(range(10)), y=list(range(10)), colorscale="Blues", text=cm,
                               texttemplate="%{text}", hovertemplate="Real %{y} → Predicho %{x}: %{z}<extra></extra>"))
    fig.update_layout(template="plotly_white", height=480, margin=dict(l=40, r=10, t=60, b=40),
                      title=f"Matriz de confusión · {r['nombre']} (exactitud {r['exactitud']:.2%})",
                      xaxis=dict(title="Predicho", dtick=1), yaxis=dict(title="Real", dtick=1, autorange="reversed"))
    return fig


def fig_curvas_autoencoder():
    h = RESULTADO_AE["historial"]
    epocas = list(range(1, len(h["loss"]) + 1))
    fig = go.Figure()
    fig.add_scatter(x=epocas, y=h["loss"], name="Entrenamiento", mode="lines+markers")
    fig.add_scatter(x=epocas, y=h["val_loss"], name="Validación", mode="lines+markers", line_dash="dot")
    fig.update_layout(template="plotly_white", height=300, title="Pérdida del autoencoder (entropía cruzada binaria)",
                      xaxis=dict(title="Época", dtick=1), margin=dict(l=40, r=10, t=50, b=40))
    return fig


TEXTO_ETICA = """
## ⚖️ Responsabilidad e impacto social de la IA

### 1. Lo que construimos tiene dos caras
En este proyecto usamos un **autoencoder convolucional** para limpiar trazos. Esa misma familia de modelos está en el origen de los ***deepfakes***: los primeros intercambios de rostro populares (2017) entrenaban dos autoencoders con un **codificador compartido**. Al reconstruir la cara de una persona con el decodificador de otra, se obtenía un video falso muy convincente. Reconstruir un "7" y fabricar un video falso se basan en el mismo principio: aprender una representación comprimida de los datos y **generar** contenido nuevo a partir de ella.

### 2. Riesgos principales
| Riesgo | Ejemplo | Impacto |
|---|---|---|
| **Suplantación de identidad** | Voz clonada en una llamada de "un familiar en apuros" o de "el director" pidiendo una transferencia | Fraude, extorsión, robo de cuentas |
| **Deepfakes íntimos no consentidos** | Montajes sexuales de personas reales. Deeptrace (2019) estimó que eran ~96 % de los deepfakes en internet | Daño psicológico y reputacional, sobre todo a mujeres |
| **Desinformación** | Audios o videos falsos de candidatos, autoridades o periodistas | Manipulación electoral, pánico, pérdida de confianza |
| **"Dividendo del mentiroso"** | Alguien descalifica una evidencia real diciendo "eso es IA" | Impunidad, erosión de la verdad compartida |
| **Falsificación de escritura** | Modelos que reconocen y generan escritura a mano pueden imitar firmas, alterar cifras en documentos o resolver CAPTCHAs | Fraude documental, ataques automatizados |

### 3. Limitaciones y sesgos de *nuestro* sistema
- **Sesgo del dataset:** MNIST se construyó con escritura de empleados de la Oficina del Censo y estudiantes de EE. UU. Estilos comunes en México y otros países (el 7 con barra, el 1 con gancho, el 4 abierto) pueden estar menos representados y fallar más.
- **Laboratorio ≠ mundo real:** ~99 % de exactitud en el conjunto de prueba no garantiza lo mismo con dibujos hechos con el ratón. Por eso preprocesamos cada entrada para que se parezca a MNIST.
- **Confianza no es certeza:** la capa softmax siempre reparte el 100 % entre los 10 dígitos, aunque dibujes una letra o un garabato. La app marca "baja confianza" por debajo del 50 %, pero un garabato puede salir como "8" con 90 %.
- **El autoencoder "inventa":** al reconstruir, rellena lo que *cree* que debería haber según lo que aprendió. Una reconstrucción convincente **no prueba** lo que había en el original. Lo mismo aplica a las fotos o audios "mejorados con IA" presentados como evidencia.

### 4. Privacidad en esta aplicación
- Las imágenes se procesan en memoria y **no se guardan**.
- El enlace público de Gradio pasa por los servidores de túnel de Gradio. **No subas imágenes con datos personales** (firmas, identificaciones, documentos).

### 5. Uso ético: qué podemos hacer
- **Consentimiento:** no usar la voz, el rostro ni la escritura de una persona sin su permiso explícito.
- **Transparencia y procedencia:** etiquetar el contenido generado. El Reglamento de IA de la Unión Europea obliga a identificar los deepfakes, y existen estándares de procedencia (C2PA) y marcas de agua invisibles (p. ej. SynthID).
- **Verificación:** ante un audio o video sospechoso, confirmar por otro canal (devolver la llamada, una palabra clave familiar) antes de actuar.
- **Marco legal en México:** la *Ley Olimpia* reconoce la violencia digital y sanciona difundir contenido íntimo sin consentimiento, y se discuten reformas para incluir explícitamente el contenido creado o alterado con IA.
- **Humano en el ciclo:** sistemas como este deben **asistir**, no decidir solos, en contextos con consecuencias (bancos, exámenes, procesos legales).

### 6. Compromisos del equipo
1. No usar estos modelos para imitar la escritura o firma de otra persona.
2. Reportar métricas reales, incluidas las limitaciones (ver el 📊 Dashboard).
3. Mantener la app con fines exclusivamente educativos.

> **Para reflexionar:** si un autoencoder entrenado con 55 000 dígitos ya puede "rellenar" trazos de forma convincente, ¿qué tan fácil es fabricar una prueba falsa con modelos entrenados con millones de rostros y voces? ¿Quién debería responder por el daño: quien crea la herramienta, quien la usa o la plataforma que difunde el resultado?
"""


def texto_acerca():
    partes = ["## ℹ️ Arquitecturas e hiperparámetros\n",
              "Todas las métricas son sobre el **conjunto de prueba de MNIST** (10 000 imágenes no vistas en el entrenamiento)."
              + (" ⚠️ *Modo rápido activado: métricas solo de prueba.*" if MODO_RAPIDO else "") + "\n"]
    for r in RESULTADOS.values():
        partes.append(f"### {r['nombre']} — {r['exactitud']:.2%}\n")
        partes.append("| Parámetro | Valor |\n|---|---|")
        partes += [f"| {k} | {v} |" for k, v in r["hiperparametros"].items()]
        partes.append(f"| Parámetros del modelo | {r['parametros']:,} |\n")
    ae = RESULTADO_AE
    partes.append("### Autoencoder convolucional\n")
    partes.append("| Parámetro | Valor |\n|---|---|")
    partes.append("| Arquitectura | Conv(32)→MaxPool→Conv(32)→MaxPool → latente 7×7×32 → ConvT(32)×2→Conv(1, sigmoide) |")
    partes.append("| Pérdida / optimizador | Entropía cruzada binaria / Adam (lr = 0.001) |")
    partes.append(f"| Épocas | {ae['epocas']} |")
    partes.append(f"| Parámetros del modelo | {ae['parametros']:,} |")
    return "\n".join(partes)


TEXTO_METRICAS_AE = (
    f"**En el conjunto de prueba con ruido 0.5:** el MSE baja de **{RESULTADO_AE['mse_sucia']:.4f}** (sucia) a "
    f"**{RESULTADO_AE['mse_reconstruida']:.4f}** (reconstruida), y la exactitud de la CNN pasa de "
    f"**{RESULTADO_AE['exactitud_cnn_sucia']:.2%}** a **{RESULTADO_AE['exactitud_cnn_reconstruida']:.2%}** "
    f"(con imágenes limpias: {RESULTADO_AE['exactitud_cnn_limpia']:.2%})."
)


CSS = """
.vacio {padding: 28px; text-align: center; border: 2px dashed var(--border-color-primary); border-radius: 12px; opacity: .8}
.consenso {font-size: 1.15rem; margin-bottom: 12px}
.consenso span {opacity: .7; font-size: .95rem}
.tarjetas {display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px}
.tarjeta {border: 1px solid var(--border-color-primary); border-top: 5px solid var(--color); border-radius: 12px;
          padding: 12px 14px; background: var(--block-background-fill)}
.tarjeta.mejor {box-shadow: 0 0 0 2px var(--color)}
.tarjeta .modelo {font-size: .85rem; opacity: .8; min-height: 2.4em}
.tarjeta .digito {font-size: 3.2rem; font-weight: 800; line-height: 1.1; color: var(--color)}
.tarjeta .barra {height: 8px; background: var(--border-color-primary); border-radius: 4px; overflow: hidden; margin: 6px 0}
.tarjeta .barra span {display: block; height: 100%; background: var(--color)}
.tarjeta .meta {font-size: .85rem; font-variant-numeric: tabular-nums}
.tarjeta .aviso {font-size: .8rem; color: #B45309; margin-top: 4px}
.tabla-ae {width: 100%; border-collapse: collapse}
.tabla-ae th, .tabla-ae td {padding: 6px 10px; border-bottom: 1px solid var(--border-color-primary); text-align: left}
"""


AYUDA_PINCEL = "Activa el **pincel** 🖌️ en la barra izquierda del lienzo y dibuja un dígito grande y centrado."
