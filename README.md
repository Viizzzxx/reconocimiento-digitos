# Reconocimiento de dígitos manuscritos · App web (Streamlit)

Comparador en vivo de cuatro modelos entrenados con MNIST: Regresión Logística, CNN, LSTM y Transfer Learning con MobileNetV2. Incluye además un autoencoder convolucional, un dashboard de evaluación y un panel de responsabilidad e impacto.

Los modelos se entrenaron en Google Colab con `proyecto_integrador.ipynb`. Esta app solo los carga, no vuelve a entrenar.

- `app.py`: interfaz de Streamlit.
- `nucleo.py`: lógica del notebook (preprocesamiento, predicción y gráficas).
- `modelos/`: los 5 modelos entrenados. No vienen en el zip de la entrega para que pese poco; están en el repositorio de GitHub y el notebook los vuelve a generar.
- `metricas.json`: métricas del entrenamiento completo.

Para ejecutarla localmente:

```bash
pip install -r requirements.txt
streamlit run app.py
```
