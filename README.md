# 🛡️ Sistema de Detección de Fraudes - TioRicoBank

Bienvenido al panel de control para la detección de anomalías financieras. Este proyecto es una solución práctica e interactiva diseñada para identificar transacciones fraudulentas en tarjetas de crédito, ayudando a proteger los activos de la entidad y sus clientes.

## 📖 Descripción Práctica

En el entorno bancario, revisar millones de transacciones manualmente es imposible. Esta herramienta utiliza Inteligencia Artificial (IA) para aprender el comportamiento "normal" de las transacciones. Cuando algo se desvía de ese patrón (una anomalía), el sistema lo marca como posible fraude.

La aplicación permite a los analistas y auditores:
1.  **Visualizar el comportamiento** de las transacciones legítimas frente a las fraudulentas.
2.  **Evaluar diferentes modelos** de detección para ver cuál es más efectivo.
3.  **Ajustar la sensibilidad** del sistema: ¿Preferimos detectar todo el fraude aunque revisemos algunas transacciones legítimas, o ser más conservadores?
4.  **Entender la causa**: Ver qué variables específicas (monto, hora, etc.) hicieron saltar la alarma.

## 🚀 Funcionalidades

*   **Dashboard Interactivo**: Interfaz web amigable construida con Streamlit.
*   **Análisis Comparativo**: Compara técnicas estadísticas (PCA) contra Redes Neuronales (Autoencoders).
*   **Simulación de Costos**: Calcula cuánto dinero se pierde o se ahorra ajustando los umbrales de detección (balanceando fraudes no detectados vs. falsas alarmas).
*   **Reportes Automáticos**: Generación de matrices de confusión y métricas de rendimiento (Recall, Precisión).

## ▶️ Cómo Ejecutar

Una vez que tengas todo listo, abre tu terminal en la carpeta del proyecto y escribe:

```bash
streamlit run app.py
```

O ir a [streamlit cloud](https://deteccion-anomalias.streamlit.app/)