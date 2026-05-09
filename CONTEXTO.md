# Proyecto ML Integral — Platzi

## Objetivo
Construir una plataforma funcional de Machine Learning aplicada a una empresa edtech (Platzi).
El proyecto sirve como aprendizaje práctico de ML de principio a fin.

## Meta final
Una plataforma visual hecha con **Streamlit** que permita usar los modelos de ML de forma interactiva — pensada para ser mostrada como portfolio.

---

## Módulos del proyecto

| # | Módulo | Objetivo | Estado |
|---|--------|----------|--------|
| 1 | Churn Prediction | Detectar estudiantes que van a cancelar | 🔲 Pendiente |
| 2 | Course Recommender | Sugerir cursos personalizados | 🔲 Pendiente |
| 3 | User Segmentation | Clustering de perfiles de estudiantes | 🔲 Pendiente |
| 4 | Conversion Prediction | Leads con alta probabilidad de pagar | 🔲 Pendiente |
| 5 | Engagement Analysis | Patrones de actividad y riesgo de abandono | 🔲 Pendiente |

---

## Stack tecnológico

- **Lenguaje:** Python
- **Datos:** pandas, numpy
- **ML:** scikit-learn
- **Visualización:** matplotlib, seaborn, plotly
- **Plataforma/UI:** Streamlit
- **Entorno:** VS Code + Git Bash + venv

---

## Estructura del proyecto

```
ProyectoMLClientes/
├── CONTEXTO.md          ← este archivo, memoria del proyecto
├── venv/                ← entorno virtual (no se sube a git)
├── data/
│   ├── raw/             ← datos originales sin tocar
│   └── processed/       ← datos limpios listos para ML
├── notebooks/           ← exploración y experimentos
│   ├── 01_churn_eda.ipynb
│   ├── 02_recomendacion_eda.ipynb
│   └── ...
├── src/                 ← código reutilizable
│   ├── churn/
│   ├── recommender/
│   ├── segmentation/
│   ├── conversion/
│   └── engagement/
├── models/              ← modelos entrenados guardados
├── app/                 ← plataforma Streamlit
│   ├── main.py
│   └── pages/
└── reports/             ← gráficas y resultados exportados
```

---

## Pipeline general de ML (se repite en cada módulo)

```
1. Entender el problema   → ¿qué queremos predecir?
2. Explorar los datos     → EDA (Exploratory Data Analysis)
3. Limpiar los datos      → preprocesamiento
4. Entrenar el modelo     → fit()
5. Evaluar el modelo      → métricas (accuracy, precision, recall...)
6. Integrar a Streamlit   → hacer la predicción interactiva
```

---

## Conceptos aprendidos

_(Se irán agregando a medida que avancemos)_

---

## Progreso por sesión

### Sesión 1 — 2026-05-07
- [x] Definición del proyecto: 5 módulos ML para Platzi
- [x] Decisión de usar Streamlit como plataforma visual
- [x] Definición del stack tecnológico
- [x] Creación de este archivo de contexto
- [ ] Configuración del entorno virtual (pendiente — instalar librerías)
- [ ] Crear estructura de carpetas

### Próximos pasos
1. Activar el entorno virtual e instalar librerías
2. Crear la estructura de carpetas del proyecto
3. Arrancar con el módulo 1: Churn Prediction
   - Entender qué datos necesitamos
   - Crear datos sintéticos (simulados) de estudiantes
   - Hacer el primer EDA

---

## Decisiones tomadas

| Decisión | Por qué |
|----------|---------|
| Usar Streamlit para la UI | Solo Python, rápido de construir, visual profesional |
| Empezar por Churn | Es el más impactante y establece el pipeline base |
| Datos sintéticos primero | No tenemos datos reales de Platzi, simulamos datos realistas |
| Avanzar de a poco | El objetivo principal es aprender entendiendo cada paso |
