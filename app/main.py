import streamlit as st

st.set_page_config(
    page_title='Platzi ML Platform',
    page_icon='🎓',
    layout='wide'
)

st.title('🎓 Platzi — Plataforma de Machine Learning')
st.markdown('---')

st.markdown("""
Esta plataforma centraliza los modelos de Machine Learning aplicados al comportamiento
de estudiantes de Platzi. Cada módulo resuelve un problema de negocio específico.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🔴 Churn Prediction
    Detecta estudiantes en riesgo de abandonar la plataforma antes de que ocurra.
    Permite actuar con campañas de retención dirigidas.
    """)

with col2:
    st.markdown("""
    ### 🟡 Course Recommender
    Sugiere el próximo curso según el historial y perfil de cada estudiante.
    Aumenta el engagement y la completitud.
    """)

with col3:
    st.markdown("""
    ### 🟢 User Segmentation
    Agrupa estudiantes por perfil de comportamiento.
    Permite estrategias diferenciadas por segmento.
    """)

st.markdown('---')
col4, col5 = st.columns(2)

with col4:
    st.markdown("""
    ### 🔵 Conversion Prediction
    Identifica usuarios gratuitos con alta probabilidad de convertirse a plan pago.
    """)

with col5:
    st.markdown("""
    ### 🟣 Engagement Analysis
    Detecta patrones de actividad y alerta caídas de engagement antes de que se conviertan en churn.
    """)

st.markdown('---')
st.caption('Platzi ML Platform · Desarrollado con Python y Streamlit')
