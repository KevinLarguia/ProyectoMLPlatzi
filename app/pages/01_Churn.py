import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title='Churn Prediction · Platzi',
    page_icon='🔴',
    layout='wide'
)

# ── CARGAR MODELO ─────────────────────────────────────────────────────────────

RUTA_MODELO = os.path.join(os.path.dirname(__file__), '../../models/churn_model.pkl')
RUTA_DATOS  = os.path.join(os.path.dirname(__file__), '../../data/raw/platzi_churn.csv')

@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)

@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)

if not os.path.exists(RUTA_MODELO):
    st.error('El modelo no está entrenado. Ejecutá primero el notebook 02_churn_modelo.ipynb')
    st.stop()

modelo = cargar_modelo()

FEATURES_NUMERICAS = [
    'edad', 'meses_suscrito', 'cursos_inscritos', 'cursos_completados',
    'tasa_completado', 'certificados_obtenidos', 'rutas_inscritas',
    'rutas_completadas', 'examenes_ruta_aprobados', 'dias_desde_ultimo_login',
    'dias_desde_ultima_clase', 'sesiones_ultimo_mes', 'horas_vistas_total',
    'horas_vistas_ultimo_mes', 'promedio_clases_semana', 'racha_dias_maxima',
    'racha_dias_actual', 'liveclasses_asistidas', 'preguntas_en_foro',
    'respuestas_en_foro', 'upvotes_recibidos', 'tutoriales_publicados',
    'platzi_rank', 'tickets_soporte', 'dias_resolucion_promedio', 'nps_score',
    'intentos_pago_fallidos', 'variacion_sesiones_vs_mes_anterior',
    'experiencia_tech_anios', 'dias_en_gratuito_antes_pago',
]
FEATURES_CATEGORICAS = [
    'plan', 'pais', 'genero', 'escuela_principal', 'dispositivo_principal',
    'tipo_pago', 'objetivo_estudiante', 'nivel_educativo', 'nivel_rewards',
]

# ── HEADER ────────────────────────────────────────────────────────────────────

st.title('🔴 Churn Prediction')
st.markdown('Predicción de abandono sobre la base completa de estudiantes de Platzi.')
st.markdown('---')

# ── CARGA DE DATOS ────────────────────────────────────────────────────────────

st.sidebar.header('Fuente de datos')
fuente = st.sidebar.radio(
    'Seleccioná los datos a analizar:',
    ['Usar base de datos actual', 'Subir CSV propio']
)

if fuente == 'Subir CSV propio':
    archivo = st.sidebar.file_uploader('Subí un CSV con el mismo formato', type='csv')
    if archivo is None:
        st.info('Subí un archivo CSV para continuar.')
        st.stop()
    df_raw = pd.read_csv(archivo)
else:
    df_raw = cargar_datos()

# ── PREDICCIÓN SOBRE TODOS LOS ESTUDIANTES ────────────────────────────────────

X = df_raw[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
df_raw['prob_churn']   = modelo.predict_proba(X)[:, 1]
df_raw['pred_churn']   = modelo.predict(X)

def nivel_riesgo(p):
    if p >= 0.65:
        return 'Alto'
    elif p >= 0.35:
        return 'Medio'
    else:
        return 'Bajo'

df_raw['riesgo'] = df_raw['prob_churn'].apply(nivel_riesgo)

# ── FILTROS ───────────────────────────────────────────────────────────────────

st.sidebar.markdown('---')
st.sidebar.header('Filtros')

riesgo_sel = st.sidebar.multiselect(
    'Nivel de riesgo',
    ['Alto', 'Medio', 'Bajo'],
    default=['Alto', 'Medio']
)

planes_sel = st.sidebar.multiselect(
    'Plan',
    sorted(df_raw['plan'].unique()),
    default=sorted(df_raw['plan'].unique())
)

paises_sel = st.sidebar.multiselect(
    'País',
    sorted(df_raw['pais'].unique()),
    default=sorted(df_raw['pais'].unique())
)

df = df_raw[
    df_raw['riesgo'].isin(riesgo_sel) &
    df_raw['plan'].isin(planes_sel) &
    df_raw['pais'].isin(paises_sel)
].copy()

# ── MÉTRICAS PRINCIPALES ──────────────────────────────────────────────────────

total       = len(df_raw)
alto_riesgo = (df_raw['riesgo'] == 'Alto').sum()
medio_riesgo= (df_raw['riesgo'] == 'Medio').sum()
bajo_riesgo = (df_raw['riesgo'] == 'Bajo').sum()
churn_rate  = df_raw['prob_churn'].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Total estudiantes',  f'{total:,}')
c2.metric('🔴 Riesgo Alto',    f'{alto_riesgo:,}',  f'{alto_riesgo/total:.1%}')
c3.metric('🟡 Riesgo Medio',   f'{medio_riesgo:,}', f'{medio_riesgo/total:.1%}')
c4.metric('🟢 Riesgo Bajo',    f'{bajo_riesgo:,}',  f'{bajo_riesgo/total:.1%}')
c5.metric('Prob. churn promedio', f'{churn_rate:.1%}')

st.markdown('---')

# ── GRÁFICOS ──────────────────────────────────────────────────────────────────

col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader('Riesgo por Plan')
    churn_plan = df_raw.groupby('plan')['prob_churn'].mean().reset_index()
    churn_plan.columns = ['Plan', 'Prob. Churn Promedio']
    churn_plan = churn_plan.sort_values('Prob. Churn Promedio', ascending=True)

    fig = px.bar(
        churn_plan, x='Prob. Churn Promedio', y='Plan',
        orientation='h',
        color='Prob. Churn Promedio',
        color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
        text=churn_plan['Prob. Churn Promedio'].map('{:.1%}'.format)
    )
    fig.update_layout(coloraxis_showscale=False, height=300,
                      margin=dict(t=10, b=10))
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

with col_der:
    st.subheader('Distribución de probabilidad de Churn')
    fig2 = px.histogram(
        df_raw, x='prob_churn', nbins=40,
        color='riesgo',
        color_discrete_map={'Alto': '#e74c3c', 'Medio': '#f39c12', 'Bajo': '#2ecc71'},
        labels={'prob_churn': 'Probabilidad de Churn', 'count': 'Estudiantes'},
        category_orders={'riesgo': ['Alto', 'Medio', 'Bajo']}
    )
    fig2.update_layout(height=300, margin=dict(t=10, b=10), bargap=0.05)
    st.plotly_chart(fig2, use_container_width=True)

col_izq2, col_der2 = st.columns(2)

with col_izq2:
    st.subheader('Riesgo por País')
    churn_pais = df_raw.groupby('pais')['prob_churn'].mean().reset_index()
    churn_pais.columns = ['País', 'Prob. Churn']
    churn_pais = churn_pais.sort_values('Prob. Churn', ascending=True)

    fig3 = px.bar(
        churn_pais, x='Prob. Churn', y='País',
        orientation='h',
        color='Prob. Churn',
        color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
        text=churn_pais['Prob. Churn'].map('{:.1%}'.format)
    )
    fig3.update_layout(coloraxis_showscale=False, height=380,
                       margin=dict(t=10, b=10))
    fig3.update_traces(textposition='outside')
    st.plotly_chart(fig3, use_container_width=True)

with col_der2:
    st.subheader('Riesgo por Escuela')
    churn_esc = df_raw.groupby('escuela_principal')['prob_churn'].mean().reset_index()
    churn_esc.columns = ['Escuela', 'Prob. Churn']
    churn_esc['Escuela'] = churn_esc['Escuela'].str.replace('Escuela de ', '')
    churn_esc = churn_esc.sort_values('Prob. Churn', ascending=True)

    fig4 = px.bar(
        churn_esc, x='Prob. Churn', y='Escuela',
        orientation='h',
        color='Prob. Churn',
        color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
        text=churn_esc['Prob. Churn'].map('{:.1%}'.format)
    )
    fig4.update_layout(coloraxis_showscale=False, height=380,
                       margin=dict(t=10, b=10))
    fig4.update_traces(textposition='outside')
    st.plotly_chart(fig4, use_container_width=True)

# ── TABLA DE ESTUDIANTES EN RIESGO ────────────────────────────────────────────

st.markdown('---')
st.subheader(f'Lista de estudiantes en riesgo — {len(df):,} resultados')

cols_tabla = [
    'student_id', 'plan', 'pais', 'escuela_principal',
    'dias_desde_ultimo_login', 'sesiones_ultimo_mes',
    'tasa_completado', 'prob_churn', 'riesgo'
]

df_tabla = df[cols_tabla].copy()
df_tabla['prob_churn']     = (df_tabla['prob_churn'] * 100).round(1).astype(str) + '%'
df_tabla['tasa_completado']= (df_tabla['tasa_completado'] * 100).round(0).astype(int).astype(str) + '%'
df_tabla.columns = [
    'ID', 'Plan', 'País', 'Escuela',
    'Días sin entrar', 'Sesiones/mes',
    'Completado', 'Prob. Churn', 'Riesgo'
]
df_tabla = df_tabla.sort_values('Prob. Churn', ascending=False)

def color_riesgo(val):
    if val == 'Alto':
        return 'background-color: #fadbd8'
    elif val == 'Medio':
        return 'background-color: #fef9e7'
    return 'background-color: #d5f5e3'

st.dataframe(
    df_tabla.style.map(color_riesgo, subset=['Riesgo']),
    use_container_width=True,
    height=400
)

# ── EXPORTAR ──────────────────────────────────────────────────────────────────

st.markdown('---')
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    csv_export = df[cols_tabla].sort_values('prob_churn', ascending=False).to_csv(index=False)
    st.download_button(
        label='⬇️ Descargar lista completa (CSV)',
        data=csv_export,
        file_name='platzi_churn_riesgo.csv',
        mime='text/csv'
    )

with col_exp2:
    alto_export = df[df['riesgo'] == 'Alto'][cols_tabla].sort_values('prob_churn', ascending=False).to_csv(index=False)
    st.download_button(
        label='⬇️ Descargar solo Riesgo Alto (CSV)',
        data=alto_export,
        file_name='platzi_churn_alto_riesgo.csv',
        mime='text/csv'
    )

st.caption('Platzi ML Platform · Módulo 1: Churn Prediction')
