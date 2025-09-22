import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import numpy as np
import google.generativeai as genai

# Configuración de la página
st.set_page_config(
    page_title="FinanceIA - Tu Asistente Financiero",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para un diseño moderno y minimalista
st.markdown("""
<style>
    /* Tema principal */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    /* Contenedor principal */
    .block-container {
        padding-top: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }

    /* Títulos modernos */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }

    /* Sidebar moderna */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 0 20px 20px 0;
    }

    /* Métricas modernas */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: none;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }

    /* Botones modernos */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    /* Botones secundarios */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
    }

    /* Inputs modernos */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 15px;
        border: 2px solid #e9ecef;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
    }

    /* Alertas y notificaciones */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Animaciones sutiles */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main > div {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Archivo para almacenar los datos
DATA_FILE = "financial_data.json"
CONFIG_FILE = "config.json"

class GeminiFinancialAI:
    def __init__(self):
        self.data = self.load_data()
        self.config = self.load_config()
        # Guardar datos corregidos si hubo cambios en los IDs
        self.save_data()
        self.setup_gemini()

    def load_config(self) -> Dict:
        """Carga la configuración desde el archivo local"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"gemini_api_key": ""}
        return {"gemini_api_key": ""}

    def save_config(self):
        """Guarda la configuración en el archivo local"""
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def setup_gemini(self):
        """Configura la API de Gemini"""
        if self.config.get("gemini_api_key"):
            try:
                genai.configure(api_key=self.config["gemini_api_key"])

                # Lista de modelos a probar en orden de preferencia
                models_to_try = [
                    'gemini-2.0-flash-exp',
                    'gemini-1.5-flash',
                    'gemini-1.5-pro',
                    'models/gemini-2.0-flash-exp',
                    'models/gemini-1.5-flash',
                    'models/gemini-1.5-pro'
                ]

                for model_name in models_to_try:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        self.gemini_available = True
                        self.model_name = model_name
                        st.success(f"✅ Conectado con {model_name}")
                        break
                    except Exception as model_error:
                        continue

                if not hasattr(self, 'gemini_available') or not self.gemini_available:
                    self.gemini_available = False
                    st.error("❌ No se encontró ningún modelo disponible")

            except Exception as e:
                self.gemini_available = False
                st.error(f"Error al configurar Gemini: {str(e)}")
        else:
            self.gemini_available = False

    def set_api_key(self, api_key: str):
        """Establece la API key de Gemini"""
        self.config["gemini_api_key"] = api_key
        self.save_config()
        self.setup_gemini()

    def load_data(self) -> Dict:
        """Carga los datos desde el archivo local"""
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Corregir IDs duplicados
                    self._fix_duplicate_ids(data)
                    return data
            except:
                return {"income": [], "expenses": [], "goals": [], "user_profile": {}}
        return {"income": [], "expenses": [], "goals": [], "user_profile": {}}

    def _fix_duplicate_ids(self, data: Dict):
        """Corrige IDs duplicados en los datos"""
        # Corregir IDs de ingresos
        for i, income in enumerate(data.get("income", [])):
            income["id"] = i + 1

        # Corregir IDs de gastos
        for i, expense in enumerate(data.get("expenses", [])):
            expense["id"] = i + 1

    def save_data(self):
        """Guarda los datos en el archivo local"""
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def add_income(self, amount: float, source: str, date_str: str, category: str = "Salario"):
        """Añade un ingreso"""
        # Calcular ID único basado en el máximo existente
        existing_ids = [item["id"] for item in self.data["income"]] if self.data["income"] else [0]
        new_id = max(existing_ids) + 1 if existing_ids else 1

        income_entry = {
            "amount": amount,
            "source": source,
            "date": date_str,
            "category": category,
            "id": new_id
        }
        self.data["income"].append(income_entry)
        self.save_data()

    def add_expense(self, amount: float, description: str, date_str: str, category: str):
        """Añade un gasto"""
        # Calcular ID único basado en el máximo existente
        existing_ids = [item["id"] for item in self.data["expenses"]] if self.data["expenses"] else [0]
        new_id = max(existing_ids) + 1 if existing_ids else 1

        expense_entry = {
            "amount": amount,
            "description": description,
            "date": date_str,
            "category": category,
            "id": new_id
        }
        self.data["expenses"].append(expense_entry)
        self.save_data()

    def set_user_profile(self, profile: Dict):
        """Establece el perfil del usuario"""
        self.data["user_profile"] = profile
        self.save_data()

    def delete_income(self, income_id: int):
        """Elimina un ingreso por su ID"""
        self.data["income"] = [item for item in self.data["income"] if item["id"] != income_id]
        self.save_data()

    def delete_expense(self, expense_id: int):
        """Elimina un gasto por su ID"""
        self.data["expenses"] = [item for item in self.data["expenses"] if item["id"] != expense_id]
        self.save_data()

    def get_total_income(self) -> float:
        """Calcula el total de ingresos"""
        return sum(item["amount"] for item in self.data["income"])

    def get_total_expenses(self) -> float:
        """Calcula el total de gastos"""
        return sum(item["amount"] for item in self.data["expenses"])

    def get_balance(self) -> float:
        """Calcula el balance actual"""
        return self.get_total_income() - self.get_total_expenses()

    def get_expenses_by_category(self) -> Dict[str, float]:
        """Agrupa gastos por categoría"""
        categories = {}
        for expense in self.data["expenses"]:
            category = expense["category"]
            categories[category] = categories.get(category, 0) + expense["amount"]
        return categories

    def get_financial_summary(self) -> str:
        """Genera un resumen financiero para Gemini"""
        total_income = self.get_total_income()
        total_expenses = self.get_total_expenses()
        balance = self.get_balance()
        expenses_by_category = self.get_expenses_by_category()

        profile = self.data.get("user_profile", {})

        summary = f"""
        PERFIL FINANCIERO DEL USUARIO:

        INFORMACIÓN PERSONAL:
        - Edad: {profile.get('age', 'No especificada')}
        - Ocupación: {profile.get('occupation', 'No especificada')}
        - Situación familiar: {profile.get('family_status', 'No especificada')}
        - Objetivos financieros: {profile.get('financial_goals', 'No especificados')}

        RESUMEN FINANCIERO ACTUAL:
        - Ingresos totales: S/{total_income:,.2f}
        - Gastos totales: S/{total_expenses:,.2f}
        - Balance actual: S/{balance:,.2f}
        - Tasa de ahorro: {(balance/total_income*100) if total_income > 0 else 0:.1f}%

        DISTRIBUCIÓN DE GASTOS POR CATEGORÍA:
        """

        for category, amount in expenses_by_category.items():
            percentage = (amount / total_expenses * 100) if total_expenses > 0 else 0
            summary += f"- {category}: S/{amount:,.2f} ({percentage:.1f}%)\n"

        recent_expenses = sorted(self.data["expenses"], key=lambda x: x["date"], reverse=True)[:10]
        if recent_expenses:
            summary += "\nGASTOS RECIENTES (ÚLTIMOS 10):\n"
            for expense in recent_expenses:
                summary += f"- {expense['date']}: {expense['description']} - S/{expense['amount']:,.2f} ({expense['category']})\n"

        recent_income = sorted(self.data["income"], key=lambda x: x["date"], reverse=True)[:5]
        if recent_income:
            summary += "\nINGRESOS RECIENTES (ÚLTIMOS 5):\n"
            for income in recent_income:
                summary += f"- {income['date']}: {income['source']} - S/{income['amount']:,.2f} ({income['category']})\n"

        return summary

    def get_gemini_analysis(self) -> str:
        """Obtiene análisis de Gemini - FUNCIÓN SÍNCRONA"""
        if not self.gemini_available or not hasattr(self, 'model'):
            return "❌ Gemini no está disponible. Por favor configura tu API key."

        if self.get_total_income() == 0 and len(self.data["expenses"]) == 0:
            return "📝 No hay datos suficientes para realizar un análisis. Comienza registrando tus ingresos y gastos."

        try:
            financial_summary = self.get_financial_summary()
            prompt = f"""Eres un experto asesor financiero personal. Analiza esta información financiera y proporciona:

1. ANÁLISIS DETALLADO de la situación financiera actual
2. RECOMENDACIONES ESPECÍFICAS para mejorar las finanzas
3. ÁREAS DE OPTIMIZACIÓN identificando gastos innecesarios
4. ESTRATEGIAS DE AHORRO personalizadas

INFORMACIÓN FINANCIERA:
{financial_summary}

Responde en español con emojis y estructura clara."""

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_specific_recommendation(self, question: str) -> str:
        """Obtiene una recomendación específica de Gemini - FUNCIÓN SÍNCRONA"""
        if not self.gemini_available or not hasattr(self, 'model'):
            return "❌ Gemini no está disponible. Por favor configura tu API key."

        try:
            financial_summary = self.get_financial_summary()
            prompt = f"""Eres un asesor financiero personal. Responde esta pregunta: "{question}"

Basándote en esta información financiera:
{financial_summary}

Proporciona una respuesta práctica y personalizada en español con emojis."""

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"❌ Error: {str(e)}"

def main():
    # Header moderno y atractivo
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0;">💰 FinanceIA</h1>
        <p style="font-size: 1.2rem; color: #6c757d; margin-top: 0;">
            Tu asistente financiero inteligente diseñado para universitarios
        </p>
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    height: 3px; width: 100px; margin: 1rem auto; border-radius: 3px;">
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Inicializar la IA financiera
    if 'financial_ai' not in st.session_state:
        st.session_state.financial_ai = GeminiFinancialAI()

    ai = st.session_state.financial_ai

    # Verificar configuración de API
    if not ai.gemini_available:
        show_api_setup(ai)
        return

    # Sidebar para navegación
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #2c3e50; margin: 0;">🧭 Navegación</h2>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        height: 2px; width: 60px; margin: 0.5rem auto; border-radius: 2px;">
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Opciones de navegación con iconos modernos
        menu_options = {
            "🏠 Dashboard": "Dashboard",
            "📝 Ingresar Datos": "Ingresar Datos",
            "🧠 Análisis IA": "Análisis con Gemini",
            "💬 Consultas": "Consulta Personalizada",
            "👤 Mi Perfil": "Perfil de Usuario",
            "📚 Historial": "Historial",
            "⚙️ Configuración": "Configuración"
        }

        page = st.selectbox("", list(menu_options.keys()),
                           format_func=lambda x: x,
                           label_visibility="collapsed")
        page = menu_options[page]

    if page == "Dashboard":
        show_dashboard(ai)
    elif page == "Ingresar Datos":
        show_data_input(ai)
    elif page == "Análisis con Gemini":
        show_gemini_analysis(ai)
    elif page == "Consulta Personalizada":
        show_custom_query(ai)
    elif page == "Perfil de Usuario":
        show_user_profile(ai)
    elif page == "Historial":
        show_history(ai)
    elif page == "Configuración":
        show_settings(ai)

def show_api_setup(ai: GeminiFinancialAI):
    """Muestra la configuración de la API de Gemini"""
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #2c3e50; font-weight: 600;">🔑 Configuración de IA</h2>
        <p style="color: #6c757d;">Conecta tu cuenta de Google Gemini para análisis inteligentes</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 2rem; border-radius: 15px; margin: 2rem 0;">
        <h3 style="color: #2c3e50; margin-top: 0;">📋 Pasos para configurar:</h3>
        <ol style="color: #495057; line-height: 1.8;">
            <li>Ve a <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #667eea;">Google AI Studio</a></li>
            <li>Inicia sesión con tu cuenta de Google</li>
            <li>Crea una nueva API key</li>
            <li>Copia la API key y pégala abajo</li>
        </ol>
        <div style="background: #fff3cd; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
            <strong>💡 Nota:</strong> La aplicación detectará automáticamente el mejor modelo disponible (como <code>gemini-2.0-flash-exp</code> o <code>gemini-1.5-flash</code>).
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key = st.text_input("", type="password",
                               placeholder="Pega tu API key aquí (ej: AIza...)",
                               label_visibility="collapsed")

        if st.button("🚀 Conectar con Gemini", type="primary", use_container_width=True):
            if api_key:
                with st.spinner("🔄 Configurando conexión con Gemini..."):
                    ai.set_api_key(api_key)
                    if ai.gemini_available:
                        st.balloons()
                        model_used = getattr(ai, 'model_name', 'Gemini')
                        st.success(f"🎉 ¡Conexión exitosa con {model_used}! Ya puedes usar los análisis de IA")
                        st.rerun()
                    else:
                        st.error("❌ No se pudo conectar. Verifica que tu API key sea correcta.")
            else:
                st.error("⚠️ Por favor ingresa una API key válida.")

    st.markdown("""
    <div style="text-align: center; margin: 2rem 0; padding: 1rem;
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border-radius: 15px; border-left: 4px solid #2196f3;">
        <strong>🔒 Privacidad:</strong> Tu API key se guarda solo en tu dispositivo y nunca se comparte.
    </div>
    """, unsafe_allow_html=True)

def show_dashboard(ai: GeminiFinancialAI):
    """Muestra el dashboard principal"""
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #2c3e50; font-weight: 600;">📊 Dashboard Financiero</h2>
        <p style="color: #6c757d;">Resumen de tu situación financiera actual</p>
    </div>
    """, unsafe_allow_html=True)

    total_income = ai.get_total_income()
    total_expenses = ai.get_total_expenses()
    balance = ai.get_balance()
    savings_rate = (balance / total_income * 100) if total_income > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 25px rgba(0, 184, 148, 0.3);">
            <h3 style="margin: 0; font-size: 1rem;">💰 Ingresos</h3>
            <h2 style="margin: 0.5rem 0 0 0; font-size: 1.8rem;">S/{total_income:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 25px rgba(253, 121, 168, 0.3);">
            <h3 style="margin: 0; font-size: 1rem;">💸 Gastos</h3>
            <h2 style="margin: 0.5rem 0 0 0; font-size: 1.8rem;">S/{total_expenses:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        balance_color = "#00b894" if balance >= 0 else "#e17055"
        balance_icon = "⚖️" if balance >= 0 else "⚠️"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {balance_color} 0%, {balance_color}dd 100%);
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);">
            <h3 style="margin: 0; font-size: 1rem;">{balance_icon} Balance</h3>
            <h2 style="margin: 0.5rem 0 0 0; font-size: 1.8rem;">S/{balance:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        savings_color = "#667eea" if savings_rate >= 20 else "#fdcb6e" if savings_rate >= 10 else "#e17055"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {savings_color} 0%, {savings_color}dd 100%);
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);">
            <h3 style="margin: 0; font-size: 1rem;">📈 Ahorro</h3>
            <h2 style="margin: 0.5rem 0 0 0; font-size: 1.8rem;">{savings_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if total_expenses > 0:
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0 1rem 0;">
            <h3 style="color: #2c3e50;">📊 Análisis Visual</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            expenses_by_category = ai.get_expenses_by_category()
            fig_pie = px.pie(
                values=list(expenses_by_category.values()),
                names=list(expenses_by_category.keys()),
                title="💰 Distribución de Gastos por Categoría",
                color_discrete_sequence=['#667eea', '#764ba2', '#fd79a8', '#00b894', '#fdcb6e', '#e17055', '#74b9ff']
            )
            fig_pie.update_layout(
                title_font_size=16,
                title_font_color='#2c3e50',
                font=dict(size=12),
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name='Ingresos',
                x=['Tu Balance'],
                y=[total_income],
                marker_color='#00b894',
                marker_line_color='rgba(0,0,0,0)',
                marker_line_width=0
            ))
            fig_bar.add_trace(go.Bar(
                name='Gastos',
                x=['Tu Balance'],
                y=[total_expenses],
                marker_color='#fd79a8',
                marker_line_color='rgba(0,0,0,0)',
                marker_line_width=0
            ))
            fig_bar.update_layout(
                title="📈 Ingresos vs Gastos",
                title_font_size=16,
                title_font_color='#2c3e50',
                barmode='group',
                yaxis_title="Cantidad (S/)",
                font=dict(size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    if total_income > 0 or len(ai.data["expenses"]) > 0:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0 1rem 0;">
            <h3 style="color: #2c3e50;">🧠 Análisis Inteligente</h3>
            <p style="color: #6c757d;">Obtén insights personalizados sobre tus finanzas</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Obtener Análisis IA", type="primary", use_container_width=True):
                with st.spinner("🤖 Analizando tus finanzas..."):
                    analysis = st.session_state.get('quick_analysis')
                    if not analysis:
                        analysis = ai.get_gemini_analysis()
                        st.session_state['quick_analysis'] = analysis

                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                                padding: 2rem; border-radius: 15px; margin: 1rem 0;
                                border-left: 4px solid #667eea;">
                    """, unsafe_allow_html=True)
                    st.markdown(analysis)
                    st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    border-radius: 15px; margin: 2rem 0;">
            <h3 style="color: #6c757d;">💡 ¡Comienza tu viaje financiero!</h3>
            <p style="color: #6c757d;">Registra tus primeros ingresos y gastos para ver análisis personalizados</p>
        </div>
        """, unsafe_allow_html=True)

def show_data_input(ai: GeminiFinancialAI):
    """Muestra la interfaz para ingresar datos"""
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #2c3e50; font-weight: 600;">📝 Registro de Movimientos</h2>
        <p style="color: #6c757d;">Registra tus ingresos y gastos de manera rápida y sencilla</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💰 Agregar Ingreso", "💸 Agregar Gasto"])

    with tab1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
                    color: white; padding: 1rem; border-radius: 15px 15px 0 0; margin-bottom: 2rem;">
            <h3 style="margin: 0; text-align: center;">💰 Nuevo Ingreso</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**💵 Datos del Ingreso**")
            income_amount = st.number_input("Cantidad (S/)", min_value=0.01, step=0.01,
                                          value=st.session_state.get("income_amount_value", 0.01),
                                          key="income_amount_input",
                                          help="Ingresa el monto que recibiste")
            income_source = st.text_input("Fuente de ingreso", placeholder="ej. Trabajo de medio tiempo, Venta de productos",
                                        value=st.session_state.get("income_source_value", ""),
                                        key="income_source_input",
                                        help="¿De dónde proviene este dinero?")

        with col2:
            st.markdown("**📅 Detalles Adicionales**")
            income_date = st.date_input("Fecha", value=st.session_state.get("income_date_value", date.today()),
                                      key="income_date_input",
                                      help="¿Cuándo recibiste este ingreso?")
            income_category = st.selectbox("Categoría",
                                         ["Salario", "Freelance", "Inversiones", "Venta", "Bono", "Pensión", "Alquiler", "Otro"],
                                         index=st.session_state.get("income_category_index", 0),
                                         key="income_category_input",
                                         help="Selecciona el tipo de ingreso")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💰 Registrar Ingreso", type="primary", use_container_width=True):
                if income_amount > 0 and income_source:
                    ai.add_income(income_amount, income_source, str(income_date), income_category)
                    st.success(f"✅ ¡Ingreso de S/{income_amount:,.2f} registrado exitosamente!")
                    # Limpiar campos del formulario
                    st.session_state["income_amount_value"] = 0.01
                    st.session_state["income_source_value"] = ""
                    st.session_state["income_date_value"] = date.today()
                    st.session_state["income_category_index"] = 0
                    # Forzar limpieza de los inputs con keys
                    if "income_amount_input" in st.session_state:
                        del st.session_state["income_amount_input"]
                    if "income_source_input" in st.session_state:
                        del st.session_state["income_source_input"]
                    if "income_date_input" in st.session_state:
                        del st.session_state["income_date_input"]
                    if "income_category_input" in st.session_state:
                        del st.session_state["income_category_input"]
                    if 'quick_analysis' in st.session_state:
                        del st.session_state['quick_analysis']
                    st.rerun()
                else:
                    st.error("⚠️ Por favor completa todos los campos obligatorios.")

    with tab2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
                    color: white; padding: 1rem; border-radius: 15px 15px 0 0; margin-bottom: 2rem;">
            <h3 style="margin: 0; text-align: center;">💸 Nuevo Gasto</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**💳 Datos del Gasto**")
            expense_amount = st.number_input("Cantidad (S/)", min_value=0.01, step=0.01,
                                           value=st.session_state.get("expense_amount_value", 0.01),
                                           key="expense_amount_input",
                                           help="¿Cuánto gastaste?")
            expense_description = st.text_input("Descripción", placeholder="ej. Almuerzo en la universidad, Pasaje",
                                              value=st.session_state.get("expense_description_value", ""),
                                              key="expense_description_input",
                                              help="Describe brevemente en qué gastaste")

        with col2:
            st.markdown("**📅 Detalles Adicionales**")
            expense_date = st.date_input("Fecha", value=st.session_state.get("expense_date_value", date.today()),
                                       key="expense_date_input",
                                       help="¿Cuándo realizaste este gasto?")
            expense_category = st.selectbox("Categoría",
                                          ["Alimentación", "Transporte", "Vivienda", "Servicios", "Salud",
                                           "Entretenimiento", "Ropa", "Educación", "Tecnología", "Deudas", "Otro"],
                                          index=st.session_state.get("expense_category_index", 0),
                                          key="expense_category_input",
                                          help="¿En qué categoría clasificarías este gasto?")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💸 Registrar Gasto", type="primary", use_container_width=True):
                if expense_amount > 0 and expense_description:
                    ai.add_expense(expense_amount, expense_description, str(expense_date), expense_category)
                    st.success(f"✅ ¡Gasto de S/{expense_amount:,.2f} registrado exitosamente!")
                    # Limpiar campos del formulario
                    st.session_state["expense_amount_value"] = 0.01
                    st.session_state["expense_description_value"] = ""
                    st.session_state["expense_date_value"] = date.today()
                    st.session_state["expense_category_index"] = 0
                    if 'quick_analysis' in st.session_state:
                        del st.session_state['quick_analysis']
                    st.rerun()
                else:
                    st.error("⚠️ Por favor completa todos los campos obligatorios.")

def show_gemini_analysis(ai: GeminiFinancialAI):
    """Muestra análisis completo con Gemini"""
    st.header("🧠 Análisis Completo con Gemini AI")
    st.markdown("Obtén un análisis detallado y recomendaciones personalizadas para optimizar tus finanzas.")

    if st.button("🚀 Generar Análisis Completo", type="primary", use_container_width=True):
        with st.spinner("🤖 Gemini está analizando tus finanzas..."):
            analysis = ai.get_gemini_analysis()

            st.markdown("---")
            st.subheader("📋 Análisis Detallado de tu Situación Financiera")
            st.markdown(analysis)

            st.session_state['last_full_analysis'] = analysis

    if 'last_full_analysis' in st.session_state:
        st.markdown("---")
        st.subheader("📄 Último Análisis Realizado")
        with st.expander("Ver análisis anterior", expanded=False):
            st.markdown(st.session_state['last_full_analysis'])

def show_custom_query(ai: GeminiFinancialAI):
    """Muestra interfaz para consultas personalizadas"""
    st.header("💬 Consulta Personalizada con Gemini")
    st.markdown("Hazle cualquier pregunta específica sobre tus finanzas a tu asistente IA.")

    st.subheader("💡 Preguntas Sugeridas")
    col1, col2, col3 = st.columns(3)

    suggested_questions = [
        "¿En qué categoría gasto más dinero?",
        "¿Cómo puedo ahorrar más dinero?",
        "¿Cuál es mi mayor gasto innecesario?",
        "¿Debo invertir mi dinero y dónde?",
        "¿Cómo puedo mejorar mi tasa de ahorro?",
        "¿Qué estrategia me recomiendas para este mes?"
    ]

    for i, question in enumerate(suggested_questions):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(f"❓ {question}", key=f"q_{i}"):
                st.session_state['custom_question'] = question

    st.markdown("---")
    custom_question = st.text_area(
        "🤔 Tu pregunta personalizada:",
        value=st.session_state.get('custom_question', ''),
        placeholder="Ej: ¿Cómo puedo reducir mis gastos en entretenimiento sin afectar mi calidad de vida?",
        height=100
    )

    if st.button("🚀 Obtener Respuesta de Gemini", type="primary", disabled=not custom_question):
        if custom_question:
            with st.spinner(f"🤖 Gemini está analizando tu pregunta..."):
                response = ai.get_specific_recommendation(custom_question)

                st.markdown("---")
                st.subheader("🎯 Respuesta Personalizada")
                st.markdown(f"**Tu pregunta:** _{custom_question}_")
                st.markdown("**Respuesta de Gemini:**")
                st.markdown(response)

                if 'custom_question' in st.session_state:
                    del st.session_state['custom_question']

def show_user_profile(ai: GeminiFinancialAI):
    """Muestra la configuración del perfil de usuario"""
    st.header("👤 Perfil de Usuario")
    st.markdown("Comparte información sobre ti para recibir recomendaciones más personalizadas.")

    current_profile = ai.data.get("user_profile", {})

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100,
                             value=current_profile.get("age", 25))

        occupation = st.text_input("Ocupación",
                                  value=current_profile.get("occupation", ""),
                                  placeholder="ej. Ingeniero, Estudiante, Empresario")

        family_status = st.selectbox("Situación Familiar",
                                   ["Soltero/a", "Casado/a", "Con pareja", "Con hijos", "Divorciado/a"],
                                   index=["Soltero/a", "Casado/a", "Con pareja", "Con hijos", "Divorciado/a"].index(current_profile.get("family_status", "Soltero/a")) if current_profile.get("family_status") in ["Soltero/a", "Casado/a", "Con pareja", "Con hijos", "Divorciado/a"] else 0)

    with col2:
        financial_goals = st.text_area("Objetivos Financieros",
                                     value=current_profile.get("financial_goals", ""),
                                     placeholder="ej. Comprar casa, ahorrar para vacaciones, crear fondo de emergencia",
                                     height=100)

        risk_tolerance = st.selectbox("Tolerancia al Riesgo",
                                    ["Conservador", "Moderado", "Agresivo"],
                                    index=["Conservador", "Moderado", "Agresivo"].index(current_profile.get("risk_tolerance", "Moderado")) if current_profile.get("risk_tolerance") in ["Conservador", "Moderado", "Agresivo"] else 1)

        monthly_income_range = st.selectbox("Rango de Ingresos Mensuales",
                                          ["Menos de S/3,000", "S/3,000 - S/9,000", "S/9,000 - S/15,000",
                                           "S/15,000 - S/30,000", "Más de S/30,000"],
                                          index=0 if not current_profile.get("monthly_income_range") else
                                          ["Menos de S/3,000", "S/3,000 - S/9,000", "S/9,000 - S/15,000",
                                           "S/15,000 - S/30,000", "Más de S/30,000"].index(current_profile.get("monthly_income_range")))

    if st.button("💾 Guardar Perfil", type="primary"):
        profile_data = {
            "age": age,
            "occupation": occupation,
            "family_status": family_status,
            "financial_goals": financial_goals,
            "risk_tolerance": risk_tolerance,
            "monthly_income_range": monthly_income_range,
            "updated_at": datetime.now().isoformat()
        }

        ai.set_user_profile(profile_data)
        st.success("✅ Perfil guardado exitosamente! Ahora recibirás recomendaciones más personalizadas.")

def show_history(ai: GeminiFinancialAI):
    """Muestra el historial de transacciones"""
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #2c3e50; font-weight: 600;">📚 Historial de Movimientos</h2>
        <p style="color: #6c757d;">Revisa y gestiona todos tus ingresos y gastos registrados</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💰 Mis Ingresos", "💸 Mis Gastos"])

    with tab1:
        if ai.data["income"]:
            income_sorted = sorted(ai.data["income"], key=lambda x: x["date"], reverse=True)

            for income in income_sorted:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                            padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                            border-left: 4px solid #00b894;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: #2c3e50;">{income['source']}</h4>
                            <p style="margin: 0.5rem 0 0 0; color: #6c757d;">
                                📅 {income['date']} | 🏷️ {income['category']}
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <h3 style="margin: 0; color: #00b894; font-size: 1.5rem;">
                                +S/{income['amount']:,.2f}
                            </h3>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns([4, 1, 4])
                with col2:
                    if st.button("🗑️ Eliminar", key=f"del_income_{income['id']}", help="Eliminar ingreso"):
                        ai.delete_income(income['id'])
                        st.success("✅ Ingreso eliminado")
                        if 'quick_analysis' in st.session_state:
                            del st.session_state['quick_analysis']
                        st.rerun()

            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0; padding: 1rem;
                        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
                        color: white; border-radius: 15px;">
                <h4 style="margin: 0;">📊 Total de ingresos: {len(income_sorted)}</h4>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                        border-radius: 15px; margin: 2rem 0;">
                <h3 style="color: #6c757d;">📝 Sin ingresos registrados</h3>
                <p style="color: #6c757d;">¡Comienza registrando tu primer ingreso!</p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        if ai.data["expenses"]:
            expenses_sorted = sorted(ai.data["expenses"], key=lambda x: x["date"], reverse=True)

            for expense in expenses_sorted:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                            padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                            border-left: 4px solid #fd79a8;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: #2c3e50;">{expense['description']}</h4>
                            <p style="margin: 0.5rem 0 0 0; color: #6c757d;">
                                📅 {expense['date']} | 🏷️ {expense['category']}
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <h3 style="margin: 0; color: #fd79a8; font-size: 1.5rem;">
                                -S/{expense['amount']:,.2f}
                            </h3>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns([4, 1, 4])
                with col2:
                    if st.button("🗑️ Eliminar", key=f"del_expense_{expense['id']}", help="Eliminar gasto"):
                        ai.delete_expense(expense['id'])
                        st.success("✅ Gasto eliminado")
                        if 'quick_analysis' in st.session_state:
                            del st.session_state['quick_analysis']
                        st.rerun()

            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0; padding: 1rem;
                        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
                        color: white; border-radius: 15px;">
                <h4 style="margin: 0;">📊 Total de gastos: {len(expenses_sorted)}</h4>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                        border-radius: 15px; margin: 2rem 0;">
                <h3 style="color: #6c757d;">📝 Sin gastos registrados</h3>
                <p style="color: #6c757d;">¡Registra tu primer gasto para comenzar!</p>
            </div>
            """, unsafe_allow_html=True)

def show_settings(ai: GeminiFinancialAI):
    """Muestra la configuración de la aplicación"""
    st.header("⚙️ Configuración")

    tab1, tab2 = st.tabs(["🔑 API Settings", "🗑️ Gestión de Datos"])

    with tab1:
        st.subheader("Configuración de API de Gemini")

        current_key = ai.config.get("gemini_api_key", "")
        masked_key = f"{current_key[:8]}..." if len(current_key) > 8 else "No configurada"

        st.info(f"API Key actual: {masked_key}")

        new_api_key = st.text_input("Nueva API key:", type="password", placeholder="AIza...")

        if st.button("🔄 Actualizar API Key"):
            if new_api_key:
                ai.set_api_key(new_api_key)
                if ai.gemini_available:
                    st.success("✅ API key actualizada correctamente!")
                else:
                    st.error("❌ Error al configurar la nueva API key.")
            else:
                st.error("⚠️ Por favor ingresa una API key válida.")

    with tab2:
        st.subheader("Gestión de Datos")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Ingresos", len(ai.data["income"]))
        with col2:
            st.metric("📊 Gastos", len(ai.data["expenses"]))
        with col3:
            profile_status = "✅ Configurado" if ai.data.get("user_profile") else "❌ Sin configurar"
            st.metric("👤 Perfil", profile_status)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Exportar Datos"):
                export_data = {
                    **ai.data,
                    "exported_at": datetime.now().isoformat(),
                    "total_income": ai.get_total_income(),
                    "total_expenses": ai.get_total_expenses(),
                    "balance": ai.get_balance()
                }

                st.download_button(
                    label="💾 Descargar archivo JSON",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"financial_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col2:
            if st.button("🗑️ Limpiar Todos los Datos", type="secondary"):
                if st.checkbox("⚠️ Confirmo que deseo eliminar TODOS los datos"):
                    ai.data = {"income": [], "expenses": [], "goals": [], "user_profile": {}}
                    ai.save_data()
                    for key in list(st.session_state.keys()):
                        if 'analysis' in key:
                            del st.session_state[key]
                    st.success("✅ Todos los datos han sido eliminados.")
                    st.rerun()

        st.markdown("---")
        st.info("💡 **Tip**: Exporta regularmente tus datos como respaldo de seguridad.")

if __name__ == "__main__":
    main()