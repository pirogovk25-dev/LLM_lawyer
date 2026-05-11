import os
import streamlit as st
import requests

API_BASE = os.getenv("API_BASE", "http://localhost:8001")

st.set_page_config(
    page_title="Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .user-message {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #4a90e2;
    }
    .assistant-message {
        background: #ffffff;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        border: 1px solid #e0e0e0;
        border-left: 3px solid #28a745;
    }
    .web-badge {
        background: #fff3cd;
        color: #856404;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .db-badge {
        background: #d4edda;
        color: #155724;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .stTextInput > div > div > input {
        border-radius: 24px;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ===== САЙДБАР =====
with st.sidebar:
    st.markdown("## ⚖️ Legal AI")
    st.markdown("---")

    st.markdown("### Статус системы")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        qdrant_ok = health.get("services", {}).get("qdrant") == "ok"
        llm_ok = health.get("services", {}).get("llm") == "ok"
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"{'🟢' if qdrant_ok else '🔴'} Qdrant")
        with col2:
            st.markdown(f"{'🟢' if llm_ok else '🔴'} LLM")
    except Exception:
        st.markdown("🔴 Бэкенд недоступен")

    st.markdown("---")

    st.markdown("### База знаний")
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=3).json()
        st.metric("Чанков в базе", stats.get("total_chunks", 0))
        st.metric("Запросов в сессии", st.session_state.total_queries)
    except Exception:
        st.markdown("Статистика недоступна")

    st.markdown("---")

    st.markdown("### Загрузить документы")
    uploaded_files = st.file_uploader(
        "Выберите .docx файлы",
        type=["docx"],
        accept_multiple_files=True,
        help="Загрузите юридические документы для индексации",
    )

    if uploaded_files and st.button("📤 Индексировать", use_container_width=True):
        with st.spinner("Индексация..."):
            files_data = [
                (
                    "files",
                    (
                        f.name,
                        f.read(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                )
                for f in uploaded_files
            ]
            try:
                resp = requests.post(f"{API_BASE}/upload", files=files_data, timeout=60)
                results = resp.json().get("results", [])
                for r in results:
                    if r["status"] == "success":
                        st.success(f"✅ {r['file']}: +{r['chunks_added']} чанков")
                    elif r["status"] == "skipped":
                        st.info(f"⏭️ {r['file']}: {r['reason']}")
                    else:
                        st.error(f"❌ {r['file']}: {r.get('reason', 'ошибка')}")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")

    st.markdown("---")

    if st.button("🗑️ Очистить чат", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

# ===== ОСНОВНАЯ ОБЛАСТЬ =====
st.markdown('<div class="main-header">⚖️ Legal AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Задайте вопрос по юридическим документам</div>', unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message"><strong>👤 Вы</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            is_web = any("[ВЕБ]" in s for s in msg.get("sources", []))
            badge = (
                '<span class="web-badge">🌐 Интернет</span>'
                if is_web
                else '<span class="db-badge">📚 База знаний</span>'
            )
            st.markdown(
                f'<div class="assistant-message"><strong>⚖️ Legal AI</strong> {badge}<br><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                with st.expander(f"📎 Источники ({len(msg['sources'])})"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**{i}.** {src[:300]}...")

st.markdown("---")
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Введите вопрос",
            placeholder="Например: Что такое государственная тайна?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Спросить →", use_container_width=True)

if submitted and question.strip():
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.total_queries += 1

    with st.spinner("Анализирую документы..."):
        try:
            resp = requests.post(
                f"{API_BASE}/ask",
                json={"question": question},
                timeout=120,
            )
            data = resp.json()
            answer = data.get("answer", "Нет ответа")
            sources = data.get("search_results", [])
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Ошибка подключения к API: {e}",
                "sources": [],
            })

    st.rerun()

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; color: #999; margin-top: 3rem;">
        <div style="font-size: 3rem;">⚖️</div>
        <div style="font-size: 1.1rem; margin-top: 1rem;">Задайте вопрос по юридическим документам</div>
        <div style="font-size: 0.85rem; margin-top: 0.5rem;">
            Примеры: «Что такое государственная тайна?», «Права НКО в России»
        </div>
    </div>
    """, unsafe_allow_html=True)
