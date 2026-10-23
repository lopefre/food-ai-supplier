import streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
import time
import json
import re


# Настройка страницы Streamlit
st.set_page_config(page_title="Поиск поставщиков Food", layout="wide")
st.title("🥑 Сервис поиска и анализа поставщиков питания")

# Жесткое скрытие сервисного футера хостинга Streamlit Cloud через инъекцию CSS
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDecoration {display:none !important;}
    div[data-testid="stDecoration"] {display:none !important;}
    </style>
    """, unsafe_allow_html=True)


# ГЛОБАЛЬНЫЕ ФУНКЦИИ ОЧИСТКИ (Вынесены наверх, чтобы быть доступными везде!)
def clean_markdown_artifacts(text):
    if not text:
        return ""
    # Очищаем только явные баги склейки строк модели
    text = text.replace("****", "")
    text = text.replace("** **", " ")
    
    # Убираем дублирование системных заголовков в начале сгенерированного текста
    text = re.sub(r'^(Итоговое резюме|Вердикт|Итоговая рекомендация|Резюме)\s*(для\s+[^:]+)?:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def get_block(text, block_name):
    pattern = rf"\[{block_name}\](.*?)(?=\[\w+\]|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        content = re.sub(r"^[\s\*•-]+", "", content)
        content = re.sub(r"[\s\*•-]+$", "", content)
        return content.strip()
    return ""


# 1. Структура данных для ИИ
class SupplierInfo(BaseModel):
    company_name: str = Field(description="Название компании")
    min_order: Optional[str] = Field(None, description="Минимальный заказ")
    delivery_terms: Optional[str] = Field(None, description="Условия доставки")
    contacts: List[str] = Field(description="Телефоны, email или ссылки")


# Инициализируем модель через секреты Streamlit
if "MISTRAL_API_KEY" in st.secrets:
    api_key = st.secrets["MISTRAL_API_KEY"]
else:
    api_key = st.sidebar.text_input("Введите Mistral API Key", type="password")

# 2. Имитация поисковика (для прототипа)
MOCK_SERP = {
    "упаковка": [
        {"name": "ОптиКом (Упаковка)", "url": "https://opti-com.ru", "text": "ОптиКом — комплексный поставщик упаковки и расходных материалов для фудсервиса. Минимальная сумма заказа для бесплатной доставки по Москве — 5000 рублей. Регионы работы: вся Россия. Контакты: +7 (495) 980-00-00, info@opticom.ru. Вся продукция сертифицирована по ГОСТ и ISO."},
        {"name": "Протэк", "url": "https://protekgroup.com", "text": "Группа компаний Протэк — производство plastic-упаковки и одноразовой посуды. Оптовые поставки от 1 паллеты. Доставка собственным автопарком по ЦФО или самовывоз со склада в Подольске. Телефон: 8 (800) 555-32-21."}
    ],
    "молоко": [
        {"name": "ЭкоНива Опт", "url": "https://ekoniva-apk.ru", "text": "ЭкоНива — сырое и пастеризованное молоко высшего сорта оптом для HoReCa. Минимальный объем от 500 литров. Сертификаты Меркурий, ХАССП. Логистика по всей европейской части РФ. Отдел продаж: sales@ekoniva.ru"}
    ]
}

# 3. Функция ИИ-анализа текста
def analyze_with_ai(text: str, url: str, key: str):
    try:
        llm = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=key)
        system_instruction = """Ты — ИИ-ассистент отдела закупок. Изучи сырой текст веб-страницы поставщика 
        и извлеки из него ключевую информацию. Ответь СТРОГО в формате JSON без разметки ```json.
        JSON должен иметь ключи: company_name, min_order, delivery_terms, contacts."""
        
        prompt = f"{system_instruction}\n\nСайт: {url}\nТекст страницы:\n{text}"
        response = llm.invoke(prompt)
        raw_content = response.content.strip()
        
        if raw_content.startswith("```"):
            lines = raw_content.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                raw_content = "\n".join(lines[1:-1])
        
        data = json.loads(raw_content)
        return SupplierInfo(
            company_name=data.get("company_name", "Неизвестный поставщик"),
            min_order=data.get("min_order"),
            delivery_terms=data.get("delivery_terms"),
            contacts=data.get("contacts", [])
        )
    except:
        return SupplierInfo(company_name="Ошибка извлечения данных", contacts=[])


# 4. Интерфейс пользователя
query = st.text_input("Введите категорию товара (например: упаковка или молоко):").lower().strip()

if query:
    if not api_key:
        st.warning("Пожалуйста, введите API ключ в боковой панели для работы ИИ.")
    else:
        matched_category = None
        for cat in MOCK_SERP.keys():
            if cat in query:
                matched_category = cat
                break
        
        if matched_category:
            st.subheader(f"🔍 Найдено сайтов по запросу '{query}': {len(MOCK_SERP[matched_category])}")
            
            results = []
            with st.spinner("ИИ анализирует содержимое сайтов в реальном времени..."):
                for site in MOCK_SERP[matched_category]:
                    time.sleep(1)
                    ai_data = analyze_with_ai(site["text"], site["url"], api_key)
                    results.append(ai_data)
            
            # Отображение карточек компаний
            if len(results) == 1:
                card = results[0]
                with st.container(border=True):
                    st.markdown("### 🏢")
                    st.subheader(card.company_name)
                    st.markdown(f"🌐 **Сайт:** {MOCK_SERP[matched_category][0]['url']}")
                    st.markdown(f"📦 **Минимальный заказ:** {card.min_order or 'Не указан'}")
                    st.markdown(f"🚚 **Условия доставки:** {card.delivery_terms or 'По запросу'}")
                    st.markdown(f"📞 **Контакты:** {', '.join(card.contacts)}")
            elif len(results) > 1:
                cols = st.columns(len(results))
                for idx, card in enumerate(results):
                    with cols[idx]:
                        with st.container(border=True):
                            st.markdown("### 🏢")
                            st.subheader(card.company_name)
                            st.markdown(f"🌐 **Сайт:** {MOCK_SERP[matched_category][idx]['url']}")
                            st.markdown(f"📦 **Минимальный заказ:** {card.min_order or 'Не указан'}")
                            st.markdown(f"🚚 **Условия доставки:** {card.delivery_terms or 'По запросу'}")
                            st.markdown(f"📞 **Контакты:** {', '.join(card.contacts)}")
            
            # ИИ Аналитика
            st.divider()
            st.subheader("🤖 ИИ-Аналитика и экспертное заключение")
            
            with st.spinner("ИИ проводит глубокий сравнительный анализ контрагентов..."):
                llm_compare = ChatMistralAI(model="open-mistral-7b", temperature=0.1, mistral_api_key=api_key)
                all_cards_text = "\n".join([f"Компания: {c.company_name}, Мин.заказ: {c.min_order}, Доставка: {c.delivery_terms}" for c in results])
                
                compare_prompt = f"""Ты — эксперт департамента снабжения HoReCa. 
                Проведи детальный структурированный разбор поставщиков под запрос '{query}' на основе предоставленных данных.
                
                ⚠️ ПРАВИЛО: Работай ТОЛЬКО с компаниями ниже. Не выдумывай чужие бренды.
                
                Ты ОБЯЗАН использовать в своем ответе следующие теги на отдельных строках, чтобы разделить текст:
                [МЕЛКИЙ_ОПТ]
                (текст разбора условий для небольших заказов)
                
                [КРУПНЫЙ_ОПТ]
                (текст разбора условий для больших объемов)
                
                [ЧЕК_ЛИСТ]
                (список вопросов закупщика перед сделкой)
                
                [ВЕРДИКТ]
                (краткое финальное резюме-рекомендация)
                
                Данные для анализа:
                {all_cards_text}"""
                
                summary = llm_compare.invoke(compare_prompt)
                full_text = summary.content

            # Извлекаем блоки аналитики
            text_small = get_block(full_text, "МЕЛКИЙ_ОПТ")
            text_large = get_block(full_text, "КРУПНЫЙ_ОПТ")
            text_checklist = get_block(full_text, "ЧЕК_ЛИСТ")
            text_verdict = get_block(full_text, "ВЕРДИКТ")
            
            # Если теги успешно найдены
            if text_small and text_large:
                st.markdown("### 🎯 Стратегические сценарии закупки")
                tab1, tab2 = st.tabs(["🚀 Тестирование рынка и мелкий опт", "📦 Масштабирование и крупный опт"])
                
                with tab1:
                    with st.container(border=True):
                        st.markdown("#### 💡 Экспертный разбор: Стратегия гибкого старта")
                        st.markdown(clean_markdown_artifacts(text_small))
                    
                with tab2:
                    with st.container(border=True):
                        st.markdown("#### 🏭 Экспертный разбор: Стратегия оптимизации затрат")
                        st.markdown(clean_markdown_artifacts(text_large))

                if text_checklist:
                    with st.expander("🔍 Инструкция для закупщика: Что проверить перед договором?", expanded=True):
                        st.markdown(clean_markdown_artifacts(text_checklist))
                
                if text_verdict:
                    st.divider()
                    st.markdown("⚖️ **Итоговая рекомендация ИИ:**")
                    st.markdown(clean_markdown_artifacts(text_verdict))
            
            else:
                # ПЛАН Б: Если модель проигнорировала теги, мы выводим полный текст, сохраняя ВСЕ переносы строк и маркдаун
                # Никаких пустых вложенных контейнеров и серой разметки хостинга!
                st.markdown("### 🎯 Полный аналитический разбор контрагентов")
                st.markdown(clean_markdown_artifacts(full_text))

        else:
            st.info("Пока в базе демо-прототипа есть категории: 'упаковка' и 'молоко'. Попробуйте ввести их.")
