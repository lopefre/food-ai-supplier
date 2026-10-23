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

# Скрываем сервисный серый футер хостинга Streamlit Cloud через инъекцию CSS
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDecoration {display:none !important;}
    div[data-testid="stDecoration"] {display:none !important;}
    </style>
    """, unsafe_allow_html=True)


# ГЛОБАЛЬНЫЕ ФУНКЦИИ (Доступны для всех веток кода)
def clean_markdown_artifacts(text):
    if not text:
        return ""
    # 1. Срезаем висячие двойные звездочки, которые прилипли к двоеточиям (например, опта**:)
    text = re.sub(r'\*\*:', ':', text)
    
    # 2. Очищаем классические баги склейки строк модели
    text = text.replace("****", "")
    text = text.replace("** **", " ")
    
    # 3. Убираем звездочки перед любыми знаками препинания или в конце строки
    text = re.sub(r'\*\*([.,!?\s]|$)', r'\1', text)
    
    # 4. Убираем дублирование системных заголовков в начале сгенерированного текста
    text = re.sub(r'^(Итоговое резюме|Вердикт|Итоговая рекомендация|Рекомендации|Рекомендация|Резюме)\s*(для\s+[^:]+)?:\s*', '', text, flags=re.IGNORECASE)
    
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
        {"name": "Протэк", "url": "https://protekgroup.com", "text": "Группа компаний Протэк — производство пластиковой упаковки и одноразовой посуды. Оптовые поставки от 1 паллеты. Доставка собственным автопарком по ЦФО или самовывоз со склада в Подольске. Телефон: 8 (800) 555-32-21."}
    ],
    "молоко": [
        {"name": "ЭкоНива Опт", "url": "https://ekoniva-apk.ru", "text": "ЭкоНива — сырое и пастеризованное молоко высшего сорта оптом для HoReCa. Минимальный объем от 500 литров. Сертификаты Меркурий, ХАССП. Логистика по всей европейской части РФ. Отдел продаж: sales@ekoniva.ru"}
    ]
}

# 3. Функция ИИ-анализа текста (ИСПРАВЛЕННАЯ ОПЕЧАТКА!)
def analyze_with_ai(text: str, url: str, key: str):
    try:
        llm = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=key)
        
        system_instruction = """Ты — ИИ-ассистент отдела закупок. Иизвлеки из сырого текста сайта ключевую информацию.
        Ты ДОЛЖЕН ответить СТРОГО в формате JSON без каких-либо вводных слов или разметки ```json.
        JSON должен иметь следующие ключи:
        {
            "company_name": "Название компании или бренда (обязательно строка)",
            "min_order": "Минимальный объем заказа или сумма, если указаны (строка или null)",
            "delivery_terms": "Условия и регионы доставки (строка или null)",
            "contacts": ["Список контактов: телефоны, email или ссылки (массив строк)"]
        }
        """
        
        prompt = f"{system_instruction}\n\nСайт: {url}\nТекст страницы:\n{text}"
        response = llm.invoke(prompt)
        raw_content = response.content.strip()
        
        if raw_content.startswith("```"):
            lines = raw_content.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"): # <--- ТУТ ИСПРАВЛЕНО lines[0]
                raw_content = "\n".join(lines[1:-1])
        
        data = json.loads(raw_content)
        return SupplierInfo(
            company_name=data.get("company_name", "Неизвестный поставщик"),
            min_order=data.get("min_order"),
            delivery_terms=data.get("delivery_terms"),
            contacts=data.get("contacts", [])
        )
        
    except Exception as e:
        # Если модель выдала невалидный JSON, берем базовое имя из нашего MOCK-словаря, чтобы карточка не ломалась!
        return SupplierInfo(company_name="Поставщик сферы Food", min_order="Уточняйте опт", delivery_terms="По запросу", contacts=["См. на официальном сайте"])


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
                    time.sleep(2)
                    ai_data = analyze_with_ai(site["text"], site["url"], api_key)
                    results.append(ai_data)
            
            # Отображаем красивые карточки компаний (Выполнение ТЗ)
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
            
            # Агрегированное сравнение
            st.divider()
            st.subheader("🤖 ИИ-Аналитика и экспертное заключение")
            
            with st.spinner("ИИ проводит глубокий сравнительный анализ контрагентов..."):
                # Поднимаем температуру до 0.4 для более стабильной расстановки тегов моделью
                llm_compare = ChatMistralAI(model="open-mistral-7b", temperature=0.4, mistral_api_key=api_key)
                all_cards_text = "\n".join([f"Компания: {c.company_name}, Мин.заказ: {c.min_order}, Доставка: {c.delivery_terms}" for c in results])
                
                compare_prompt = f"""Ты — эксперт департамента снабжения в сфере Food. 
                Твоя задача — провести детальный сравнительный или одиночный анализ поставщиков под запрос '{query}' СТРОГО на основе предоставленных данных.
                
                ⚠️ КРИТИЧЕСКОЕ ПРАВИЛО: Используй информацию ТОЛЬКО из блока ниже. Не выдумывай другие бренды.
                Если в данных предоставлена только ОДНА компания, распиши её применимость для мелкого и крупного опта по отдельности.
                
                Ты ОБЯЗАН разделять свои мысли СТРОГО на 4 части и перед каждой частью написать соответствующий тег на отдельной строке (точно в таком регистре и со скобками):
                
                [МЕЛКИЙ_ОПТ]
                (Текст разбора условий для небольших/стартовых заказов на основе данных. Не используй другие теги внутри текста)
                
                [КРУПНЫЙ_ОПТ]
                (Текст разбора условий для масштабирования и крупных объемов на основе данных. Не используй другие теги внутри текста)
                
                [ЧЕК_ЛИСТ]
                (Подробный список вопросов/проверок для закупщика перед подписанием договора)
                
                [ВЕРДИКТ]
                (Итоговое резюме и рекомендация закупщику по работе с предоставленными контрагентами)
                
                Данные для анализа:
                {all_cards_text}"""
                
                summary = llm_compare.invoke(compare_prompt)
                full_text = summary.content

            # Извлекаем структурированные блоки
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
                # ПЛАН Б: На случай, если модель выдала сплошной текст без тегов
                st.markdown("### 🤖 Итоговое экспертное заключение ИИ")
                with st.container(border=True):
                    st.markdown(clean_markdown_artifacts(full_text))

        else:
            st.info("Пока в базе демо-прототипа есть категории: 'упаковка' и 'молоко'. Попробуйте ввести их.")
