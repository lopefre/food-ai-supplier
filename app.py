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
st.set_page_config(page_title="Поиск Поставщиков Food", layout="wide")
st.title("🥑 Сервис поиска и анализа поставщиков питания")

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

# 3. Функция ИИ-анализа текста
def analyze_with_ai(text: str, url: str, key: str):
    try:
        # Инициализируем базовую модель напрямую, без автоматического структурирования LangChain
        llm = ChatMistralAI(model="open-mistral-7b", temperature=0, mistral_api_key=key)
        
        # Мы явно просим модель вернуть JSON и описываем схему прямо в промпте.
        system_instruction = """Ты — ИИ-ассистент отдела закупок. Изучи сырой текст веб-страницы поставщика 
        и извлеки из него ключевую информацию.
        
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
        
        # Делаем обычный вызов модели
        response = llm.invoke(prompt)
        
        # Извлекаем сырой текст ответа
        raw_content = response.content.strip()
        
        # На всякий случай очищаем от возможных маркдаун-тегов, если модель их добавила
        if raw_content.startswith("```"):
            lines = raw_content.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                raw_content = "\n".join(lines[1:-1])
        
        # Парсим строку в стандартный Python-словарь
        data = json.loads(raw_content)
        
        # Оборачиваем обратно в Pydantic модель, чтобы остальной код Streamlit не сломался
        return SupplierInfo(
            company_name=data.get("company_name", "Неизвестный поставщик"),
            min_order=data.get("min_order"),
            delivery_terms=data.get("delivery_terms"),
            contacts=data.get("contacts", [])
        )
        
    except json.JSONDecodeError:
        # Если модель выдала невалидный JSON
        st.error("⚠️ Ошибка: Модель вернула некорректный формат данных. Пробую вернуть дефолтную карточку.")
        return SupplierInfo(company_name="Ошибка парсинга JSON", contacts=[])
        
    except Exception as e:
        # Ловим сетевую ошибку httpx или ошибку авторизации
        st.error(f"❌ Произошла ошибка при обращении к ИИ-модели: {str(e)}")
        st.info("💡 Проверьте, правильно ли указан MISTRAL_API_KEY в Advanced Settings -> Secrets на Streamlit Cloud.")
        # Возвращаем пустой объект, чтобы приложение не падало в белый экран, а продолжало работать
        return SupplierInfo(company_name="Ошибка авторизации/сети API", contacts=["Проверьте ключи в Secrets"])


# 4. Интерфейс пользователя
query = st.text_input("Введите категорию товара (например: упаковка или молоко):").lower().strip()

if query:
    if not api_key:
        st.warning("Пожалуйста, введите API ключ в боковой панели для работы ИИ.")
    else:
        # Ищем совпадения в нашей базе сайтов
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
                    # Передаем текст страницы (имитируя скрейпинг) в Mistral AI
                    ai_data = analyze_with_ai(site["text"], site["url"], api_key)
                    results.append(ai_data)
            time.sleep(2)
            # Отображаем красивые карточки (Выполнение ТЗ: "помогать принять решение")
            cols = st.columns(len(results))
            for idx, card in enumerate(results):
                with cols[idx]:
                    st.markdown("### 🏢")
                    st.subheader(card.company_name)
                    st.markdown(f"🌐 **Сайт:** {MOCK_SERP[matched_category][idx]['url']}")
                    st.markdown(f"📦 **Минимальный заказ:** {card.min_order or 'Не указан'}")
                    st.markdown(f"🚚 **Условия доставки:** {card.delivery_terms or 'По запросу'}")
                    st.markdown(f"📞 **Контакты:** {', '.join(card.contacts)}")
            
            # Агрегированное сравнение
                        # СУПЕР-ФИЧА ДЛЯ ИИ-ИНЖЕНЕРА: Динамический ИИ-анализ с глубокой проработкой
            st.divider()
            st.subheader("🤖 ИИ-Аналитика и экспертное заключение")
            
            with st.spinner("ИИ проводит глубокий сравнительный анализ контрагентов..."):
                llm_compare = ChatMistralAI(model="open-mistral-7b", temperature=0.2, mistral_api_key=api_key)
                all_cards_text = "\n".join([f"Компания: {c.company_name}, Мин.заказ: {c.min_order}, Доставка: {c.delivery_terms}, Контакты: {c.contacts}" for c in results])
                
                # Жестко требуем от ИИ развернутый, детальный и длинный разбор по нашей структуре
                compare_prompt = f"""Ты — главный эксперт департамента снабжения и логистики в сфере Food. 
                Проведи детальный, глубокий и развернутый сравнительный анализ поставщиков под запрос '{query}'.
                
                Ты ДОЛЖЕН структурировать свой ответ строго с использованием следующих разделителей (маркеров), чтобы я мог распарсить твой текст в интерфейсе. Напиши много полезной информации для каждого блока.
                
                Формат твоего ответа:
                
                [МЕЛКИЙ_ОПТ]
                Здесь напиши ОЧЕНЬ ПОДРОБНЫЙ разбор поставщика, который лучше всего подходит для мелкого опта, стартапов или работы по Москве. Проанализируй его плюсы, ценовые пороги (например, 5000 руб.), гибкость, скорость оформления заказа и логистические преимущества. Напиши минимум 3-4 развернутых предложения.
                
                [КРУПНЫЙ_ОПТ]
                Здесь напиши ОЧЕНЬ ПОДРОБНЫЙ разбор поставщика для крупного опта, производств или работы по ЦФО. Распиши особенности работы от 1 паллеты, наличие собственного автопарка, экономическую выгоду от масштаба и стабильность поставок. Напиши минимум 3-4 развернутых предложения.
                
                [ЧЕК_ЛИСТ]
                Здесь составь подробный список критически важных вопросов для закупщика (что уточнить по ассортименту, срокам доставки из Подольска/Москвы, документам ХАССП/Меркурий, условиям отсрочки платежа).
                
                [ВЕРДИКТ]
                Итоговое стратегическое резюме: как закупщику правильно распределить бюджет между этими контрагентами для максимальной выгоды.
                
                Данные для анализа:
                {all_cards_text}"""
                
                summary = llm_compare.invoke(compare_prompt)
                full_text = summary.content

            # Функция для безопасного вытаскивания блоков из текста ИИ
            def get_block(text, block_name):
                # Ищем текст от нужного маркера до следующего маркера в скобках [следующий] или до конца текста
                pattern = rf"\[{block_name}\](.*?)(?=\[\w+\]|$)"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    # Очищаем от висячих звёздочек, маркеров списков и пустых строк в начале и конце
                    content = re.sub(r"^[\s\*•-]+", "", content)
                    content = re.sub(r"[\s\*•-]+$", "", content)
                    return content.strip()
                return "Аналитика формируется..."

            # Вытаскиваем очищенные тексты блоков
            text_small = get_block(full_text, "МЕЛКИЙ_ОПТ")
            text_large = get_block(full_text, "КРУПНЫЙ_ОПТ")
            text_checklist = get_block(full_text, "ЧЕК_ЛИСТ")
            text_verdict = get_block(full_text, "ВЕРДИКТ")

            # Отрисовываем интерфейс 
            st.markdown("### 🎯 Стратегические сценарии закупки")
            tab1, tab2 = st.tabs(["🚀 Тестирование рынка и мелкий опт", "📦 Масштабирование и крупный опт"])
            
            with tab1:
                with st.container(border=True):
                    st.markdown("#### 💡 Экспертный разбор: Стратегия гибкого старта")
                    st.markdown(text_small)
                
            with tab2:
                with st.container(border=True):
                    st.markdown("#### 🏭 Экспертный разбор: Стратегия优化 затрат")
                    st.markdown(text_large)

            # Выводим динамический чек-лист
            with st.expander("🔍 Инструкция для закупщика: Что проверить перед подписанием договора?", expanded=True):
                st.markdown(text_checklist)
            
            # Финальный красивый акцент на итоговом вердикте
            st.markdown("---")
            st.markdown(f"⚖️ **Итоговая рекомендация ИИ:** {text_verdict}")

            
        else:
            st.info("Пока в базе демо-прототипа есть категории: 'упаковка' и 'молоко'. Попробуйте ввести их.")
