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
            # Если результатов несколько, бьем на колонки. Если один — выводим на всю ширину, чтобы избежать пустых блоков.
            # Отображаем красивые карточки (Выполнение ТЗ: "собирать основную информацию")
            if len(results) == 1:
                # Если поставщик всего один (категория Молоко) — выводим на всю ширину
                card = results[0]
                with st.container(border=True):
                    st.markdown("### 🏢")
                    st.subheader(card.company_name)
                    st.markdown(f"🌐 **Сайт:** {MOCK_SERP[matched_category][0]['url']}")
                    st.markdown(f"📦 **Минимальный заказ:** {card.min_order or 'Не указан'}")
                    st.markdown(f"🚚 **Условия доставки:** {card.delivery_terms or 'По запросу'}")
                    st.markdown(f"📞 **Контакты:** {', '.join(card.contacts)}")
            elif len(results) > 1:
                # Если поставщиков несколько (категория Упаковка) — строим сетку колонок
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
                llm_compare = ChatMistralAI(model="open-mistral-7b", temperature=0.1, mistral_api_key=api_key)
                all_cards_text = "\n".join([f"Компания: {c.company_name}, Мин.заказ: {c.min_order}, Доставка: {c.delivery_terms}, Контакты: {c.contacts}" for c in results])
                
                # Строгий промпт с ограничением контекста (Защита от галлюцинаций ИИ)
                compare_prompt = f"""Ты — эксперт департамента снабжения в сфере Food. 
                Твоя задача — провести детальный сравнительный или одиночный анализ поставщиков под запрос '{query}' СТРОГО на основе предоставленных данных.
                
                ⚠️ КРИТИЧЕСКОЕ ПРАВИЛО: Тебе разрешено использовать информацию ТОЛЬКО из блока "Данные для анализа" ниже. Тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО выдумывать, упоминать или добавлять любые другие компании из твоей памяти, которых нет в списке ниже.
                
                Если в данных предоставлена только ОДНА компания, не пытайся её с кем-то сравнивать. Вместо этого в блоках [МЕЛКИЙ_ОПТ] и [КРУПНЫЙ_ОПТ] распиши, подходит ли эта конкретная компания под данные задачи, какие у неё условия для мелких и крупных заказов, и каковы её ограничения.
                
                Ты ДОЛЖЕН разделять свои мысли СТРОГО на 4 части и перед каждой частью написать соответствующий тег на отдельной строке:
                
                [МЕЛКИЙ_ОПТ]
                (Детальный разбор условий для небольших/стартовых заказов на основе данных ниже. Если компания одна — оцени её применимость для мелкого опта)
                
                [КРУПНЫЙ_ОПТ]
                (Детальный разбор условий для масштабирования и крупных объемов на основе данных ниже. Если компания одна — оцени её применимость для крупного опта)
                
                [ЧЕК_ЛИСТ]
                (Подробный список вопросов/проверок для закупщика перед подписанием договора именно с этой компанией/этими компаниями)
                
                [ВЕРДИКТ]
                (Итоговое резюме и рекомендация закупщику по работе с предоставленными контрагентами)
                
                Данные для анализа (работай ТОЛЬКО с ними):
                {all_cards_text}"""

                
                summary = llm_compare.invoke(compare_prompt)
                full_text = summary.content

            # Проверяем, использовала ли модель наши маркеры
            if "[МЕЛКИЙ_ОПТ]" in full_text and "[КРУПНЫЙ_ОПТ]" in full_text:
                import re

                def get_block(text, block_name):
                    pattern = rf"\[{block_name}\](.*?)(?=\[\w+\]|$)"
                    match = re.search(pattern, text, re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                        content = re.sub(r"^[\s\*•-]+", "", content)
                        content = re.sub(r"[\s\*•-]+$", "", content)
                        return content.strip()
                    return ""

                text_small = get_block(full_text, "МЕЛКИЙ_ОПТ")
                text_large = get_block(full_text, "КРУПНЫЙ_ОПТ")
                text_checklist = get_block(full_text, "ЧЕК_ЛИСТ")
                text_verdict = get_block(full_text, "ВЕРДИКТ")
                
                # ФУНКЦИЯ ДЛЯ КЛИНГИНГА: Убираем любые кривые звездочки модели в любом месте текста
                def clean_markdown_artifacts(text):
                    if not text:
                        return ""
                    # Удаляем двойные звездочки, если вокруг них нет букв (одиночные висячие артефакты)
                    text = re.sub(r'\*\*\s+\*\*', ' ', text)
                    text = re.sub(r'\*\*\*\*', '', text)
                    # Если модель поставила звездочки перед точкой или знаком препинания (типа ограничением**.)
                    text = re.sub(r'\*\*([.,!?\s]|$)', r'\1', text)
                    # И просто убираем любые оставшиеся битые двойные звездочки, которые портят вид
                    text = text.replace("**", "")
                    # УДАЛЯЕМ ДУБЛИ ЗАГОЛОВКОВ: убираем фразы типа "Итоговое резюме...", "Вердикт:" из начала текста
                    text = re.sub(r'^(Итоговое резюме|Вердикт|Итоговая рекомендация|Резюме)\s*(для\s+[^:]+)?:\s*', '', text, flags=re.IGNORECASE)
                    return text.strip()
    
                # Если модель разделила текст по тегам, выводим красивую структуру
                if text_small and text_large:
                    st.markdown("### 🎯 Стратегические сценарии закупки")
                    tab1, tab2 = st.tabs(["🚀 Тестирование рынка и мелкий опт", "📦 Масштабирование и крупный опт"])
                    
                    with tab1:
                        with st.container(border=True):
                            st.markdown("#### 💡 Стратегия гибкого старта")
                            st.markdown(clean_markdown_artifacts(text_small))
                        
                    with tab2:
                        with st.container(border=True):
                            st.markdown("#### 🏭 Стратегия оптимизации затрат")
                            st.markdown(clean_markdown_artifacts(text_large))
    
                    if text_checklist and text_checklist != "Аналитика формируется...":
                        with st.expander("🔍 Что проверить перед подписанием договора?", expanded=True):
                            st.markdown(clean_markdown_artifacts(text_checklist))
                    
                    if text_verdict and text_verdict != "Аналитика формируется...":
                        st.divider()
                        st.markdown(f"⚖️ **Итоговая рекомендация ИИ:** {clean_markdown_artifacts(text_verdict)}")
                
                else:
                    # ПЛАН Б: Если модель проигнорировала теги и выдала сплошной текст
                    # МЫ НЕ создаем пустых контейнеров внизу. Просто выводим чистый текст
                    st.markdown("### 🤖 Итоговое экспертное заключение ИИ")
                    st.markdown(clean_markdown_artifacts(full_text))

        else:
            st.info("Пока в базе демо-прототипа есть категории: 'упаковка' и 'молоко'. Попробуйте ввести их.")
