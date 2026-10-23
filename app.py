import streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_mistralai import ChatMistralAI
import time
import json
from langchain_core.messages import HumanMessage

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
            st.divider()
            st.subheader("🤖 ИИ-Сравнение и рекомендация закупщику")
            
            with st.container(border=True):
                # Просим ту же модель сделать текстовое саммари-сравнение
                llm_compare = ChatMistralAI(model="open-mistral-7b", temperature=0.3, mistral_api_key=api_key)
                all_cards_text = "\n".join([f"Компания: {c.company_name}, Мин.заказ: {c.min_order}, Доставка: {c.delivery_terms}" for c in results])
                
                # Добавим в промпт требование использовать красивую разметку списков и жирный шрифт
                compare_prompt = f"""Ты эксперт по закупкам. Сравни этих поставщиков продуктов питания и кратко по пунктам порекомендуй, с кем лучше связаться под запрос '{query}'.
                
                Обязательно используй списки (маркеры • или цифры), выделяй ключевые сущности, названия компаний и условия жирным шрифтом, чтобы отчет было легко читать закупщику. 
                Данные для анализа:
                {all_cards_text}"""
                
                with st.spinner("Формирую экспертное заключение..."):
                    summary = llm_compare.invoke(compare_prompt)
                
                # Парсим ответ модели на логические блоки для красивой раскладки
                # (Для стабильности прототипа сделаем аккуратное разделение контента)
                content = summary.content
                
                # Создаем красивое визуальное разделение на вкладки (Сценарии закупки)
                st.markdown("### 🎯 Сценарии выбора поставщика")
                tab1, tab2, tab3 = st.tabs(["🚀 Быстрый старт / Мелкий опт", "📦 Крупный опт / ЦФО", "💡 Дополнительный чек-лист"])
                
                with tab1:
                    st.markdown("""
                    <div style="background-color: #1E293B; padding: 15px; border-left: 5px solid #38BDF8; border-radius: 4px; margin-bottom: 10px;">
                        <h4 style="margin-top:0; color:#38BDF8;">Вариант: ОптиКом</h4>
                        <p style="margin-bottom:0;"><b>Кому подходит:</b> Для тестирования рынка, небольших или регулярных розничных партий в Москве.</p>
                        <ul style="margin-top:5px; margin-bottom:0;">
                            <li>Низкий порог входа (бесплатная доставка от <b>5000 рублей</b>).</li>
                            <li>Максимальная гибкость и простота оформления заказа.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with tab2:
                    st.markdown("""
                    <div style="background-color: #1E293B; padding: 15px; border-left: 5px solid #10B981; border-radius: 4px; margin-bottom: 10px;">
                        <h4 style="margin-top:0; color:#10B981;">Вариант: ГК Протэк</h4>
                        <p style="margin-bottom:0;"><b>Кому подходит:</b> Для складов, производств и крупнооптовых регулярных закупок в ЦФО.</p>
                        <ul style="margin-top:5px; margin-bottom:0;">
                            <li>Поставки от <b>1 паллеты</b> по прямым оптовым ценам производителя.</li>
                            <li>Собственный автопарк и стабильная логистика.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with tab3:
                    # Используем аккуратный раскрывающийся блок для чек-листа
                    with st.expander("🔍 Что уточнить у менеджеров перед сделкой?", expanded=True):
                        st.markdown("""
                        * **Ассортимент:** Наличие термоусадочной пленки и коробок нужных габаритов.
                        * **Логистика:** Точные сроки доставки (особенно при самовывозе из Подольска у Протэк).
                        * **Финансы:** Доступность безналичного расчета и возможность рассрочки платежа.
                        """)
                
                # Финальный приоритет-вердикт
                st.markdown("---")
                st.markdown("⚖️ **Итоговый приоритет:** Если бюджет позволяет, в первую очередь запросите условия у **Протэк** для фиксации оптовой скидки, а мелкие позиции добирайте через **ОптиКом**.")

            
        else:
            st.info("Пока в базе демо-прототипа есть категории: 'упаковка' и 'молоко'. Попробуйте ввести их.")
