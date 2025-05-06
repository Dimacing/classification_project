import streamlit as st
import requests
from typing import Dict, List
import time
import pandas as pd
import json


BACKEND_URL = "http://127.0.0.1:8003"
MODELS = ["simple_nn", "random_forest", "logistic_regression", "transformer"]
LABELS = ["спорт", "юмор", "реклама", "соцсети", "политика", "личная жизнь"]


def get_available_models() -> List[str]:
    return MODELS

def predict_text(text: str) -> Dict:
    target_url = f"{BACKEND_URL}/predict_text"
    st.info(f"Sending request to: {target_url}")
    try:
        response = requests.post(target_url, json={"text": text}, timeout=30)
        st.info(f"API Response Status: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Ошибка подключения к серверу API ({target_url}). Запущен ли он?")
        return None
    except requests.exceptions.Timeout:
        st.error("Превышено время ожидания ответа от сервера.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка запроса к API: {e}")
        if e.response is not None:
             st.error(f"API Response Content: {e.response.text}")
             try:
                 error_detail = e.response.json().get("detail", "Нет деталей")
                 st.error(f"Детали от сервера: {error_detail}")
             except json.JSONDecodeError:
                 pass
        return None

def predict_file(file) -> Dict:
    target_url = f"{BACKEND_URL}/predict_file"
    st.info(f"Sending request to: {target_url}")
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(target_url, files=files, timeout=60)
        st.info(f"API Response Status: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Ошибка подключения к серверу API ({target_url}). Запущен ли он?")
        return None
    except requests.exceptions.Timeout:
        st.error("Превышено время ожидания ответа от сервера при загрузке файла.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка запроса к API при загрузке файла: {e}")
        if e.response is not None:
             st.error(f"API Response Content: {e.response.text}")
             try:
                 error_detail = e.response.json().get("detail", "Нет деталей")
                 st.error(f"Детали от сервера: {error_detail}")
             except json.JSONDecodeError:
                  pass
        return None

def display_results(results: Dict):
    st.subheader("Результаты анализа:")
    available_models = get_available_models()
    if not results: st.warning("API не вернул результаты."); return

    for model_name in available_models:
        if model_name not in results:
             st.warning(f"Результаты для модели '{model_name}' отсутствуют в ответе API.")
             continue

        st.markdown(f"--- \n### Модель: `{model_name}`")
        model_results = results.get(model_name, {})


        if isinstance(model_results, dict) and "error" in model_results:
            st.error(f"Ошибка при предсказании: {model_results['error']}")
            continue
        elif not isinstance(model_results, dict):
             st.error(f"Получен некорректный формат результатов для модели {model_name}: {model_results}")
             continue

        try:
            valid_items = {k: v for k, v in model_results.items() if isinstance(k, str) and isinstance(v, (int, float))}
            if len(valid_items) != len(model_results):
                 st.warning(f"Некоторые результаты для {model_name} имели некорректный формат и были пропущены.")
            if not valid_items:
                 st.info("Нет валидных числовых результатов для отображения.")
                 continue
            sorted_preds = sorted(valid_items.items(), key=lambda item: item[1], reverse=True)
        except (AttributeError, TypeError) as sort_e:
             st.error(f"Не удалось обработать/отсортировать результаты для {model_name}. Ошибка: {sort_e}. Данные: {model_results}")
             continue

        cols = st.columns(2)
        col_idx = 0
        for label, score in sorted_preds:
             with cols[col_idx % 2]:
                 try:
                     score = float(score)
                     if 0.0 <= score <= 1.0:
                          st.progress(value=score, text=f"{label.capitalize()}: {score:.2%}")
                     else:
                          pred_text = 'Да' if score >= 0.5 else 'Нет'
                          st.markdown(f"**{label.capitalize()}**: {pred_text} ({score:.3f})")
                 except (ValueError, TypeError) as format_e:
                      st.markdown(f"**{label.capitalize()}**: {score} (Ошибка форматирования: {format_e})")
             col_idx += 1

def rating_section():
    st.subheader("Оцените качество работы моделей:")
    available_models = get_available_models()
    if not available_models: st.warning("Нет доступных моделей для оценки."); return
    if 'results' not in st.session_state or st.session_state['results'] is None: return

    with st.form("rating_form"):
        ratings_input = {}
        num_cols = min(len(available_models), 3)
        cols = st.columns(num_cols)
        for i, model_name in enumerate(available_models):
            with cols[i % num_cols]:
                ratings_input[model_name] = st.slider(
                    label=f"Оценка `{model_name}`", min_value=1, max_value=5, value=3, key=f"rating_{model_name}"
                )
        submitted = st.form_submit_button("Отправить оценки")
        if submitted:
            submit_ratings(ratings_input)

def submit_ratings(ratings: Dict):
    success_count = 0; error_count = 0
    target_url_base = f"{BACKEND_URL}/rate_model"
    for model_name, rating in ratings.items():
        try:
            response = requests.post(target_url_base, json={"model_name": model_name, "rating": rating}, timeout=10)
            response.raise_for_status()
            success_count += 1
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка отправки оценки для `{model_name}`: {e}")
            error_count += 1

    if success_count > 0 and error_count == 0:
        st.success("Спасибо за ваши оценки!")
        time.sleep(1)
        st.experimental_rerun()
    elif success_count > 0 and error_count > 0:
         st.warning("Часть оценок отправлена, но возникли ошибки.")
         time.sleep(1)
         st.experimental_rerun()
    elif error_count > 0:
         st.error("Не удалось отправить оценки.")

def show_model_ratings():
    st.sidebar.subheader("Пользовательские рейтинги")
    target_url = f"{BACKEND_URL}/model_ratings"
    try:
        response = requests.get(target_url, timeout=10)
        response.raise_for_status()
        ratings = response.json()
        if not ratings: st.sidebar.info("Оценок пока нет."); return

        available_models = get_available_models()
        displayed_count = 0
        for model_name in available_models:
            score = ratings.get(model_name)
            if score is not None and score > 0:
                st.sidebar.markdown(f"- **`{model_name}`**: {score:.2f} ★")
                displayed_count +=1

        if displayed_count == 0:
             st.sidebar.info("Оценок пока нет.")

    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Не удалось загрузить рейтинги ({target_url}): {e}")


def show_evaluation_metrics():
    st.title("📈 Сравнение производительности моделей")
    st.markdown("Метрики рассчитаны на **валидационном** наборе данных скриптом `evaluate.py`.")
    st.info("Для обновления запустите `python evaluate.py` после (пере)обучения моделей.")
    st.markdown("---")

    metrics_url = f"{BACKEND_URL}/evaluation_metrics"
    try:
        response = requests.get(metrics_url, timeout=15)
        response.raise_for_status()
        all_metrics_data = response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Не удалось подключиться к API ({metrics_url}) для получения метрик.")
        return
    except requests.exceptions.HTTPError as e:
         if e.response.status_code == 404: st.warning("Файл с метриками оценки (`evaluation_metrics.json`) не найден на сервере. Запустите `python evaluate.py`.")
         else: st.error(f"Ошибка при запросе метрик (HTTP {e.response.status_code}): {e}")
         return
    except requests.exceptions.RequestException as e: st.error(f"Ошибка при запросе метрик: {e}"); return
    except json.JSONDecodeError: st.error("Ошибка декодирования ответа с метриками от API (не JSON)."); return

    available_models = get_available_models()

    for model_name in available_models:
        st.markdown(f"--- \n#### Метрики для модели: `{model_name}`")
        model_metrics = all_metrics_data.get(model_name)

        if not model_metrics: st.warning("Нет данных по метрикам для этой модели."); continue
        if "error" in model_metrics: st.error(f"Ошибка при расчете метрик на сервере: {model_metrics['error']}"); continue
        overall_cols = st.columns(4)
        with overall_cols[0]: st.metric("Точность (Subset Acc)", f"{model_metrics.get('subset_accuracy', 0):.3f}")
        with overall_cols[1]: st.metric("Hamming Loss", f"{model_metrics.get('hamming_loss', 0):.3f}")
        with overall_cols[2]: st.metric("F1 Micro", f"{model_metrics.get('f1_micro', 0):.3f}")
        with overall_cols[3]: st.metric("F1 Macro", f"{model_metrics.get('f1_macro', 0):.3f}")
        if "classification_report" in model_metrics:
            with st.expander("Детальный отчет по классам (Classification Report)"):
                report_data = model_metrics["classification_report"]
                labels_in_report = [l for l in report_data.keys() if l in LABELS]
                if labels_in_report:
                    report_df_data = {label: report_data[label] for label in labels_in_report}
                    report_df = pd.DataFrame.from_dict(report_df_data, orient='index').round(3)
                    report_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
                    st.dataframe(report_df)
                else:
                    st.warning("Нет данных по отдельным классам в classification report.")
        else:
            st.info("Classification report недоступен для этой модели.")


def main():
    st.set_page_config(page_title="Классификатор Текста", layout="wide")

    with st.sidebar:
        st.title("🎛️ Навигация")
        page = st.radio(
            "Выберите раздел:",
            ("Классификация", "Оценка моделей"),
            key="page_selector",
            label_visibility="collapsed"
        )
        st.markdown("---")
        show_model_ratings()

    if page == "Классификация":
        st.title("🤖 Классификация текста по темам")
        st.markdown(f"Определение тематики текста по категориям: **{', '.join(LABELS)}**.")
        st.markdown("---")

        for key in ['text_input', 'results', 'analyzed_text', 'analyzed_text_source']:
            if key not in st.session_state:
                st.session_state[key] = None if key != 'text_input' else ""

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ввод текста")
            st.text_area(
                "Введите текст для анализа:",
                value=st.session_state['text_input'],
                height=150,
                key="text_area_widget",
                on_change=lambda: st.session_state.update(text_input=st.session_state.text_area_widget)
            )
            uploaded_file = st.file_uploader(
                "Или загрузите текстовый файл (.txt):", type=["txt"], key="file_uploader"
            )

        with col2:
            st.subheader("Анализ")
            analyze_button = st.button("Анализировать", type="primary", use_container_width=True)
            if analyze_button:
                if uploaded_file is not None:
                    try:
                        final_text = uploaded_file.read().decode("utf-8")
                        input_method = "file"
                        st.info(f"Анализ из файла: {uploaded_file.name}")
                        st.session_state['text_input'] = ""
                        st.session_state['analyzed_text_source'] = uploaded_file.name
                    except Exception as e:
                        st.error(f"Ошибка чтения файла: {e}")
                        st.session_state['results'] = None; st.session_state['analyzed_text'] = None; return
                elif st.session_state['text_input']:
                    final_text = st.session_state['text_input'].strip()
                    if final_text:
                        input_method = "text"
                        st.info("Анализ введенного текста.")
                        st.session_state['analyzed_text_source'] = "текстовое поле"
                    else:
                        st.warning("Текстовое поле пусто.")
                        st.session_state['results'] = None; st.session_state['analyzed_text'] = None; return
                else:
                    st.warning("Пожалуйста, введите текст или загрузите файл.")
                    st.session_state['results'] = None; st.session_state['analyzed_text'] = None; return


                if final_text and input_method != "none":
                    with st.spinner("🧠 Анализируем текст..."):
                        api_response = None
                        if input_method == "file":
                            uploaded_file.seek(0)
                            api_response = predict_file(uploaded_file)
                        elif input_method == "text":
                            api_response = predict_text(final_text)

                        st.session_state['results'] = api_response if api_response is not None else None
                        st.session_state['analyzed_text'] = final_text if api_response is not None else None

        if st.session_state.get('results') is not None:
            st.markdown("---")
            if st.session_state.get('analyzed_text') is not None:
                 source = st.session_state.get('analyzed_text_source', 'источника')
                 with st.expander(f"Проанализированный текст (из {source}):", expanded=False):
                      st.text(st.session_state['analyzed_text'][:2000]) # Show preview
                      if len(st.session_state['analyzed_text']) > 2000: st.caption("...")
            display_results(st.session_state['results'])
            st.markdown("---")
            rating_section()

    elif page == "Оценка моделей":
        show_evaluation_metrics()

if __name__ == "__main__":
    main()