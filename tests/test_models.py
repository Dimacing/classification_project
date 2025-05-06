import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path
import shutil
import joblib
from sklearn.pipeline import Pipeline

from src.models.simple_nn import SimpleNNModel
from src.models.random_forest_model import RandomForestModel
from src.models.logistic_regression_model import LogisticRegressionModel
from src.config.config import NUM_CLASSES, LABELS, RF_MAX_FEATURES
from src.data.dataset import TextDataset

@pytest.fixture(scope="module")
def dummy_texts():
    return ["это первый текст спорт", "второй текст про юмор", "третий текст политика и реклама", "еще один текст", "спорт и личная жизнь очень"]

@pytest.fixture(scope="module")
def dummy_labels():
    return np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1]
    ], dtype=np.float32)

@pytest.fixture(scope="module")
def adapted_text_vectorizer_tfidf(dummy_texts):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=RF_MAX_FEATURES,
        output_mode='tf-idf'
    )
    vectorizer.adapt(dummy_texts)
    config = vectorizer.get_config()
    weights = vectorizer.get_weights()
    return vectorizer, {'config': config, 'weights': weights}

@pytest.fixture
def simple_nn_instance(adapted_text_vectorizer_tfidf):
    vectorizer_instance, _ = adapted_text_vectorizer_tfidf
    return SimpleNNModel(num_classes=NUM_CLASSES, text_vectorizer=vectorizer_instance)

@pytest.fixture
def random_forest_instance():
    return RandomForestModel(num_classes=NUM_CLASSES, max_features=RF_MAX_FEATURES)

@pytest.fixture
def logistic_regression_instance():
    return LogisticRegressionModel(num_classes=NUM_CLASSES, max_features=RF_MAX_FEATURES)

@pytest.fixture
def dummy_dataset_simple(dummy_texts, dummy_labels):
    return TextDataset(dummy_texts, dummy_labels, batch_size=2)

@pytest.fixture(scope="function")
def temp_model_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("model_tests")

def test_simple_nn_build(simple_nn_instance):
    simple_nn_instance.build_model()
    model = simple_nn_instance.model
    assert model is not None
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, NUM_CLASSES)
    assert isinstance(model.layers[1], tf.keras.layers.TextVectorization)

def test_simple_nn_predict(simple_nn_instance, dummy_texts):
    simple_nn_instance.build_model()
    preds = simple_nn_instance.predict(np.array(dummy_texts[:2]))
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2, NUM_CLASSES)
    assert np.all((preds >= 0) & (preds <= 1))

def test_simple_nn_save_load(simple_nn_instance, temp_model_dir, dummy_texts, adapted_text_vectorizer_tfidf):
    _, vectorizer_state = adapted_text_vectorizer_tfidf
    model_save_dir = temp_model_dir / "simple_nn_test_model"
    vectorizer_save_path = temp_model_dir / "simple_nn_vectorizer.pkl"
    joblib.dump(vectorizer_state, vectorizer_save_path)
    simple_nn_instance.build_model()
    preds_before = simple_nn_instance.predict(np.array([dummy_texts[0]]))
    simple_nn_instance.save(model_save_dir)
    assert model_save_dir.is_dir()
    assert vectorizer_save_path.is_file()
    loaded_instance = SimpleNNModel()
    loaded_instance.load(model_save_dir)
    assert loaded_instance.model is not None
    assert loaded_instance.text_vectorizer is not None
    assert loaded_instance.num_classes == NUM_CLASSES
    assert loaded_instance.text_vectorizer.vocabulary_size() > 1
    preds_after = loaded_instance.predict(np.array([dummy_texts[0]]))
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5, atol=1e-5)

def test_random_forest_init(random_forest_instance):
    assert random_forest_instance.model is not None
    assert isinstance(random_forest_instance.model, Pipeline)
    assert 'tfidf' in random_forest_instance.model.named_steps
    assert 'clf' in random_forest_instance.model.named_steps
    assert random_forest_instance.num_classes == NUM_CLASSES

def test_random_forest_train(random_forest_instance, dummy_texts, dummy_labels):
    random_forest_instance.train(dummy_texts, dummy_labels)
    assert hasattr(random_forest_instance.model.named_steps['clf'], 'estimators_')

def test_random_forest_predict(random_forest_instance, dummy_texts, dummy_labels):
    random_forest_instance.train(dummy_texts, dummy_labels)
    preds = random_forest_instance.predict(dummy_texts[:2])
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2, NUM_CLASSES)
    assert np.all((preds == 0) | (preds == 1))

def test_random_forest_save_load(random_forest_instance, temp_model_dir, dummy_texts, dummy_labels):
    model_path = temp_model_dir / "rf_test_model.joblib"
    random_forest_instance.train(dummy_texts, dummy_labels)
    preds_before = random_forest_instance.predict([dummy_texts[0]])
    random_forest_instance.save(model_path)
    assert model_path.is_file()
    loaded_instance = RandomForestModel(num_classes=NUM_CLASSES)
    loaded_instance.load(model_path)
    assert loaded_instance.model is not None
    assert loaded_instance.num_classes == NUM_CLASSES
    preds_after = loaded_instance.predict([dummy_texts[0]])
    np.testing.assert_array_equal(preds_before, preds_after)

def test_logistic_regression_init(logistic_regression_instance):
    assert logistic_regression_instance.model is not None
    assert isinstance(logistic_regression_instance.model, Pipeline)
    assert 'tfidf' in logistic_regression_instance.model.named_steps
    assert 'clf' in logistic_regression_instance.model.named_steps
    assert logistic_regression_instance.num_classes == NUM_CLASSES

def test_logistic_regression_train(logistic_regression_instance, dummy_texts, dummy_labels):
    logistic_regression_instance.train(dummy_texts, dummy_labels)
    assert hasattr(logistic_regression_instance.model.named_steps['clf'], 'estimators_')

def test_logistic_regression_predict(logistic_regression_instance, dummy_texts, dummy_labels):
    logistic_regression_instance.train(dummy_texts, dummy_labels)
    preds = logistic_regression_instance.predict(dummy_texts[:2])
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2, NUM_CLASSES)
    assert np.all((preds == 0) | (preds == 1))

def test_logistic_regression_save_load(logistic_regression_instance, temp_model_dir, dummy_texts, dummy_labels):
    model_path = temp_model_dir / "logreg_test_model.joblib"
    logistic_regression_instance.train(dummy_texts, dummy_labels)
    preds_before = logistic_regression_instance.predict([dummy_texts[0]])
    logistic_regression_instance.save(model_path)
    assert model_path.is_file()
    loaded_instance = LogisticRegressionModel(num_classes=NUM_CLASSES)
    loaded_instance.load(model_path)
    assert loaded_instance.model is not None
    assert loaded_instance.num_classes == NUM_CLASSES
    preds_after = loaded_instance.predict([dummy_texts[0]])
    np.testing.assert_array_equal(preds_before, preds_after)
