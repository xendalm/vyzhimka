import torch
from tqdm.auto import tqdm
from scipy.stats import pearsonr
from scipy.special import expit as sigmoid 
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import gc
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logits_and_labels(model, data_loader, zero_token_id, one_token_id, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating and collecting logits"):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            labels = batch.pop("labels")

            outputs = model.generate(
                **batch,
                max_new_tokens=1, 
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
            logits = outputs["scores"][0]

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            del batch, labels, outputs, logits
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    logits_for_target_tokens = all_logits[:, [zero_token_id, one_token_id]]

    return logits_for_target_tokens, all_labels

def find_best_threshold(logits_target_tokens, labels_binary):
    scores_diff = logits_target_tokens[:, 1] - logits_target_tokens[:, 0]
    probabilities_of_one = torch.sigmoid(torch.tensor(scores_diff)).numpy()
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    for threshold in tqdm(thresholds, desc="Finding best threshold"):
        predictions_binary = (probabilities_of_one > threshold).astype(int)
        f1 = f1_score(labels_binary, predictions_binary, average="weighted")
        f1_scores.append(f1)

    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    print(f"\nOptimal threshold found: {best_threshold:.4f} (max F1 = {f1_scores[best_threshold_idx]:.4f})")
    return best_threshold

def get_ece_mce(prob_class_1, labels_binary, n_bins = 10):
    bin_edges = np.percentile(prob_class_1, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0
    ece_mass = 0
    mce_mass = 0

    # Присваиваем сэмплы соответствующим бинам
    bin_indices = np.digitize(prob_class_1, bin_edges) - 1
    # np.digitize возвращает индекс бина, в который попадает значение.
    # -1 используется, т.к. bin_edges имеют n_bins+1 границ, а бины нумеруются от 0 до n_bins-1.

    for i in range(n_bins):
        # Выбираем сэмплы, попавшие в текущий бин
        samples_in_bin_mask = (bin_indices == i)
        samples_in_bin = prob_class_1[samples_in_bin_mask]
        labels_in_bin = labels_binary[samples_in_bin_mask]

        if len(samples_in_bin) > 0:
            # Средняя предсказанная уверенность в бине
            mean_predicted_value_bin = np.mean(samples_in_bin)
            # Фактическая доля положительных (точность) в бине
            fraction_of_positives_bin = np.mean(labels_in_bin)

            # Ошибка калибровки для данного бина
            bin_calibration_error = np.abs(fraction_of_positives_bin - mean_predicted_value_bin)

            # ECE: взвешиваем ошибку по количеству сэмплов в бине
            ece_mass += bin_calibration_error * len(samples_in_bin)

            mce_mass = max(mce_mass, bin_calibration_error)
        else:
            print(f"Bin {i} is empty.")


    # Нормализуем ECE по общему количеству сэмплов
    ece_mass /= len(labels_binary)

    return ece_mass, mce_mass

def calculate_final_metrics(logits_target_tokens, labels_binary, threshold):
    scores_diff = logits_target_tokens[:, 1] - logits_target_tokens[:, 0]
    pearson_corr, _ = pearsonr(scores_diff, labels_binary)
    auc = roc_auc_score(labels_binary, scores_diff)

    # probabilities = softmax(logits_target_tokens, axis=1) # Shape (num_samples, 2)
    prob_class_1 = sigmoid(scores_diff)
    predictions_binary = (prob_class_1 > threshold).astype(int)
    accuracy = accuracy_score(labels_binary, predictions_binary)
    f1 = f1_score(labels_binary, predictions_binary, average="weighted")

    confidences = np.maximum(prob_class_1, 1 - prob_class_1)#np.max(probabilities, axis=1)
    mean_confidence_overall = np.mean(confidences)
    correct_predictions_mask = (predictions_binary == labels_binary)
    incorrect_predictions_mask = ~correct_predictions_mask
    mean_confidence_correct = np.mean(confidences[correct_predictions_mask]) if np.any(correct_predictions_mask) else 0.0
    mean_confidence_incorrect = np.mean(confidences[incorrect_predictions_mask]) if np.any(incorrect_predictions_mask) else 0.0

    ece, mce = get_ece_mce(prob_class_1, labels_binary)

    return {
        "pearson_corr": pearson_corr,
        "roc_auc": auc,
        "accuracy": accuracy,
        "f1": f1,
        "mean_confidence_overall": mean_confidence_overall,
        "mean_confidence_correct": mean_confidence_correct,
        "mean_confidence_incorrect": mean_confidence_incorrect,
        "ece": ece,
        "mce": mce,
    }

# FOR CALIBRATION


class DifferenceAsProbabilityClassifier(BaseEstimator, ClassifierMixin): # Наследуем от BaseEstimator и ClassifierMixin
    def __init__(self, diff_scores=None): # Добавляем diff_scores как параметр __init__ для get_params
        self.diff_scores = diff_scores
        if diff_scores is not None:
            self.raw_probs = 1 / (1 + np.exp(-self.diff_scores))
        else:
            self.raw_probs = None # Или пустой массив, если удобнее

    def fit(self, X, y):
        # Этот метод нужен для CalibratedClassifierCV, но мы предполагаем,
        # что raw_probs уже установлены при инициализации.
        # Если бы это был настоящий классификатор, здесь было бы обучение.
        # Для CalibratedClassifierCV, который использует predict_proba,
        # X и y используются для обучения калибратора, а не базовой модели.
        # В нашем случае X_dummy и y_val будут переданы, но мы не будем
        # "обучать" self.raw_probs здесь.
        self.classes_ = np.unique(y) # CalibratedClassifierCV требует self.classes_
        return self

    def predict_proba(self, X):
        if self.raw_probs is None:
            raise ValueError("Raw probabilities (diff_scores) not initialized. Call __init__ with diff_scores.")
        # X здесь фактически не используется для вычисления,
        # но CalibratedClassifierCV будет вызывать predict_proba с подмножеством X_dummy
        # Мы должны вернуть предсказания для соответствующего подмножества
        # Это ключевой момент: мы должны брать raw_probs, соответствующие X
        # Поскольку X_dummy - это индексы, мы можем их использовать
        indices = X.flatten() # X_dummy - это np.array([[0],[1],...]), так что flatten()
        # Проверяем, чтобы индексы не выходили за пределы
        if np.max(indices) >= len(self.raw_probs) or np.min(indices) < 0:
            raise ValueError("Indices in X_dummy out of bounds for raw_probs.")
        selected_raw_probs = self.raw_probs[indices]
        return np.vstack([1 - selected_raw_probs, selected_raw_probs]).T

    # Добавляем методы get_params и set_params для совместимости с sklearn
    def get_params(self, deep=True):
        return {"diff_scores": self.diff_scores} # Возвращаем параметры, которые могут быть установлены

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # Переинициализируем raw_probs, если diff_scores были изменены
        if 'diff_scores' in parameters and parameters['diff_scores'] is not None:
            self.raw_probs = 1 / (1 + np.exp(-self.diff_scores))
        return self


def find_best_threshold_from_probabilities(probabilities, labels_binary):
     thresholds = np.linspace(0, 1, 101)
     f1_scores = []
     for threshold in tqdm(thresholds, desc="Finding best threshold"):
         predictions_binary = (probabilities > threshold).astype(int)
         f1 = f1_score(labels_binary, predictions_binary, average="weighted")
         f1_scores.append(f1)
     best_threshold_idx = np.argmax(f1_scores)
     best_threshold = thresholds[best_threshold_idx]
     print(f"\nOptimal threshold found: {best_threshold:.4f} (max F1 = {f1_scores[best_threshold_idx]:.4f})") # Отключим логи, чтобы не засорять вывод для каждой модели
     return best_threshold

def calculate_calibrated_metrics(logits_target_tokens, labels_binary, threshold, calibrated_probabilities):
    scores_diff = logits_target_tokens[:, 1] - logits_target_tokens[:, 0]
    pearson_corr, _ = pearsonr(scores_diff, labels_binary)
    auc = roc_auc_score(labels_binary, scores_diff)

    predictions_binary = (calibrated_probabilities > threshold).astype(int)
    accuracy = accuracy_score(labels_binary, predictions_binary)
    f1 = f1_score(labels_binary, predictions_binary, average="weighted")

    confidences = np.maximum(calibrated_probabilities, 1 - calibrated_probabilities)
    mean_confidence_overall = np.mean(confidences)
    correct_predictions_mask = (predictions_binary == labels_binary)
    incorrect_predictions_mask = ~correct_predictions_mask
    mean_confidence_correct = np.mean(confidences[correct_predictions_mask]) if np.any(correct_predictions_mask) else 0.0
    mean_confidence_incorrect = np.mean(confidences[incorrect_predictions_mask]) if np.any(incorrect_predictions_mask) else 0.0

    ece, mce = get_ece_mce(calibrated_probabilities, labels_binary)

    return {
        "pearson_corr": pearson_corr,
        "roc_auc": auc,
        "accuracy": accuracy,
        "f1": f1,
        "mean_confidence_overall": mean_confidence_overall,
        "mean_confidence_correct": mean_confidence_correct,
        "mean_confidence_incorrect": mean_confidence_incorrect,
        "ece": ece,
        "mce": mce,
    }