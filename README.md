# Ai-face-Classification
Бинарная классификация изображений лиц: определение, является ли лицо реальным или сгенерированным нейросетью.
Проект реализован в формате Jupyter Notebook и ориентирован на задачу компьютерного зрения с обучением на изображениях лиц, использованием аугментаций, кросс-валидации и кастомной CNN-архитектуры.

В основе решения лежит кастомная сверточная нейросеть, которая использует:
- RGB-признаки исходного изображения;
- high-pass признаки, полученные с помощью **Laplacian filter**;
- residual-блоки с **Squeeze-and-Excitation**;
- обучение с **5-fold Stratified Cross-Validation**;
- **EMA (Exponential Moving Average)** весов;
- **AMP** для ускорения обучения на GPU;
- **TTA** на инференсе;
- подбор оптимального **threshold** по метрике **F1-score**.

## Стек 
- Python
- PyTorch
- OpenCV
- Albumentations
- NumPy / Pandas
- scikit-learn
- tqdm
- Jupyter Notebook

```bash
pip install albumentations opencv-python scikit-learn tqdm
```

## Результат
Решение дает F1-score равный 0.951 на локальной валидации и около 0.974 на public leaderboard.
