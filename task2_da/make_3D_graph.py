import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import sys
import os


def info_before_print():
    """Печатает инструкцию по использованию для пользователя"""
    print(
        "Использование: python make_3D_graph.py n_samples centers n_features random_state cluster_std"
    )
    print(
        "Подробнее про признаки можно почитать в описании функции sklearn.datasets.make_blobs()"
    )
    print("n_samples int - количество сгенерированных точек данных")
    print("centers int - rоличество кластеров (групп) для создания")
    print(
        "n_features int - Количество признаков (измерений) у каждой точки (хотя бы 3, покажутся только первые 3)"
    )
    print('random_state - "Семя" для генератора случайных чисел')
    print("cluster_std float - Стандартное отклонение кластеров")
    print("Например python make_3D_graph.py 300 3 3 42 1.2")


def generate_data(In_samples, Icenters, In_features, Irandom_state, Icluster_std):
    """
    Генерирует синтетические данные с кластерами используя make_blobs.

    Parameters:
    In_samples (int): Количество точек данных
    Icenters (int): Количество кластеров
    In_features (int): Количество признаков (минимум 3)
    Irandom_state (int): Seed для воспроизводимости
    Icluster_std (float): Разброс кластеров

    Returns:
    tuple: (X, y) - матрица признаков и метки кластеров
    """
    X, y = make_blobs(
        n_samples=In_samples,
        centers=Icenters,
        n_features=In_features,
        random_state=Irandom_state,
        cluster_std=Icluster_std,
    )
    return X, y


def make_3D_graph(X, y):

    # Создание 3D графика
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Создание scatter plot с разными цветами для каждого кластера
    scatter = ax.scatter(
        X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis", s=50, alpha=0.8
    )

    # Подписи осей
    ax.set_xlabel("Признак 1", fontsize=12, labelpad=10)
    ax.set_ylabel("Признак 2", fontsize=12, labelpad=10)
    ax.set_zlabel("Признак 3", fontsize=12, labelpad=10)

    ax.set_title("3D визуализация признаков из make_blobs()", fontsize=14, pad=20)

    # Добавление цветовой легенды
    legend = ax.legend(*scatter.legend_elements(), title="Кластеры", loc="upper right")
    ax.add_artist(legend)

    # Настройка угла обзора для лучшего вида
    ax.view_init(elev=20, azim=45)  # elev - угол наклона, azim - поворот

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 6:
        info_before_print()
        sys.exit()

    try:
        # Преобразуем аргументы командной строки в нужные типы
        In_samples = int(sys.argv[1])
        Icenters = int(sys.argv[2])
        In_features = int(sys.argv[3])
        Irandom_state = int(sys.argv[4])
        Icluster_std = float(sys.argv[5])

    except ValueError as e:
        print(f"Ошибка в преобразовании типов: {e}")
        print("Проверьте, что передали правильные типы данных")
        info_before_print()
        sys.exit(1)

    # Дополнительные проверки
    if In_samples <= 0:
        print("Ошибка: n_samples должен быть положительным числом")
        sys.exit(1)

    if Icenters <= 0:
        print("Ошибка: centers должен быть положительным числом")
        sys.exit(1)

    if In_features < 3:
        print("Ошибка: n_features должен быть не менее 3 для 3D графика")
        sys.exit(1)

    if Icluster_std <= 0:
        print("Ошибка: cluster_std должен быть положительным числом")
        sys.exit(1)

    # Генерация и отображение данных
    try:
        X, y = generate_data(
            In_samples, Icenters, In_features, Irandom_state, Icluster_std
        )
        make_3D_graph(X, y)
    except Exception as e:
        print(f"Ошибка при создании графика: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
