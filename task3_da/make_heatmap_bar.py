"""
Модуль для создания комплексных графиков: heatmap корреляции + bar plot дисперсии.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Tuple, Optional, List
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки и валидации данных."""

    @staticmethod
    def load_dataset(dataset_name: str = "california_housing"):
        """
        Загрузка датасета с обработкой ошибок.

        Parameters:
        -----------
        dataset_name : str
            Название датасета ('california_housing' или 'boston')

        Returns:
        --------
        tuple : (X, y, feature_names, dataset_description)
        """
        try:
            if dataset_name == "california_housing":
                dataset = fetch_california_housing()
            elif dataset_name == "boston":
                # Для обратной совместимости
                try:
                    from sklearn.datasets import load_boston

                    dataset = load_boston()
                except ImportError:
                    logger.warning(
                        "Boston dataset deprecated, using California Housing"
                    )
                    dataset = fetch_california_housing()
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            logger.info(f"Successfully loaded {dataset_name} dataset")
            return dataset.data, dataset.target, dataset.feature_names, dataset.DESCR

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise


class CorrelationAnalyzer:
    """Класс для анализа корреляции и дисперсии данных."""

    def __init__(self):
        self.corr_matrix = None
        self.variance_ratios = None
        self.feature_names = None

    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисление корреляционной матрицы.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с данными

        Returns:
        --------
        pd.DataFrame : корреляционная матрица
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        self.corr_matrix = df.corr()
        return self.corr_matrix

    def compute_variance_ratios(
        self, X: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        """
        Вычисление долей дисперсии для каждого признака.

        Parameters:
        -----------
        X : np.ndarray
            Массив признаков
        feature_names : List[str]
            Список названий признаков

        Returns:
        --------
        np.ndarray : доли дисперсии
        """
        if X.size == 0:
            raise ValueError("Input array is empty")

        self.feature_names = feature_names
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Вычисление дисперсии каждого признака
        variances = np.var(X_scaled, axis=0)
        total_variance = np.sum(variances)

        if total_variance == 0:
            raise ValueError("Total variance is zero")

        self.variance_ratios = variances / total_variance
        return self.variance_ratios


class ComplexPlotBuilder:
    """Класс для построения комплексных графиков."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (16, 8),
        heatmap_cmap: str = "coolwarm",
        bar_color: str = "skyblue",
    ):
        """
        Инициализация построителя графиков.

        Parameters:
        -----------
        figsize : Tuple[int, int]
            Размер фигуры
        heatmap_cmap : str
            Цветовая схема для heatmap
        bar_color : str
            Цвет для bar plot
        """
        self.figsize = figsize
        self.heatmap_cmap = heatmap_cmap
        self.bar_color = bar_color

    def create_complex_plot(
        self,
        corr_matrix: pd.DataFrame,
        variance_ratios: np.ndarray,
        feature_names: List[str],
        title: str = "Комплексный анализ данных",
    ) -> plt.Figure:
        """
        Создание комплексного графика: heatmap + bar plot.

        Parameters:
        -----------
        corr_matrix : pd.DataFrame
            Матрица корреляции
        variance_ratios : np.ndarray
            Доли дисперсии признаков
        feature_names : List[str]
            Список названий признаков
        title : str
            Заголовок графика

        Returns:
        --------
        plt.Figure : созданная фигура
        """
        # Валидация входных данных
        self._validate_input_data(corr_matrix, variance_ratios, feature_names)

        # Создание фигуры с заданным размером
        fig = plt.figure(figsize=self.figsize)

        try:
            # 1. Heatmap корреляции
            ax_heatmap = self._create_heatmap(fig, corr_matrix)

            # 2. Bar plot дисперсии
            ax_bar = self._create_bar_plot(fig, variance_ratios, feature_names)

            # Общие настройки
            plt.suptitle(title, fontsize=16, y=0.95)
            plt.tight_layout()

            return fig

        except Exception as e:
            plt.close(fig)
            logger.error(f"Error creating plot: {e}")
            raise

    def _validate_input_data(
        self,
        corr_matrix: pd.DataFrame,
        variance_ratios: np.ndarray,
        feature_names: List[str],
    ):
        """Валидация входных данных."""
        if corr_matrix is None or corr_matrix.empty:
            raise ValueError("Correlation matrix is empty")

        if variance_ratios is None or len(variance_ratios) == 0:
            raise ValueError("Variance ratios are empty")

        if feature_names is None or len(feature_names) == 0:
            raise ValueError("Feature names are empty")

        if len(variance_ratios) != len(feature_names):
            raise ValueError("Variance ratios and feature names have different lengths")

    def _create_heatmap(self, fig: plt.Figure, corr_matrix: pd.DataFrame) -> plt.Axes:
        """
        Создание heatmap корреляции.

        Parameters:
        -----------
        fig : plt.Figure
            Фигура для отрисовки
        corr_matrix : pd.DataFrame
            Матрица корреляции

        Returns:
        --------
        plt.Axes : ось с heatmap
        """
        ax_heatmap = plt.subplot2grid((1, 10), (0, 0), colspan=7, fig=fig)

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap=self.heatmap_cmap,
            center=0,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции"},
            ax=ax_heatmap,
            annot_kws={"size": 8},
        )

        ax_heatmap.set_title("Матрица корреляции признаков", fontsize=14, pad=20)
        ax_heatmap.tick_params(axis="both", which="major", labelsize=9)

        return ax_heatmap

    def _create_bar_plot(
        self, fig: plt.Figure, variance_ratios: np.ndarray, feature_names: List[str]
    ) -> plt.Axes:
        """
        Создание bar plot дисперсии.

        Parameters:
        -----------
        fig : plt.Figure
            Фигура для отрисовки
        variance_ratios : np.ndarray
            Доли дисперсии
        feature_names : List[str]
            Названия признаков

        Returns:
        --------
        plt.Axes : ось с bar plot
        """
        ax_bar = plt.subplot2grid((1, 10), (0, 7), colspan=3, fig=fig)

        # Сортировка по убыванию дисперсии для лучшей читаемости
        sorted_indices = np.argsort(variance_ratios)[::-1]
        sorted_variance = variance_ratios[sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]

        # Создание горизонтального bar plot
        bars = ax_bar.barh(
            range(len(sorted_features)),
            sorted_variance,
            color=self.bar_color,
            edgecolor="navy",
            alpha=0.7,
        )

        # Настройка осей и подписей
        ax_bar.set_yticks(range(len(sorted_features)))
        ax_bar.set_yticklabels(sorted_features, fontsize=10)
        ax_bar.set_xlabel("Доля дисперсии", fontsize=12)
        ax_bar.set_title("Относительная важность признаков", fontsize=14, pad=20)
        ax_bar.grid(axis="x", alpha=0.3, linestyle="--")

        # Добавление числовых значений
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax_bar.text(
                width + 0.005,  # Небольшой отступ от столбца
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        # Убираем верхнюю и правую оси для лучшего вида
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

        return ax_bar


def create_complex_plot_from_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str = "custom_dataset",
    **plot_kwargs,
) -> plt.Figure:
    """
    Основная функция для создания комплексного графика из данных.

    Parameters:
    -----------
    X : np.ndarray
        Массив признаков
    y : np.ndarray
        Целевая переменная
    feature_names : List[str]
        Список названий признаков
    dataset_name : str
        Название датасета для заголовка
    **plot_kwargs : dict
        Дополнительные параметры для построителя графиков

    Returns:
    --------
    plt.Figure : созданная фигура
    """
    try:
        # Создание DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df["TARGET"] = y

        # Анализ данных
        analyzer = CorrelationAnalyzer()
        corr_matrix = analyzer.compute_correlation_matrix(df)
        variance_ratios = analyzer.compute_variance_ratios(X, feature_names)

        # Построение графика
        plot_builder = ComplexPlotBuilder(**plot_kwargs)
        fig = plot_builder.create_complex_plot(
            corr_matrix,
            variance_ratios,
            feature_names,
            title=f"Комплексный анализ: {dataset_name}",
        )

        logger.info("Complex plot created successfully")
        return fig

    except Exception as e:
        logger.error(f"Error in create_complex_plot_from_data: {e}")
        raise


def main():
    """Основная функция для демонстрации работы."""
    try:
        # Загрузка данных
        X, y, feature_names, description = DataLoader.load_dataset("california_housing")

        # Создание комплексного графика
        fig = create_complex_plot_from_data(
            X,
            y,
            feature_names,
            dataset_name="California Housing",
            figsize=(18, 9),
            heatmap_cmap="RdBu_r",
            bar_color="lightcoral",
        )

        # Сохранение графика
        plt.savefig("complex_plot.png", dpi=300, bbox_inches="tight")
        logger.info("Plot saved as 'complex_plot.png'")

        # Показ графика
        plt.show()

        # Дополнительная информация
        print(f"\nДополнительная информация:")
        print(f"Количество признаков: {len(feature_names)}")
        print(f"Количество samples: {X.shape[0]}")
        print(
            f"Сумма долей дисперсии: {np.sum(np.var(StandardScaler().fit_transform(X), axis=0) / np.sum(np.var(StandardScaler().fit_transform(X), axis=0))):.3f}"
        )

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
