import cv2
import sys
import os


def info_before_print():
    """Печатает инструкцию по использованию для пользователя"""
    print(
        "Использование: python add_border_to_image.py image_file_name frame_width_in_pixels"
    )
    print("Где image_file_name - имя вашего файла в формате ./photo или путь до него")
    print("frame_width_in_pixels - введите желаемую ширину рамки")
    print("Например: python add_border_to_image.py ./exampple_2.png 20")
    print("Рамка будет синего цвета")


def add_frame_to_image(image_path, frame_width):
    """
    Добавляет рамку к изображению и возвращает результат

    Аргументы:
    image_path (str): Путь к файлу изображения
    frame_width (int): Ширина рамки в пикселях
    frame_color (tuple): Цвет рамки в формате BGR (по умолчанию белый)

    Возвращает:
        tuple: (framed_image, output_image_path) - изображение с рамкой и имя выходного файла

    Возможные ошибки:
        FileNotFoundError: Если файл не существует
        ValueError: Если изображение не может быть загружено или неверные параметры
    """
    # проверяем что файл существует
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не найден!")
    # открываем
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Ошибка загрузки изображения!")
    # добавляем рамку
    bordered_image = cv2.copyMakeBorder(
        img,
        frame_width,
        frame_width,
        frame_width,
        frame_width,
        cv2.BORDER_CONSTANT,
        None,
        value=255,
    )

    # создаем имя для сохранения файла
    name, ext = os.path.splitext(image_path)
    output_image_path = f"{name}_framed{ext}"

    return bordered_image, output_image_path


def main():
    """
    Основная функция для обработки изображения с рамкой

    Обрабатывает аргументы командной строки, добавляет рамку к изображению
    и сохраняет результат. Показывает полученное изображение.
    """

    # проверяем что аргументы были переданы
    if len(sys.argv) < 3:
        info_before_print()
        sys.exit()

    image_path = sys.argv[1]
    # проверяем корректность ширины рамки и правильный тип
    try:
        # с помощью этого параметра пользователь может определять ширину рамки
        frame_width = int(sys.argv[2])
        if frame_width < 0:
            print("Рамка не может быть отрицательной ширины")
            sys.exit()
    except ValueError:
        print("Ширина рамки должна быть целым числом")
        sys.exit()

    try:
        bordered_image, output_image_path = add_frame_to_image(image_path, frame_width)

        # сохраняем изображение
        cv2.imwrite(output_image_path, bordered_image)
        print(f"Изображение с рамкой сохранено как: {output_image_path}")

        # показываем изображение
        cv2.imshow("Image with Border", bordered_image)

        # ждем нажатия какой-нибудь клавиши, чтобы закрылось окно с картинкой
        key = cv2.waitKey(0)

        # закрываем все окна
        cv2.destroyAllWindows()

    # обработка ошибок
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
