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


def main():
    """Производит обработку входных данных и добавляет рамку изображению"""

    # проверяем что аргументы были переданы
    if len(sys.argv) < 3:
        info_before_print()
        sys.exit()

    filename = sys.argv[1]

    # Проверяем что файл существует
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден!")
        sys.exit()

    # Открываем
    img = cv2.imread(filename)
    if img is None:
        print("Ошибка загрузки изображения!")
        sys.exit()

    # проверяем кооректность ширины рамки. если меньше 0, выводим ошибку
    # также проверяем что это int
    try:
        # с помощью этого параметра пользователь может определять ширину рамки
        frame_width = int(sys.argv[2])
        if frame_width < 0:
            print("Рамка не может быть отрицательной ширины")
            sys.exit()
    except ValueError:
        print("Ширина рамки должна быть целым числом")
        sys.exit()

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
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_framed{ext}"

    # сохраняем изображение
    cv2.imwrite(output_filename, bordered_image)
    print(f"Изображение с рамкой сохранено как: {output_filename}")

    # показываем изображение
    cv2.imshow("Image with Border", bordered_image)

    # ждем нажатия какой-нибудь клавиши, чтобы закрылось окно с картинкой
    key = cv2.waitKey(0)

    # Закрываем все окна
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
