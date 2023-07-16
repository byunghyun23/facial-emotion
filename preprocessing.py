import click
import json
import cv2
import numpy as np
import os
from pathlib import Path
import face_detection


@click.command()
@click.option('--root_dir', default='data', help='Root directory')
@click.option('--output_dir', default='processed', help='Processed directory')
def run(root_dir, output_dir):
    detector = face_detection.Mtcnn()

    dir = Path(root_dir)
    json_files = []
    extensions = {'.json'}

    for path in dir.glob(r'*'):
        if path.suffix in extensions:
            json_files.append(path)
    print(json_files)

    for json_file in json_files:
        with open(str(json_file), 'r', encoding='UTF-8') as f:
            json_data = json.load(f)

        max_len = len(json_data)
        input_dir = str(json_file)[-7:-5]

        for i in range(max_len):
            try:
                filename = json_data[i]['filename']
                # expr = json_data[i]['faceExp_uploader']

                input_path = root_dir + '/' + input_dir + '/' + filename
                # image = cv2.imread(input_path)
                image = np.fromfile(input_path, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                image = detector.get_face(image)

                output_path = root_dir + '/' + output_dir + '/' + filename

                # cv2.imwrite(output_path, image)
                extension = os.path.splitext(output_path)[1]
                result, encoded_img = cv2.imencode(extension, image)
                if result:
                    with open(output_path, mode='w+b') as f:
                        encoded_img.tofile(f)

                print(str(json_file), (str(i) + '/' + str(max_len)))

            except Exception as e:
                pass


if __name__ == '__main__':
    run()