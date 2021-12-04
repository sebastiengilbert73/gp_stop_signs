import logging
import argparse
import os
import xml.etree.ElementTree as ET
import vision_genprog.tasks.image_processing as image_processing
import genprog.core as gpcore
import cv2
import ast

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def main(
        imageFilepath,
        individualFilepath,
        primitivesFilepath,
        imageShapeHW,
        outputDirectory
):
    logging.info("test_image.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Create the interpreter
    primitive_functions_tree = ET.parse(primitivesFilepath)
    interpreter = image_processing.Interpreter(primitive_functions_tree, imageShapeHW)

    # Load the individual
    individual = gpcore.LoadIndividual(individualFilepath)

    # Load the image
    original_img = cv2.imread(imageFilepath, cv2.IMREAD_GRAYSCALE)
    input_img = cv2.resize(original_img, dsize=(imageShapeHW[1], imageShapeHW[0]))
    input_img_filepath = os.path.join(outputDirectory, "testImage_main_input.png")
    cv2.imwrite(input_img_filepath, input_img)

    output_heatmap = interpreter.Evaluate(
        individual=individual,
        variableNameToTypeDict={'image': 'grayscale_image'},
        variableNameToValueDict={'image': input_img},
        expectedReturnType='binary_image'
    )
    output_heatmap_filepath = os.path.join(outputDirectory, "testImage_main_heatmap.png")
    cv2.imwrite(output_heatmap_filepath, output_heatmap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imageFilepath', help="The filepath to the image")
    parser.add_argument('individualFilepath', help="The filepath to the individual to test")
    parser.add_argument('--primitivesFilepath',
                        help="The filepath to the primitives xml file. Default: 'vision_genprog/tasks/image_processing.xml'",
                        default='vision_genprog/tasks/image_processing.xml')
    parser.add_argument('--imageShapeHW', help="The image shape (height, width). Default='(256, 256)'",
                        default='(256, 256)')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './test_image_outputs'",
                        default='./test_image_outputs')
    args = parser.parse_args()

    imageShapeHW = ast.literal_eval(args.imageShapeHW)

    main(
        args.imageFilepath,
        args.individualFilepath,
        args.primitivesFilepath,
        imageShapeHW,
        args.outputDirectory
    )