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
    validationPairsDirectory,
    individualFilepath,
    primitivesFilepath,
    imageShapeHW,
    outputDirectory
):
    logging.info("batch_analysis.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Create the interpreter
    primitive_functions_tree = ET.parse(primitivesFilepath)
    interpreter = image_processing.Interpreter(primitive_functions_tree, imageShapeHW)

    # Load the individual
    individual = gpcore.LoadIndividual(individualFilepath)

    # List the images in the directory
    filenames = os.listdir(validationPairsDirectory)
    filepaths = [os.path.join(validationPairsDirectory, f)
                 for f in filenames
                 if 'input' in f]
    #logging.debug("main(): filepaths = {}".format(filepaths))

    for filepath in filepaths:
        input_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        logging.debug("main(): Evaluating '{}'...".format(filepath))
        output_heatmap = interpreter.Evaluate(
            individual=individual,
            variableNameToTypeDict={'image': 'grayscale_image'},
            variableNameToValueDict={'image': input_img},
            expectedReturnType='binary_image'
        )
        output_heatmap_filepath = os.path.join(outputDirectory, os.path.basename(filepath)[:-4] + "_heatmap.png")
        cv2.imwrite(output_heatmap_filepath, output_heatmap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('validationPairsDirectory', help="The filepath to the validation image pairs")
    parser.add_argument('individualFilepath', help="The filepath to the individual to test")
    parser.add_argument('--primitivesFilepath',
                        help="The filepath to the primitives xml file. Default: 'vision_genprog/tasks/image_processing.xml'",
                        default='vision_genprog/tasks/image_processing.xml')
    parser.add_argument('--imageShapeHW', help="The image shape (height, width). Default='(256, 256)'",
                        default='(256, 256)')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './batch_analysis_outputs'", default='./batch_analysis_outputs')
    args = parser.parse_args()

    imageShapeHW = ast.literal_eval(args.imageShapeHW)

    main(
        args.validationPairsDirectory,
        args.individualFilepath,
        args.primitivesFilepath,
        imageShapeHW,
        args.outputDirectory
    )