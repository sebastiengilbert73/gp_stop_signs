import ast
import logging
import argparse
import xml.etree.ElementTree as ET
import cv2
import vision_genprog.tasks.image_processing as image_processing
import vision_genprog.semanticSegmentersPop as semanticSegmentersPop
import os
import ast
import synthetic_heatmap.generators.stop_sign as stop_sign

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def main(
    validationPairsDirectory,
    primitivesFilepath,
    imageShapeHW,
    outputDirectory,
    numberOfIndividuals,
    levelToFunctionProbabilityDict,
    proportionOfConstants,
    constantCreationParametersList,
    numberOfGenerations,
    weightForNumberOfNodes,
    numberOfTournamentParticipants,
    mutationProbability,
    proportionOfNewIndividuals,
    maximumNumberOfMissedCreationTrials,
    numberOfTrainingPairs
    ):
    logging.info("evolve_population.main()")

    # Create the output directory
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Create the interpreter
    primitive_functions_tree = ET.parse(primitivesFilepath)
    interpreter = image_processing.Interpreter(primitive_functions_tree, imageShapeHW)

    variableName_to_type = {'image': 'grayscale_image'}
    return_type = 'binary_image'  # We want a semantic segmentation separating background from objects of interest

    # Load the validation pairs
    validation_input_output_tuples = InputOutputTuples(validationPairsDirectory, imageShapeHW, 'image')

    # Create a population
    semantic_segmenters_pop = semanticSegmentersPop.SemanticSegmentersPopulation()
    semantic_segmenters_pop.Generate(
        numberOfIndividuals=numberOfIndividuals,
        interpreter=interpreter,
        returnType=return_type,
        levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
        proportionOfConstants=proportionOfConstants,
        constantCreationParametersList=constantCreationParametersList,
        variableNameToTypeDict=variableName_to_type,
        functionNameToWeightDict=None
    )

    # Original population cost
    validation_individual_to_cost_dict = semantic_segmenters_pop.EvaluateIndividualCosts(
        inputOutputTuplesList=validation_input_output_tuples,
        variableNameToTypeDict=variableName_to_type,
        interpreter=interpreter,
        returnType=return_type,
        weightForNumberOfElements=weightForNumberOfNodes
    )
    (champion, validation_lowest_cost) = semantic_segmenters_pop.Champion(validation_individual_to_cost_dict)
    validation_median_cost = semantic_segmenters_pop.MedianCost(validation_individual_to_cost_dict)
    validation_average_cost = semantic_segmenters_pop.AverageCost(validation_individual_to_cost_dict)
    validation_cost_std_dev = semantic_segmenters_pop.StandardDeviationOfCost(validation_individual_to_cost_dict)
    train_individual_to_cost_dict = None
    with open(os.path.join(args.outputDirectory, "generations.csv"), 'w+') as generations_file:
        generations_file.write("generation,train_lowest_cost,validation_lowest_cost,train_median_cost,validation_median_cost,train_average_cost,validation_average_cost,train_cost_std_dev,validation_cost_std_dev\n")
        generations_file.write("0,-,{},-,{},-,{},-,{}\n".format(validation_lowest_cost, validation_median_cost, validation_average_cost, validation_cost_std_dev))

    # Create a generator of stop sign image and heatmaps
    stop_sign_generator = stop_sign.StopSign()

    lowest_validation_champion_cost = 1.0e9

    for generation_ndx in range(1, numberOfGenerations + 1):
        logging.info("***** Generation {} *****".format(generation_ndx))

        # Generate synthetic training input-output tuples
        train_input_output_tuples_list = GenerateSyntheticInputOutputTuples(
            stop_sign_generator, imageShapeHW, numberOfTrainingPairs
        )
        # Evolve one generation
        train_individual_to_cost_dict = semantic_segmenters_pop.NewGenerationWithTournament(
            inputOutputTuplesList=train_input_output_tuples_list,
            variableNameToTypeDict=variableName_to_type,
            interpreter=interpreter,
            returnType=return_type,
            numberOfTournamentParticipants=numberOfTournamentParticipants,
            mutationProbability=mutationProbability,
            currentIndividualToCostDict=train_individual_to_cost_dict,
            proportionOfConstants=proportionOfConstants,
            levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
            functionNameToWeightDict=None,
            constantCreationParametersList=constantCreationParametersList,
            proportionOfNewIndividuals=proportionOfNewIndividuals,
            weightForNumberOfElements=weightForNumberOfNodes,
            maximumNumberOfMissedCreationTrials=maximumNumberOfMissedCreationTrials
        )
        train_champion, train_lowest_cost = semantic_segmenters_pop.Champion(train_individual_to_cost_dict)
        train_median_cost = semantic_segmenters_pop.MedianCost(train_individual_to_cost_dict)
        train_average_cost = semantic_segmenters_pop.AverageCost(train_individual_to_cost_dict)
        train_cost_std_dev = semantic_segmenters_pop.StandardDeviationOfCost(train_individual_to_cost_dict)

        # Validation of the new population
        validation_individual_to_cost_dict = semantic_segmenters_pop.EvaluateIndividualCosts(
            inputOutputTuplesList=validation_input_output_tuples,
            variableNameToTypeDict=variableName_to_type,
            interpreter=interpreter,
            returnType=return_type,
            weightForNumberOfElements=weightForNumberOfNodes
        )
        (champion, validation_lowest_cost) = semantic_segmenters_pop.Champion(validation_individual_to_cost_dict)
        validation_median_cost = semantic_segmenters_pop.MedianCost(validation_individual_to_cost_dict)
        validation_average_cost = semantic_segmenters_pop.AverageCost(validation_individual_to_cost_dict)
        validation_cost_std_dev = semantic_segmenters_pop.StandardDeviationOfCost(validation_individual_to_cost_dict)

        logging.info("train_lowest_cost = {}; validation_lowest_cost = {}".format(train_lowest_cost, validation_lowest_cost))
        with open(os.path.join(args.outputDirectory, "generations.csv"), 'a') as generations_file:
            generations_file.write("{},{},{},{},{},{},{},{},{}\n".format(generation_ndx,
                                                                        train_lowest_cost, validation_lowest_cost,
                                                                        train_median_cost, validation_median_cost,
                                                                        train_average_cost, validation_average_cost,
                                                                        train_cost_std_dev, validation_cost_std_dev))
        if validation_lowest_cost < lowest_validation_champion_cost:
            lowest_validation_champion_cost = validation_lowest_cost
            champion_filepath = os.path.join(outputDirectory, "champion_{}_{:.4f}.xml".format(generation_ndx, validation_lowest_cost))
            champion.Save(champion_filepath)

def InputOutputTuples(image_pairs_directory, expected_image_shapeHW, variable_name='image'):
    # List[Tuple[Dict[str, Any], Any]]
    input_heatmap_filepaths = InputHeatmapFilepaths(image_pairs_directory)
    inputOutput_list = []
    for input_filepath, heatmap_filepath in input_heatmap_filepaths:
        image = cv2.imread(input_filepath, cv2.IMREAD_GRAYSCALE)
        if image.shape != expected_image_shapeHW:
            raise ValueError("InputOutputTuples(): The shape of image '{}' ({}) is not the expected shape {}".format(
                input_filepath, image.shape, expected_image_shapeHW))
        heatmap = cv2.imread(heatmap_filepath, cv2.IMREAD_GRAYSCALE)
        inputOutput_list.append(({variable_name: image}, heatmap))
    return inputOutput_list

def InputHeatmapFilepaths(images_directory):
    input_heatmap_filepaths = []
    input_filepaths_in_directory = [os.path.join(images_directory, filename) for filename in os.listdir(images_directory)
                              if os.path.isfile(os.path.join(images_directory, filename))
                              and '_input' in filename
                              and filename.upper().endswith('.PNG')]
    for input_filepath in input_filepaths_in_directory:
        corresponding_heatmap_filepath = os.path.join(images_directory, os.path.basename(input_filepath).replace('_input', '_heatmap'))
        if not os.path.exists(corresponding_heatmap_filepath):
            raise FileNotFoundError("InputHeatmapFilepaths(): Could not find the heatmap file '{}'".format(corresponding_heatmap_filepath))
        input_heatmap_filepaths.append((input_filepath, corresponding_heatmap_filepath))
    return input_heatmap_filepaths

def GenerateSyntheticInputOutputTuples(generator, image_sizeHW, number_of_pairs_to_generate,
                                       variable_name='image'):
    input_output_tuples_list = []
    for image_ndx in range(number_of_pairs_to_generate):
        (input_image, heatmap, result_msg) = generator.Generate(image_sizeHW)
        input_output_tuples_list.append(({variable_name: input_image}, heatmap))
    return input_output_tuples_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('validationPairsDirectory', help="The filepath to the validation image pairs")
    parser.add_argument('--primitivesFilepath', help="The filepath to the primitives xml file. Default: 'vision_genprog/tasks/image_processing.xml'", default='vision_genprog/tasks/image_processing.xml')
    parser.add_argument('--imageShapeHW', help="The image shape (height, width). Default='(256, 256)'", default='(256, 256)')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs'", default='./outputs')
    parser.add_argument('--numberOfIndividuals', help="The number of individuals. Default: 200", type=int, default=200)
    parser.add_argument('--levelToFunctionProbabilityDict',
                        help="The probability to generate a function, at each level. Default: '{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}'",
                        default='{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}')
    parser.add_argument('--proportionOfConstants',
                        help='The probability to generate a constant, when a variable could be used. Default: 0',
                        type=float, default=0)
    parser.add_argument('--constantCreationParametersList',
                        help="The parameters to use when creating constants: [minFloat, maxFloat, minInt, maxInt, width, height]. Default: '[-1, 1, 0, 255, 256, 256]'",
                        default='[-1, 1, 0, 255, 256, 256]')
    parser.add_argument('--numberOfGenerations', help="The number of generations to run. Default: 32", type=int,
                        default=32)
    parser.add_argument('--weightForNumberOfNodes',
                        help="Penalty term proportional to the number of nodes. Default: 0.001", type=float,
                        default=0.001)
    parser.add_argument('--numberOfTournamentParticipants',
                        help="The number of participants in selection tournaments. Default: 2", type=int, default=2)
    parser.add_argument('--mutationProbability', help="The probability to mutate a child. Default: 0.1", type=float,
                        default=0.1)
    parser.add_argument('--proportionOfNewIndividuals',
                        help="The proportion of randomly generates individuals per generation. Default: 0.1",
                        type=float, default=0.1)
    parser.add_argument('--maximumNumberOfMissedCreationTrials',
                        help="The maximum number if missed creation trials. Default: 1000", type=int, default=1000)
    parser.add_argument('--numberOfTrainingPairs', help="The number of generated training pairs, per epoch. Default: 160", type=int, default=160)
    args = parser.parse_args()

    imageShapeHW = ast.literal_eval(args.imageShapeHW)
    levelToFunctionProbabilityDict = ast.literal_eval(args.levelToFunctionProbabilityDict)
    constantCreationParametersList = ast.literal_eval(args.constantCreationParametersList)
    main(
        args.validationPairsDirectory,
        args.primitivesFilepath,
        imageShapeHW,
        args.outputDirectory,
        args.numberOfIndividuals,
        levelToFunctionProbabilityDict,
        args.proportionOfConstants,
        constantCreationParametersList,
        args.numberOfGenerations,
        args.weightForNumberOfNodes,
        args.numberOfTournamentParticipants,
        args.mutationProbability,
        args.proportionOfNewIndividuals,
        args.maximumNumberOfMissedCreationTrials,
        args.numberOfTrainingPairs
    )