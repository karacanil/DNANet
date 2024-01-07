import os
import numpy
from PIL import Image

def calculate_iou(result1, result2):
    intersection = numpy.logical_and(result1, result2)
    union = numpy.logical_or(result1, result2)
    iou_score = numpy.sum(intersection) / numpy.sum(union)
    return iou_score

def load_image_as_array(image_path):
    image = Image.open(image_path)
    return numpy.array(image)

def process_images(pred_dir, gt_dir):
    pred_images = [f for f in os.listdir(pred_dir) if f.endswith('_pred.png')]
    
    iou_scores = []
    max_iou = 0
    max_iou_images = ('', '')

    for pred_image in pred_images:
        gt_image_name = pred_image.replace('_pred', '')
        pred_image_path = os.path.join(pred_dir, pred_image)
        gt_image_path = os.path.join(gt_dir, gt_image_name)

        if os.path.exists(gt_image_path):
            pred_array = load_image_as_array(pred_image_path)
            gt_array = load_image_as_array(gt_image_path)
            iou_score = calculate_iou(pred_array, gt_array)
            print(f'IoU for {pred_image} and {gt_image_name} is {iou_score}')
            iou_scores.append(iou_score)

            if iou_score > max_iou:
                max_iou = iou_score
                max_iou_images = (pred_image, gt_image_name)
        else:
            print(f'No matching ground truth for {pred_image}')

    if iou_scores:
        average_iou = sum(iou_scores) / len(iou_scores)
        print(f'\nAverage IoU: {average_iou}')
        print(f'Maximum IoU: {max_iou}, for images {max_iou_images[0]} and {max_iou_images[1]}')
    else:
        print('No IoU scores were calculated.')

# Example usage
prediction_directory = 'IRSTD1k_Img'
groundtruth_directory = 'IRSTD1k_Label'
process_images(prediction_directory, groundtruth_directory)
