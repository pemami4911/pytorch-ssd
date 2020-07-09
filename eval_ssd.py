import torch
from torch.utils.data import DataLoader

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.datasets.detrac_dataset import UADETRAC_Detection_Dataset
from vision.datasets.detrac_reid import UADETRAC_ReID_Dataset
from vision.utils import box_utils, measurements, descriptor_utils
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ms_ssd_lite
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TestTransform
from tqdm import tqdm

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument('--label_dir', type=str, help="The directory of the UA-DETRAC sequence lables")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument('--temperature', default=0.1, type=float)

# Matching Eval
parser.add_argument('--match_candidates', type=int, default=10, help="candidate matches to rank, higher is harder")

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)

def compute_nn_metric(features1, features2, bbox1, bbox2, temp=1):
    """
    For each candidate in features1, there is one match and candidates-1 false matches.
    
    Compute the fraction out of #candidates attempts that the correct match is the NN 
    match.

    Args:
        features1: (candidates, D)
        features2: (candidates, D)
        boxes1: (candidates,4)
        boxes2: (candidates,4)
    """
    candidates = features1.shape[0]
    desc1 = descriptor_utils.get_descriptors(features1, bbox1.squeeze(1).float())
    desc2 = descriptor_utils.get_descriptors(features2, bbox2.squeeze(1).float())

    desc1 = desc1.unsqueeze(1).repeat_interleave(candidates,1)
    desc2 = desc2.unsqueeze(0).repeat_interleave(candidates,0)

    cosine_sim = torch.nn.functional.cosine_similarity(desc1,desc2,dim=2)  # (candidates,candidates)
    cosine_sim /= temp
    # get max 1-->2
    idxs = torch.max(cosine_sim, dim=0, keepdim=False)[1]
    
    idxs = idxs.data.cpu().numpy()  # (candidates)
    targets = np.arange(candidates)  # (candidates)

    correct = (idxs == targets)
    score = float((correct).sum()) / float(candidates)
    return score


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    #class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, is_test=True)
        true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(args.dataset, dataset_type="test")
        true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    elif args.dataset_type == 'ua-detrac':
        dataset = UADETRAC_Detection_Dataset(args.dataset, args.label_dir)
        class_names = dataset.class_names
    elif args.dataset_type == 'ua-detrac-reid':
        config = mobilenetv1_ssd_config
        test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
        dataset = UADETRAC_ReID_Dataset(args.dataset, args.label_file, transform=test_transform)
        class_names = dataset.class_names
        matching_loader = DataLoader(dataset, args.match_candidates, num_workers=0, shuffle=False, drop_last=True)

    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True, device=DEVICE)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True, device=DEVICE)
    elif args.net == 'mb2-ms-ssd-lite':
        net = create_mobilenetv2_ms_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True, device=DEVICE)

    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)  

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ms-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, device=DEVICE, reid=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    if not args.dataset_type == 'ua-detrac-reid':
        with open(eval_path / "dets.txt", "w+") as f:
            results = []
            cur_seq = ""
            frame_count = 0
            w, h, a = [], [], []
            for i in range(len(dataset)):
                print("process image", i)
                timer.start("Load Image")
                if args.dataset_type == 'ua-detrac':
                    image, frame_id, sequence, sizes = dataset.get_image(i)
                    if cur_seq != sequence:
                        cur_seq = sequence
                        frame_count = 0
                    w += [sizes[:,0]]
                    h += [sizes[:,1]]
                    a += [sizes[:,0] / sizes[:,1]]
                else:
                    image = dataset.get_image(i)
                #print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
                timer.start("Predict")
                boxes, labels, probs = predictor.predict(image)
                #print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
                indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
                results.append(torch.cat([
                    indexes.reshape(-1, 1),
                    labels.reshape(-1, 1).float(),
                    probs.reshape(-1, 1),
                    boxes + 1.0  # matlab's indexes start from 1
                ], dim=1))
                boxes = boxes.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                probs = probs.data.cpu().numpy()
                for j in range(boxes.shape[0]):
                    
                    f.write("{},{},{},{},{},{},{},{},{}\n".format(cur_seq,frame_id,j,boxes[j,0],boxes[j,1],boxes[j,2],boxes[j,3],probs[j],labels[j]))
                frame_count += 1
            print('mean/max/min width: {}/{},{}, mean/max/min height: {},{},{}, mean/max/min aspect: {},{},{}'.format(
                np.mean(np.concatenate(w)), np.max(np.concatenate(w)), np.min(np.concatenate(w)),
                np.mean(np.concatenate(h)), np.max(np.concatenate(h)), np.min(np.concatenate(h)),
                np.mean(np.concatenate(a)), np.max(np.concatenate(a)), np.min(np.concatenate(a))
                ))
    #     results = torch.cat(results)
    #     for class_index, class_name in enumerate(class_names):
    #         if class_index == 0: continue  # ignore background
    #         prediction_path = eval_path / f"det_test_{class_name}.txt"
    #         with open(prediction_path, "w") as f:
    #             sub = results[results[:, 1] == class_index, :]
    #             for i in range(sub.size(0)):
    #                 prob_box = sub[i, 2:].numpy()
    #                 image_id = dataset.ids[int(sub[i, 0])]
    #                 print(
    #                     image_id + " " + " ".join([str(v) for v in prob_box]),
    #                     file=f
    #                 )
    #     aps = []
    #     print("\n\nAverage Precision Per-class:")
    #     for class_index, class_name in enumerate(class_names):
    #         if class_index == 0:
    #             continue
    #         prediction_path = eval_path / f"det_test_{class_name}.txt"
    #         ap = compute_average_precision_per_class(
    #             true_case_stat[class_index],
    #             all_gb_boxes[class_index],
    #             all_difficult_cases[class_index],
    #             prediction_path,
    #             args.iou_threshold,
    #             args.use_2007_metric
    #         )
    #         aps.append(ap)
    #         print(f"{class_name}: {ap}")

    #     print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    else:
        match_score = []
        for i, data in tqdm(enumerate(matching_loader)):
            images1, boxes1, images2, boxes2 = data
            images1 = images1.to(DEVICE)
            boxes1 = boxes1.to(DEVICE)
            images2 = images2.to(DEVICE)
            boxes2 = boxes2.to(DEVICE)

            _, _, features1 = net(images1)
            _, _, features2 = net(images2)

            score = compute_nn_metric(features1, features2, boxes1, boxes2, args.temperature)
            match_score += [score]

        avg_score = np.mean(match_score)

        print(f"\nAverage match score {avg_score}")
