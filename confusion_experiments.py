# confusion_experiments.py
# model_confusion_experiment() runs the experiments in section 5.2
# it measures confusion for the sdn for correct and wrong predictions at the final layer
# for cnn it measure the softmax confidence scores
# it generates histograms and outputs the comparison

import torch
import numpy as np

import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs


def get_sdn_stats(layer_correct, layer_wrong, instance_confusion):
    layer_keys = sorted(list(layer_correct.keys()))

    correct_confusion = []
    wrong_confusion = []

    for inst in layer_correct[layer_keys[-1]]:
        correct_confusion.append(instance_confusion[inst])
        
    for inst in layer_wrong[layer_keys[-1]]:
        wrong_confusion.append(instance_confusion[inst])

    mean_correct_confusion = np.mean(correct_confusion)
    mean_wrong_confusion = np.mean(wrong_confusion)

    print('Confusion of corrects: {}, Confusion of wrongs: {}'.format(mean_correct_confusion, mean_wrong_confusion))

    return correct_confusion, wrong_confusion

def get_cnn_stats(correct, wrong, instance_confidence):
    print('get cnn stats')

    correct_confidence = []
    wrong_confidence = []

    for inst in correct:
        correct_confidence.append(instance_confidence[inst])
    for inst in wrong:
        wrong_confidence.append(instance_confidence[inst])

    mean_correct_confidence = np.mean(correct_confidence)
    mean_wrong_confidence = np.mean(wrong_confidence)

    print('Confidence of corrects: {}, Confidence of wrongs: {}'.format(mean_correct_confidence, mean_wrong_confidence))
    return correct_confidence, wrong_confidence


def model_confusion_experiment(models_path, device='cpu'):
    sdn_name = 'tinyimagenet_vgg16bn_sdn_ic_only'
    cnn_name = 'tinyimagenet_vgg16bn_cnn'

    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = af.get_dataset(sdn_params['task'])

    sdn_images = 'confusion_images/{}'.format(sdn_name)
    cnn_images = 'confusion_images/{}'.format(cnn_name)

    cnn_model, _ = arcs.load_model(models_path, cnn_name, epoch=-1)
    cnn_model.to(device)

    top1_test, top5_test = mf.sdn_test(sdn_model, dataset.test_loader, device)
    print('SDN Top1 Test accuracy: {}'.format(top1_test))
    print('SDN Top5 Test accuracy: {}'.format(top5_test))

    top1_test, top5_test = mf.cnn_test(cnn_model, dataset.test_loader, device)
    print('CNN Top1 Test accuracy: {}'.format(top1_test))
    print('CNN Top5 Test accuracy: {}'.format(top5_test))


    # the the normalization stats from the training set
    confusion_stats = mf.sdn_confusion_stats(sdn_model, loader=dataset.train_loader, device=device)
    print(confusion_stats)
    # SETTING 1 - IN DISTRIBUTION
    sdn_layer_correct, sdn_layer_wrong, sdn_instance_confusion = mf.sdn_get_confusion(sdn_model, loader=dataset.test_loader, confusion_stats=confusion_stats, device=device)
    sdn_correct_confusion, sdn_wrong_confusion = get_sdn_stats(sdn_layer_correct, sdn_layer_wrong, sdn_instance_confusion)

    
    cnn_correct, cnn_wrong, cnn_instance_confidence = mf.cnn_get_confidence(cnn_model, loader=dataset.test_loader, device=device)
    cnn_correct_confidence, cnn_wrong_confidence = get_cnn_stats(cnn_correct, cnn_wrong, cnn_instance_confidence)

    af.create_path(sdn_images)
    af.create_path(cnn_images)

    # corrects and wrongs
    af.overlay_two_histograms(sdn_images, '{}_corrects_wrongs'.format(sdn_name), sdn_correct_confusion, sdn_wrong_confusion, 'Correct', 'Wrong', 'SDN Confusion')
    af.overlay_two_histograms(cnn_images, '{}_corrects_wrongs'.format(cnn_name), cnn_correct_confidence, cnn_wrong_confidence, 'Correct', 'Wrong', 'CNN Confidence')

    mean_first = np.mean(sdn_correct_confusion)
    mean_second = np.mean(sdn_wrong_confusion)

    in_first_above_second = 0
    in_second_below_first = 0

    for item in sdn_correct_confusion:
        if float(item) > float(mean_second):
            in_first_above_second += 1

    for item in sdn_wrong_confusion:
        if float(item) < float(mean_first):
            in_second_below_first += 1


    print('SDN more confused correct: {}/{}'.format(in_first_above_second, len(sdn_correct_confusion)))
    print('SDN less confused wrong: {}/{}'.format(in_second_below_first, len(sdn_wrong_confusion)))


    mean_first = np.mean(cnn_correct_confidence)
    mean_second = np.mean(cnn_wrong_confidence)

    in_first_below_second = 0
    in_second_above_first = 0
    for item in cnn_correct_confidence:
        if float(item) < float(mean_second):
            in_first_below_second += 1
    for item in cnn_wrong_confidence:
        if float(item) > float(mean_first):
            in_second_above_first += 1

    print('CNN less confident correct: {}/{}'.format(in_first_below_second, len(cnn_correct_confidence)))
    print('CNN more confident wrong: {}/{}'.format(in_second_above_first, len(cnn_wrong_confidence)))

    # all instances
    sdn_all_confusion = list(sdn_instance_confusion.values())
    cnn_all_confidence = list(cnn_instance_confidence.values())


    print('Avg SDN Confusion: {}'.format(np.mean(sdn_all_confusion)))
    print('Std SDN Confusion: {}'.format(np.std(sdn_all_confusion)))

    print('Avg CNN Confidence: {}'.format(np.mean(cnn_all_confidence)))
    print('Std CNN Confidence {}'.format(np.std(cnn_all_confidence)))


    
def main():
    torch.manual_seed(af.get_random_seed())    # reproducible
    np.random.seed(af.get_random_seed())
    device = af.get_pytorch_device()
    trained_models_path = 'networks/{}'.format(af.get_random_seed())

    model_confusion_experiment(trained_models_path, device)

if __name__ == '__main__':
    main()