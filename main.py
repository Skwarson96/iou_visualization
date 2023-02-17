import numpy as np
import torch
from torchvision import ops
import cv2


def move_rectangle_2(x, y, param, box_nr, color):
    if param["old_box_"+box_nr][0] is not None:
        cv2.rectangle(param["img"], (param["old_box_"+box_nr][0], param["old_box_"+box_nr][1]),
                      (param["old_box_"+box_nr][2], param["old_box_"+box_nr][3]), (0, 0, 0), 3)
    else:
        cv2.rectangle(param["img"],
                      (int(param["box_"+box_nr][0]), int(param["box_"+box_nr][1])),
                      (int(param["box_"+box_nr][2]), int(param["box_"+box_nr][3])), (0, 0, 0), 3)

    x_min = x - (param["box_"+box_nr][2] - param["box_"+box_nr][0]) / 2
    y_min = y - (param["box_"+box_nr][3] - param["box_"+box_nr][1]) / 2
    x_max = x + (param["box_"+box_nr][2] - param["box_"+box_nr][0]) / 2
    y_max = y + (param["box_"+box_nr][3] - param["box_"+box_nr][1]) / 2
    # print((x_min, y_min), (x_max, y_max))
    cv2.rectangle(param["img"], (int(x_min), int(y_min)),
                  (int(x_max), int(y_max)), (0, 0, 0), -1)
    cv2.rectangle(param["img"], (int(x_min), int(y_min)),
                  (int(x_max), int(y_max)), color, 3)


    param["box_"+box_nr] = [int(x_min), int(y_min), int(x_max),
                      int(y_max)]
    param["old_box_"+box_nr] = [int(x_min), int(y_min), int(x_max),
                          int(y_max)]

def move_rectangle(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        if param["box_1"][0]<x<param["box_1"][2] and param["box_1"][1]<y<param["box_1"][3]:
            param["box_1_move"]=True
        if param["box_2"][0]<x<param["box_2"][2] and param["box_2"][1]<y<param["box_2"][3]:
            param["box_2_move"]=True

    elif event == cv2.EVENT_MOUSEMOVE and param["box_1_move"]==True:
        box_nr = "1"
        color = (0, 255, 0)
        cv2.rectangle(param["img"], (int(param["box_2"][0]), int(param["box_2"][1])),
                      (int(param["box_2"][2]), int(param["box_2"][3])), (255, 0, 0), 3)
        move_rectangle_2(x, y, param, box_nr, color)

    elif event == cv2.EVENT_MOUSEMOVE and param["box_2_move"] == True:
        box_nr = "2"
        color = (255, 0, 0)
        cv2.rectangle(param["img"], (int(param["box_1"][0]), int(param["box_1"][1])),
                      (int(param["box_1"][2]), int(param["box_1"][3])), (0, 255, 0), 3)
        move_rectangle_2(x, y, param, box_nr, color)


        # if param["old_box_1"][0] is not None:
        #     cv2.rectangle(param["img"], (param["old_box_1"][0], param["old_box_1"][1]),
        #                   (param["old_box_1"][2], param["old_box_1"][3]), (0, 0, 0), 3)
        # else:
        #     cv2.rectangle(param["img"],
        #                   (int(param["box_1"][0]), int(param["box_1"][1])),
        #                   (int(param["box_1"][2]), int(param["box_1"][3])), (0, 0, 0), 3)
        #
        # box_1_x_min = x-(param["box_1"][2]-param["box_1"][0])/2
        # box_1_y_min = y-(param["box_1"][3]-param["box_1"][1])/2
        # box_1_x_max = x+(param["box_1"][2]-param["box_1"][0])/2
        # box_1_y_max = y+(param["box_1"][3]-param["box_1"][1])/2
        # # print((x_min, y_min), (x_max, y_max))
        # cv2.rectangle(param["img"], (int(box_1_x_min), int(box_1_y_min)), (int(box_1_x_max), int(box_1_y_max)), (0, 0, 0), -1)
        # cv2.rectangle(param["img"], (int(box_1_x_min), int(box_1_y_min)), (int(box_1_x_max), int(box_1_y_max)), (0, 255, 0), 3)
        #
        # cv2.rectangle(param["img"], (int(param["box_2"][0]), int(param["box_2"][1])), (int(param["box_2"][2]), int(param["box_2"][3])), (255, 0, 0), 3)
        #
        #
        # param["box_1"]=[int(box_1_x_min), int(box_1_y_min), int(box_1_x_max), int(box_1_y_max)]
        # param["old_box_1"]=[int(box_1_x_min), int(box_1_y_min), int(box_1_x_max), int(box_1_y_max)]





    elif event == cv2.EVENT_LBUTTONUP:
        param["box_1_move"] = False
        param["box_2_move"] = False


def main():
    print("main")

    # Create a black image

    #       Xmin Ymin Xmax Ymax
    box_1 = [100, 100, 200, 200]
    old_box_1 = [None, None, None, None]
    box_2 = [300, 100, 400, 200]
    old_box_2 = [None, None, None, None]



    img = np.zeros((512, 512, 3), np.uint8)
    box_1_move = False
    box_2_move = False
    param = {"img": img, "box_1": box_1, "old_box_1": old_box_1, "box_1_move":box_1_move, "box_2": box_2, "old_box_2": old_box_2,"box_2_move":box_2_move}
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", move_rectangle, param)

    #                   Xmin      Ymin        Xmax      Ymax
    cv2.rectangle(img, (box_1[0], box_1[1]), (box_1[2], box_1[3]), (0, 255, 0), 3)
    cv2.rectangle(img, (box_2[0], box_2[1]), (box_2[2], box_2[3]), (255, 0, 0), 3)

    # display the window
    while True:
        cv2.imshow("img", img)
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()

    # # Bounding box coordinates.
    # ground_truth_bbox = torch.tensor([[1202, 123, 1650, 868]], dtype=torch.float)
    # prediction_bbox = torch.tensor([[1162.0001, 92.0021, 1619.9832, 694.0033]],
    #                                dtype=torch.float)
    #
    # # Get iou.
    # iou = ops.box_iou(ground_truth_bbox, prediction_bbox)
    # print('IOU : ', iou.numpy()[0][0])

if __name__ == '__main__':
    main()


