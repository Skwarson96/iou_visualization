import numpy as np
import torch
from torchvision import ops
import cv2


def move_rectangle(x, y, param, box_nr, color):
    if param["old_box_" + box_nr][0] is not None:
        cv2.rectangle(
            param["img"],
            (param["old_box_" + box_nr][0], param["old_box_" + box_nr][1]),
            (param["old_box_" + box_nr][2], param["old_box_" + box_nr][3]),
            (0, 0, 0),
            3,
        )
    else:
        cv2.rectangle(
            param["img"],
            (int(param["box_" + box_nr][0]), int(param["box_" + box_nr][1])),
            (int(param["box_" + box_nr][2]), int(param["box_" + box_nr][3])),
            (0, 0, 0),
            3,
        )

    x_min = x - (param["box_" + box_nr][2] - param["box_" + box_nr][0]) / 2
    y_min = y - (param["box_" + box_nr][3] - param["box_" + box_nr][1]) / 2
    x_max = x + (param["box_" + box_nr][2] - param["box_" + box_nr][0]) / 2
    y_max = y + (param["box_" + box_nr][3] - param["box_" + box_nr][1]) / 2

    cv2.rectangle(
        param["img"], (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 0), -1
    )
    cv2.rectangle(
        param["img"], (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 3
    )

    param["box_" + box_nr] = [int(x_min), int(y_min), int(x_max), int(y_max)]
    param["old_box_" + box_nr] = [int(x_min), int(y_min), int(x_max), int(y_max)]


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        if (
            param["box_1"][0] < x < param["box_1"][2]
            and param["box_1"][1] < y < param["box_1"][3]
        ):
            param["box_1_move"] = True
        if (
            param["box_2"][0] < x < param["box_2"][2]
            and param["box_2"][1] < y < param["box_2"][3]
        ):
            param["box_2_move"] = True

    elif event == cv2.EVENT_MOUSEMOVE and param["box_1_move"] is True:
        x_border = int((param["box_1"][2] - param["box_1"][0]) / 2)
        y_border = int((param["box_1"][3] - param["box_1"][1]) / 2)

        if (
            x_border < x < param["img"].shape[1] - x_border
            and y_border < y < param["img"].shape[0] - y_border
        ):

            box_nr = "1"
            move_rectangle(x, y, param, box_nr, param["box_1_color"])

            cv2.rectangle(
                param["img"],
                (int(param["box_2"][0]), int(param["box_2"][1])),
                (int(param["box_2"][2]), int(param["box_2"][3])),
                param["box_2_color"],
                3,
            )

    elif event == cv2.EVENT_MOUSEMOVE and param["box_2_move"] is True:
        x_border = int((param["box_1"][2] - param["box_1"][0]) / 2)
        y_border = int((param["box_1"][3] - param["box_1"][1]) / 2)

        if (
            x_border < x < param["img"].shape[1] - x_border
            and y_border < y < param["img"].shape[0] - y_border
        ):

            box_nr = "2"
            move_rectangle(x, y, param, box_nr, param["box_2_color"])

            cv2.rectangle(
                param["img"],
                (int(param["box_1"][0]), int(param["box_1"][1])),
                (int(param["box_1"][2]), int(param["box_1"][3])),
                param["box_1_color"],
                3,
            )

    elif event == cv2.EVENT_LBUTTONUP:
        param["box_1_move"] = False
        param["box_2_move"] = False


def iou_calculate(img, box_1, box_2):

    ground_truth_bbox = torch.tensor([box_1], dtype=torch.float)
    prediction_bbox = torch.tensor([box_2], dtype=torch.float)

    iou = ops.box_iou(ground_truth_bbox, prediction_bbox)

    cv2.rectangle(img, (0, img.shape[0] - 30), (120, img.shape[0]), (0, 0, 0), -1)

    cv2.putText(
        img=img,
        text=f"IOU : {str(np.round(iou.numpy()[0][0], 5))}",
        org=(0, img.shape[0] - 10),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=0.5,
        color=(0, 255, 255),
        thickness=1,
    )


def main():

    #       Xmin Ymin Xmax Ymax
    box_1 = [100, 100, 200, 200]
    old_box_1 = [None, None, None, None]
    box_2 = [300, 100, 400, 200]
    old_box_2 = [None, None, None, None]
    box_1_color = (0, 255, 0)
    box_2_color = (255, 0, 0)

    img = np.zeros((512, 512, 3), np.uint8)
    box_1_move = False
    box_2_move = False
    param = {
        "img": img,
        "box_1": box_1,
        "old_box_1": old_box_1,
        "box_1_move": box_1_move,
        "box_1_color": box_1_color,
        "box_2": box_2,
        "old_box_2": old_box_2,
        "box_2_move": box_2_move,
        "box_2_color": box_2_color,
    }
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", mouse_callback, param)

    #                   Xmin      Ymin        Xmax      Ymax
    cv2.rectangle(img, (box_1[0], box_1[1]), (box_1[2], box_1[3]), (0, 255, 0), 3)
    cv2.rectangle(img, (box_2[0], box_2[1]), (box_2[2], box_2[3]), (255, 0, 0), 3)

    # display the window
    while True:
        cv2.imshow("img", img)
        iou_calculate(param["img"], param["box_1"], param["box_2"])
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
