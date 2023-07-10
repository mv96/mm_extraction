from grobid_clean import Preprocess_using_grobid
from visualize_annot import annotations_page
import pandas as pd
import cv2
import math
from tqdm import tqdm


class assigning_labels:
    def __init__(self, show_images=False, grobid_xml=None, labels_xml=None):
        self.labels_xml = labels_xml
        self.show_images = show_images
        self.path_images = labels_xml.rsplit("/", 1)[0] + "/images"
        self.scales = None
        self.grobid_xml = grobid_xml

    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]

        source -
        https://stackoverflow.com/questions/
        25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
        """
        assert bb1["x1"] < bb1["x2"]
        assert bb1["y1"] < bb1["y2"]
        assert bb2["x1"] < bb2["x2"]
        assert bb2["y1"] < bb2["y2"]

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1["x1"], bb2["x1"])
        y_top = max(bb1["y1"], bb2["y1"])
        x_right = min(bb1["x2"], bb2["x2"])
        y_bottom = min(bb1["y2"], bb2["y2"])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
        bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def visualize_boxes_merged(self, grobid_xml, dict_coords):
        def draw_rectangle(images_dir, box):
            x1, y1, x2, y2 = (
                math.floor(box[1][0]),
                math.floor(box[1][1]),
                math.ceil(box[2][0]),
                math.ceil(box[2][1]),
            )
            label = box[4]

            if label == "basic":
                label_color = (255, 0, 0)  # basic=blue
            elif label == "overlap":
                label_color = (0, 0, 255)  # overlap in red
            else:
                label_color = (0, 255, 0)  # everything normal in green
            # determine i
            i = int(box[0])
            image_path = images_dir + f"/image_{i}.png"
            image = cv2.imread(image_path)
            # draw rectangles
            image = cv2.rectangle(image, (x1, y1), (x2, y2), label_color, 2)
            text = box[3]

            return image, text, label

        # visualize the boxes
        track = []
        texts = []
        coords = []
        labels = []

        # getting patches ready
        for k, v in tqdm(dict_coords.items()):
            for element in v:
                image, text, label = draw_rectangle(
                    grobid_xml.rsplit("/", 1)[0] + "/images", box=element
                )
                track.append(image)
                texts.append(text)
                coords.append(element)
                labels.append(label)

        image_pointer = 0
        # show start
        cv2.imshow("test", track[image_pointer])
        print(texts[image_pointer])
        while True:
            try:
                k = cv2.waitKey(0)
                if k == ord("a"):  # means go back and display prev image
                    if image_pointer > 0:
                        image_pointer -= 1
                        cv2.imshow("test", track[image_pointer])
                        # print(texts[image_pointer])
                        print(coords[image_pointer])
                elif k == ord("d"):
                    image_pointer += 1
                    cv2.imshow("test", track[image_pointer])
                    # print(texts[image_pointer])
                    print(coords[image_pointer])
                elif k == ord("c"):
                    cv2.destroyAllWindows()
                    break
                else:
                    continue
            except IndexError:
                cv2.destroyAllWindows()
                print("End of PDF")

    def fit(self):
        grobid_xml, labels_xml = self.grobid_xml, self.labels_xml
        # traverse through each of the annotation box
        # further traverse through each of the boxes on the given page
        # for every box in the grobid see if this box fits under the  annotation
        # if yes then merge it all together
        prep = Preprocess_using_grobid()
        final = prep.fit(grobid_xml=grobid_xml, show_results=False)
        self.scales = prep.scales

        annot = annotations_page(labels_xml=labels_xml, show_images=False)
        annotations = annot.fit()  ######### error##############
        margin = 10
        main = []
        overlapping = False
        name_of_the_box = ""

        for k in final.keys():
            if k not in annotations.keys():
                for element in final[k]:
                    main.append(element + ["basic"])
            else:
                for page_box in final[k]:  # grobid para in grobid paras
                    get_out = False
                    for annot_box in annotations[k]:
                        if (
                            annot_box[1][0] - margin
                            <= page_box[1][0]
                            <= annot_box[2][0] + margin
                            and annot_box[1][1] - margin
                            <= page_box[1][1]
                            <= annot_box[2][1] + margin
                            and annot_box[1][0] - margin
                            <= page_box[2][0]
                            <= annot_box[2][0] + margin
                            and annot_box[1][1] - margin
                            <= page_box[2][1]
                            <= annot_box[2][1] + margin
                        ):  # means box fits completely
                            if overlapping is not True:
                                main.append(page_box + [annot_box[3]])
                                get_out = True
                                break
                            else:
                                name_of_the_box = name_of_the_box + "_" + annot_box[3]
                                continue
                        else:
                            bb1 = {
                                "x1": page_box[1][0],
                                "y1": page_box[1][1],
                                "x2": page_box[2][0],
                                "y2": page_box[2][1],
                            }
                            bb2 = {
                                "x1": annot_box[1][0],
                                "y1": annot_box[1][1],
                                "x2": annot_box[2][0],
                                "y2": annot_box[2][1],
                            }
                            # print(page_box,annot_box)
                            if self.get_iou(bb1, bb2) != 0:
                                if name_of_the_box == "":
                                    name_of_the_box = "overlap" + "_" + annot_box[3]
                                    overlapping = True  # set this flag to true to know that this page already had some ovrlap
                                else:
                                    name_of_the_box = (
                                        name_of_the_box + "_" + annot_box[3]
                                    )
                                    # there might be a possibility that this page block contains other annotations too
                                    continue
                            else:
                                if overlapping is True:
                                    # overlapping box
                                    main.append(page_box + [name_of_the_box])
                                    overlapping = False
                                    get_out = True
                                    break
                    if get_out is True:
                        continue
                    else:
                        main.append(page_box + ["basic"])

                    # for the box that does not fit completely may partially fit
                    #############keep a track of it #we might need to filter it using pdfalto

        annotated = pd.DataFrame(
            data=main, columns=["page_no", "top_left", "bot_right", "text", "label"]
        )

        if self.show_images is True:
            dict_coords = {}
            for ind, row in annotated.iterrows():
                if row["page_no"] not in dict_coords:
                    dict_coords[row["page_no"]] = [
                        [
                            row["page_no"],
                            row["top_left"],
                            row["bot_right"],
                            row["text"],
                            row["label"],
                        ]
                    ]
                else:
                    dict_coords[row["page_no"]].append(
                        [
                            row["page_no"],
                            row["top_left"],
                            row["bot_right"],
                            row["text"],
                            row["label"],
                        ]
                    )
            self.visualize_boxes_merged(grobid_xml, dict_coords)

        return annotated, self.scales
