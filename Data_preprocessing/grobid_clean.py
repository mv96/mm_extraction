import os
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import copy
from itertools import groupby
from operator import itemgetter
import cv2
import math
from tqdm import tqdm
from pdf2image import convert_from_path
import shutil
import subprocess


class Preprocess_using_grobid:
    def __init__(self, show_results=False):
        self.show_results = show_results

    def read_file_xml_to_bs4(self, xml_file):
        """reads xml returns bs4 object"""
        with open(xml_file, "r") as file:
            # Read each line in the file, readlines() returns a list of lines
            content = file.readlines()
            # Combine the lines in the list into a string
            content = "".join(content)
            bs_content = bs(content, "xml")
            file.close()
        return bs_content

    def extract_header_info(self, bs_object, show_results=False):
        # get the header information
        try:
            header_info = bs_object.teiHeader
        except:
            header_info = ""
        # what can we get inside the header

        ##title
        try:
            title = header_info.find("title").get_text()
        except:
            title = ""

        ##publisher
        try:
            publisher = header_info.find("publisher").get_text()
        except:
            publisher = ""

        ##date published
        try:
            date_published = header_info.find("date").get_text()
        except:
            date_published = ""

        # authors
        try:
            forenames = header_info.find_all("forename")
            surnames = header_info.find_all("surname")
            emails = header_info.find_all("email")
            organisations = header_info.find_all("orgName")

            persons = []
            for e1, e2, e3, e4 in zip(forenames, surnames, emails, organisations):
                persons.append(
                    [e1.get_text() + " " + e2.get_text(), e3.get_text(), e4.get_text()]
                )
        except:
            persons = []
            # keywords of the paper
        try:
            keywords = header_info.keywords
            keywords = keywords.find_all("term")
            keywords = [element.get_text() for element in keywords]
        except:
            keywords = []

        # abstract information text
        try:
            abstract_info = header_info.abstract
            abstract = [abstract.get_text() for abstract in abstract_info.find_all("s")]
            abstract = "".join(abstract)
        except:
            abstract = ""

        # resolution and page_count
        try:
            test = bs_object.facsimile.find_all("surface")
            resolution = [
                [float(element["lrx"]), float(element["lry"])] for element in test
            ]
            page_count = len(resolution)
        except:
            resolution = []
            page_count = []

        res = [
            title,
            publisher,
            date_published,
            persons,
            keywords,
            abstract,
            resolution,
            page_count,
        ]

        if show_results is True:
            for element in res:
                print(element)

        return res

    def write_images(self, pdf_path, pages):
        # print("writing_images")
        # get images of the pdf
        images = convert_from_path(
            pdf_path, first_page=1, use_pdftocairo=True, fmt="png"
        )

        # but first check if path exists
        path = pdf_path.rsplit("/", 1)[0] + "/images"
        # create if the directory does not exist

        if os.path.exists(path):
            shutil.rmtree(path)

        os.mkdir(path)

        i = 0
        for image in images:
            # image.show()
            image.save(path + "/" + "image_" + str(i + 1) + ".png", "PNG")
            i += 1

        return images

    def get_paragraph_information(self, bs_content):
        text_1 = bs_content.TEI.body
        text_2 = bs_content.TEI.back

        tags = text_1.find_all(["head", "p", "formula"])
        tags += text_2.find_all(["head", "p", "formula"])

        return tags

    def extract_coords(self, coords, merge=False):
        t = coords.split(";")
        if len(t) == 1:
            t = t[0]
            t = t.split(",")
            if merge is False:
                coords = [
                    int(t[0]),
                    (float(t[1]), float(t[2])),
                    (float(t[3]), float(t[4])),
                ]
                return coords
            else:
                coords = [int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4])]
                return coords

    def convert_coords(self, list_of_coords_raw):
        new = [element.split(",") for element in list_of_coords_raw]
        # converting part
        total = []
        for element in new:
            n_element = []
            for sub_element in element:
                n_element.append(float(sub_element))
            total.append(n_element)

        for i, element in enumerate(total.copy()):
            n_element = [
                int(element[0]),
                element[1],
                element[2],
                element[3] + element[1],
                element[4] + element[2],
            ]
            total[i] = n_element

        return total

    def extract_coords_v2(self, coords, merge=False):
        # heading not in same page then skip
        same_pages = len(set([element.split(",")[0] for element in coords.split(";")]))
        if same_pages:
            coords = coords.split(";")
            new = self.convert_coords(coords)
            new = self.merge_dumb(new)
            return new
        else:
            print("head coords not in same page")

    def merge_dumb(self, list_of_coords):
        # map entire
        # print(list_of_coords)

        t = len(set([element[0] for element in list_of_coords]))
        if t == 1:
            page_no = list_of_coords[0][0]
            list_of_coords = list(map(lambda x: x[1:], list_of_coords))

            if isinstance(list_of_coords[0][0], tuple):
                all_coord = []
                for element in list_of_coords:
                    for sub_element in element:
                        all_coord.append(sub_element)

                all_x = [coord[0] for coord in all_coord]
                all_y = [coord[1] for coord in all_coord]

                return [page_no, [min(all_x), min(all_y)], [max(all_x), max(all_y)]]

            else:
                # we need min max of both x and y to reconstruct
                min_x_so_far = 10000000
                min_y_so_far = 10000000
                max_x_so_far = -1
                max_y_so_far = -1

                for element in list_of_coords:
                    min_x_so_far = min(element[0], element[2], min_x_so_far)
                    min_y_so_far = min(element[1], element[3], min_y_so_far)
                    max_x_so_far = max(element[0], element[2], max_x_so_far)
                    max_y_so_far = max(element[1], element[3], max_y_so_far)

                # print([page_no,[min_x_so_far,min_y_so_far],[max_x_so_far,max_y_so_far]])

                return [
                    page_no,
                    [min_x_so_far, min_y_so_far],
                    [max_x_so_far, max_y_so_far],
                ]
        else:
            counts = {}
            for element in list_of_coords:
                k = element[0]
                v = element[1:]
                if k not in counts:
                    counts[k] = v
                else:
                    counts[k] + v

            o_ps = []
            for k, v in counts.items():
                o_ps.append(self.merge_dumb([[k] + v]))

            return o_ps

    def smart_merge(self, list_of_coords):
        page_no = int(list_of_coords[0][0])
        o_p = self.merge_dumb(list_of_coords)

        # print(list_of_coords)
        # print(o_p)
        # print(list_of_coords[-1])
        # we need 5 coordinates to draw the polygon
        coord_1 = tuple(o_p[1])
        max_max = tuple(o_p[2])
        coord_2 = max_max[0], o_p[1][1]
        height = list_of_coords[-1][4] - list_of_coords[-1][2]
        coord_3 = max_max[0], list_of_coords[-1][4] - height
        coord_4 = list_of_coords[-1][3], list_of_coords[-1][4] - height
        coord_5 = list_of_coords[-1][3], max_max[1]
        coord_6 = o_p[1][0], max_max[1]
        # print([page_no,coord_1,coord_2,coord_3,coord_4,coord_5,coord_6])
        return [page_no, coord_1, coord_2, coord_3, coord_4, coord_5, coord_6]

    def sentence_merge_same_page(self, sentence_coords):
        sent_coords = sentence_coords.split(";")
        list_of_coords = list(
            map(lambda x: self.extract_coords(x, merge=True), sent_coords)
        )
        n = self.smart_merge(list_of_coords)

        return n

    def check_for_different_page(self, list_of_coords_raw):
        """if True means part of coords lie in a different page"""
        test = [element.split(",") for element in list_of_coords_raw]
        state = set([element[0] for element in test])
        if len(state) != 1:
            return True
        else:
            return False

    def extract_coords_for_para(self, para):
        # check if the coordinates are one the same page

        sentences = para.find_all("s")  # all s in para p

        res = []

        sent_coords_text = [
            [element["coords"].split(";"), element.text] for element in sentences
        ]
        sent_coords = [element[0] for element in sent_coords_text]

        all_sent_coords = []

        for element in sent_coords:
            all_sent_coords += element

        status = len(set([int(element.split(",")[0]) for element in all_sent_coords]))
        if status != 1:  # every sentence not on same page
            page_wise_box = {}  # container for each page

            for element in sent_coords_text:
                state = self.check_for_different_page(element[0])  # on sentence
                converted_coord = self.convert_coords(element[0])
                text = element[1]
                if state is False:
                    new_coord = self.smart_merge(converted_coord)
                    page_no = new_coord[0]
                    if page_no not in page_wise_box:
                        page_wise_box[page_no] = [[new_coord, element[1]]]
                    else:
                        page_wise_box[page_no].append([new_coord, element[1]])
                else:
                    t = self.convert_coords(element[0]) + [element[1]]
                    pagess = list(set([element[0] for element in t[:-1]]))
                    pagess.sort()
                    start = [[] for i in range(len(pagess))]
                    for element in t[:-1]:
                        page_c = element[0]
                        ind = pagess.index(page_c)
                        start[ind].append(element)
                    start = list(map(lambda x: self.smart_merge(x), start))
                    for i, element in enumerate(start):
                        page_no = element[0]
                        if i == 0:
                            creation = [element, text]
                        else:
                            creation = [element, ""]
                        if page_no not in page_wise_box:
                            page_wise_box[page_no] = [creation]
                        else:
                            page_wise_box[page_no].append(creation)

            # merge page wise box
            for k, v in page_wise_box.copy().items():
                coords = self.merge_dumb([element[0] for element in v])
                complete_text = "".join([element[1] for element in v])
                page_wise_box[k] = [coords, complete_text]

            break_para = list(page_wise_box.values())
            res.append(break_para)

        else:
            new = [[element["coords"], element.text] for element in sentences]
            new = [
                [self.convert_coords(element[0].split(";")), element[1]]
                for element in new
            ]

            all_coords = []
            all_text = []

            for element in new:
                all_text.append(element[1])
                for coord in element[0]:
                    all_coords.append(coord)

            text = "".join(all_text)
            coords = self.merge_dumb(all_coords)

            res.append([coords, text])

        return res

    def big_blocks_tags_to_coords(self, big_blocks_tags):
        coords_all = []

        for element in big_blocks_tags:
            # three posibilities
            if str(element).startswith("<head"):  # means head
                text = element.text

                try:
                    coords = element["coords"]
                    coords = extract_coords(coords)
                    coords = (
                        coords[0],
                        (coords[1][0], coords[1][1]),
                        (coords[1][0] + coords[2][0], coords[1][1] + coords[2][1]),
                    )

                except:
                    try:
                        coords = element["coords"]
                        coords = self.extract_coords_v2(coords)
                    except:
                        coords = None

                res = [coords, text]
                coords_all.append(res)

            elif str(element).startswith("<p"):  # means para
                # print("para")
                text = element.text

                try:
                    coords = element["coords"]
                except:
                    try:
                        coords = self.extract_coords_for_para(element)
                    except:
                        continue

                    for element in coords:
                        try:
                            coords_all.append([element[0], element[1]])
                        except:
                            continue

            elif str(element).startswith("<formula"):  # means formula
                text = element.text
                try:
                    coords = element["coords"]
                    coords = self.extract_coords(coords)
                    coords = (
                        coords[0],
                        (coords[1][0], coords[1][1]),
                        (coords[1][0] + coords[2][0], coords[1][1] + coords[2][1]),
                    )

                except:
                    print("cant find coordiantes in formula")
                    coords = None
                res = [coords, text]
                coords_all.append(res)

        return coords_all

    def rescale(self, grobid_xml, coordinates, page_no, resolution):
        """rescales coordinates to fit the enlarged image size"""
        # get scale factors
        coords = []
        for element in coordinates:
            if isinstance(element[1], list):
                for sub_element in element:
                    coords.append(sub_element)
            else:
                coords.append(element)

        # print(coords)

        scales = []
        # we need to rescale the coordinates according to the increased dimension
        for i in range(1, page_no + 1):
            images_dir = grobid_xml.rsplit("/", 1)[0] + "/images"
            image_path = images_dir + f"/image_{i}.png"
            image = cv2.imread(image_path)
            # print(image.shape)
            # print(resolution[i-1])
            scale_x = image.shape[0] / resolution[i - 1][1]
            scale_y = image.shape[1] / resolution[i - 1][0]
            # print(scale_x,scale_y)
            # print(resolution)
            scales.append([scale_x, scale_y])

        self.scales = scales

        new_coords = []
        for element in coords.copy():
            if element[0] is None:
                continue
            else:
                element[0][1]
                element[0][2]
                scale_ind = scales[element[0][0] - 1]

                new_coords.append(
                    [
                        element[0][0],
                        [
                            element[0][1][0] * scale_ind[0],
                            element[0][1][1] * scale_ind[1],
                        ],
                        [
                            element[0][2][0] * scale_ind[0],
                            element[0][2][1] * scale_ind[1],
                        ],
                        element[1],
                    ]
                )

        return new_coords

    def coords_to_dict(self, blocks_of_coords):
        empty = {}
        for element in blocks_of_coords:
            if element[0] not in empty:
                empty[element[0]] = [element]
            else:
                empty[element[0]].append(element)
        return empty

    def sort_coordinates_in_dict(self, coords_dict):
        for k, v in coords_dict.copy().items():
            top_left_x = []
            top_left_y = []
            bot_right_x = []
            bot_right_y = []
            page_no = []
            texts = []
            for element in v:
                page_no.append(element[0])
                top_left_x.append(element[1][0])
                top_left_y.append(element[1][1])
                bot_right_x.append(element[2][0])
                bot_right_y.append(element[2][1])
                texts.append(element[3])

            for_df = pd.DataFrame(
                [page_no, top_left_x, top_left_y, bot_right_x, bot_right_y, texts]
            ).T
            for_df.columns = [
                "page_no",
                "top_left_x",
                "top_left_y",
                "bot_right_x",
                "bot_right_y",
                "text",
            ]
            # sort for_df
            # sort order "top_left_y","top_left_x","bot_right_y","bot_right_x"
            for_df = for_df.sort_values(
                by=["top_left_y", "top_left_x", "bot_right_y", "bot_right_x"],
                ascending=[True, True, True, True],
            )
            n_list = for_df.values.tolist()
            sorted_list = []

            for element in n_list:
                sorted_list.append(
                    [
                        int(element[0]),
                        (element[1], element[2]),
                        (element[3], element[4]),
                        element[5],
                    ]
                )

            coords_dict[k] = sorted_list

        return coords_dict

    def visualize_boxes(self, grobid_xml, dict_coords):
        print("inside_visualizations")
        # visualize the boxes
        track = []
        texts = []
        coords = []

        # getting patches ready
        for k, v in tqdm(dict_coords.items()):
            for element in v:
                image, text = self.draw_rectangle(
                    grobid_xml.rsplit("/", 1)[0] + "/images", box=element
                )
                track.append(image)
                texts.append(text)
                coords.append(element)

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

    def draw_rectangle(self, images_dir, box):
        x1, y1, x2, y2 = (
            math.floor(box[1][0]),
            math.floor(box[1][1]),
            math.ceil(box[2][0]),
            math.ceil(box[2][1]),
        )

        # determine i
        i = int(box[0])
        image_path = images_dir + f"/image_{i}.png"
        image = cv2.imread(image_path)
        # draw rectangles
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = box[3]

        return image, text

    def compare_blocks(self, boxa, boxb):
        small_box_coords = boxa[1], boxa[2]
        big_box_coords = boxb[1], boxb[2]
        small_box_text = boxa[3]
        big_box_text = boxb[3]

        if (
            big_box_coords[0][0] <= small_box_coords[0][0] <= big_box_coords[1][0]
            and big_box_coords[0][1] <= small_box_coords[0][1] <= big_box_coords[1][1]
            and big_box_coords[0][0] <= small_box_coords[1][0] <= big_box_coords[1][0]
            and big_box_coords[0][1] <= small_box_coords[0][1] <= big_box_coords[1][1]
        ):
            return True
        else:
            return False

    def return_text(self, xml_data, box1, box2):
        cms = [element[3] for element in xml_data]
        # print(cms)
        text1 = box1[3]
        text2 = box2[3]

        # see if text 1 in text 2
        ind1 = cms.index(text1)
        ind2 = cms.index(text2)

        if ind1 < ind2:
            final_text = text1 + " " + text2
        else:
            final_text = text2 + " " + text1

        # print("="*20)
        # print(text1)
        # print("="*20)
        # print(text2)
        # print("="*20)
        # print(final_text)
        return final_text

    def sub_boxes_single_update(self, dict_of_coords):
        for page, para_boxes in dict_of_coords.items():
            for box1 in para_boxes:
                for box2 in para_boxes:
                    if box1 == box2:
                        continue
                    if self.compare_blocks(box1, box2):
                        # print("box1 in box2")
                        # print(page)
                        new_box = box2[:3] + [
                            self.return_text(dict_of_coords[page], box1, box2)
                        ]
                        # print("updating")
                        # print(len(dict_of_coords[page]))
                        para_boxes.remove(box1)
                        para_boxes.remove(box2)
                        para_boxes.append(new_box)
                        # print(len(dict_of_coords[page]))
                        return dict_of_coords
                    elif self.compare_blocks(box2, box1):
                        # print("box 2 in box1")
                        # print(page)
                        new_box = box1[:3] + [
                            self.return_text(dict_of_coords[page], box1, box2)
                        ]
                        # print("updating")
                        # print(len(dict_of_coords[page]))
                        para_boxes.remove(box1)
                        para_boxes.remove(box2)
                        para_boxes.append(new_box)
                        # print(len(dict_of_coords[page]))
                        return dict_of_coords
                    else:
                        # print("not fitting")
                        continue
        return False

    def filter_self_containing_boxes(self, dict_of_coords):
        count = 0
        while True:
            val = self.sub_boxes_single_update(dict_of_coords)
            if val is False:
                break
            else:
                dict_of_coords = val
                count += 1
        # print(count)

        return dict_of_coords

    def merge_overlaps(self, dict_of_coords):
        count = 0
        while True:
            val = self.merge_overlaps_single_update(dict_of_coords)
            if val is False:
                break
            else:
                dict_of_coords = val
                count += 1
        # print(count)

        return dict_of_coords

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

    def merge_overlaps_single_update(self, dict_of_coords):
        for page, para_boxes in dict_of_coords.items():
            for box1 in para_boxes:
                for box2 in para_boxes:
                    if box1 == box2:
                        continue
                    else:
                        bb1 = {
                            "x1": box1[1][0],
                            "y1": box1[1][1],
                            "x2": box1[2][0],
                            "y2": box1[2][1],
                        }
                        bb2 = {
                            "x1": box2[1][0],
                            "y1": box2[1][1],
                            "x2": box2[2][0],
                            "y2": box2[2][1],
                        }
                        if self.get_iou(bb1, bb2) > 0:
                            # check which box is first
                            # see the top left coordinate to match
                            # generate combined box
                            # now replace this box within the dictionary
                            merged_coordinates = self.merge_dumb(
                                [
                                    [
                                        box1[0],
                                        box1[1][0],
                                        box1[1][1],
                                        box1[2][0],
                                        box1[2][1],
                                    ],
                                    [
                                        box2[0],
                                        box2[1][0],
                                        box2[1][1],
                                        box2[2][0],
                                        box2[2][1],
                                    ],
                                ]
                            )

                            if (
                                bb1["x1"] < bb2["x1"]
                            ):  # only uses y coordinates to compare
                                # means box1 is left
                                merged_text = box1[3] + " " + box2[3]
                            else:
                                merged_text = box2[3] + " " + box1[3]

                            new_box = merged_coordinates + [merged_text]
                            para_boxes.remove(box1)
                            para_boxes.remove(box2)
                            para_boxes.append(new_box)
                            return dict_of_coords
                        else:
                            continue

        return False

    def remove_pdf_links(self, inp_pdf, out_pdf=None):
        if out_pdf is None:
            out_pdf = inp_pdf.rsplit(".", 1)
            out_pdf_temp = out_pdf[0] + "_stripped_temp." + out_pdf[1]
            out_pdf = out_pdf[0] + "_stripped." + out_pdf[1]

        # code for conversion
        first_cmd = " ".join(
            [
                "pdftk",
                "{}".format(inp_pdf),
                "output",
                "{}".format(out_pdf_temp),
                "uncompress",
            ]
        )
        second_cmd = "LC_ALL=C sed -n '/^\/Annots/!p' {} > {}".format(
            out_pdf_temp, inp_pdf
        )

        # call first command

        subprocess.call(first_cmd, shell=True)

        # call second command
        ret = subprocess.call(second_cmd, shell=True)

        os.remove(out_pdf_temp)

    def fit(self, grobid_xml, show_results=False):
        ###################step 1
        # give xml and get a bs4 object
        bs_content = self.read_file_xml_to_bs4(grobid_xml)

        ################## step 2

        # gives the header information - title, publisher, date published, persons,
        # keywords,abstrabstract
        # resolution page count
        header_info = self.extract_header_info(bs_content)

        # write images in the pdf directory if not present
        ### insert here link removal
        self.remove_pdf_links(inp_pdf=grobid_xml.replace(".tei.xml", ".pdf"))

        self.write_images(grobid_xml.replace(".tei.xml", ".pdf"), header_info[-1])

        ################## step 3 extract bigger blocks
        # big block is heading or
        # a paragraph or
        # a formula block
        # good thing sequence is mentained
        big_blocks_tags = self.get_paragraph_information(bs_content)

        ################## step 4 convert html tags to understandable coordinates
        res = self.big_blocks_tags_to_coords(big_blocks_tags)  ###here

        # print(res)

        # rescale
        nc = self.rescale(
            grobid_xml=grobid_xml,
            coordinates=res,
            page_no=header_info[-1],
            resolution=header_info[-2],
        )

        # make dictionary
        coords_dict = self.coords_to_dict(nc)

        # fix subblocks with blocks
        # print(dict_coords[3])
        coords_dict = self.filter_self_containing_boxes(coords_dict)

        # visualize_boxes(coords_dict)

        # merge overlapping boxes
        coords_dict = self.merge_overlaps(coords_dict)

        # sort these missing coordinates by spatial index
        dict_coords = self.sort_coordinates_in_dict(coords_dict)

        # print(dict_coords)

        # visualize the boxes
        if show_results == True:
            print("generating visualizations")
            self.visualize_boxes(grobid_xml, dict_coords)
            return dict_coords

        else:
            return dict_coords
