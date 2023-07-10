from bs4 import BeautifulSoup as bs
import os
from pdf2image import convert_from_path
import shutil
from tqdm import tqdm
import cv2
import math
import pandas as pd


class annotations_page:
    def __init__(self, labels_xml, show_images=False):
        self.labels_xml = labels_xml
        self.show_images = show_images
        self.path_images = labels_xml.rsplit("/", 1)[0] + "/images"

    def compute_skip(self, labels_xml):
        path_images = labels_xml.rsplit("/", 1)[0] + "/images"
        # read the pdfalto xml file to see for page_infos
        pdf_alto_xml = labels_xml.replace("_annot.xml", ".xml")
        # read using the beautiful soup
        bs_content = self.read_file_xml_to_bs4(pdf_alto_xml)
        results = bs_content.findAll("Page")
        if len(results) > len(os.listdir(path_images)):
            return True
        else:
            return False

    def read_file_xml_to_bs4(self, xml_file):
        """reads xml returns bs4 object"""
        with open(xml_file, "r") as file:
            # Read each line in the file, readlines() returns a list of lines
            content = file.readlines()
            # Combine the lines in the list into a string
            content = "".join(content)
            bs_content = bs(content, "xml")  ######xml version
            file.close()

        return bs_content

    def filter_texts(self, annot):
        try:
            text = annot.DEST.text
        except:
            return False
        if text.startswith("uri"):
            return True
        else:
            return False

    def tags_to_boxes(self, tag, skip):
        if skip is True:
            page_no = int(tag.get("pagenum")) - 1
        else:
            page_no = int(tag.get("pagenum"))
        label = tag.DEST.text
        coords = []
        for element in tag.QUADRILATERAL:
            coords.append((float(element.get("HPOS")), float(element.get("VPOS"))))
        return [page_no] + coords[1:3] + [label]

    def page_wise(self, boxes):
        empty = {}
        for element in boxes:
            if element[0] not in empty:
                empty[element[0]] = [element[1:]]
            else:
                empty[element[0]].append(element[1:])

        return empty

    def check_path_or_create(self, labels_xml):
        path_images = labels_xml.rsplit("/", 1)[0] + "/images"
        if os.path.exists(path_images):
            shutil.rmtree(path_images)
            os.mkdir(path_images)
            d = self.write_images(labels_xml, path_images)
        else:
            os.mkdir(path_images)
            d = self.write_images(labels_xml, path_images)
        return path_images

    def write_images(self, labels_xml, images_dir):
        pdf_path = labels_xml.replace("_annot.xml", ".pdf")
        if not os.path.exists(pdf_path):
            print("pdf can't be read")
            return False
        # get images of the pdf
        images = convert_from_path(
            pdf_path, first_page=1, use_pdftocairo=True, fmt="png"
        )
        i = 0
        for image in images:  # could be with tqdm to add progress bar
            # image.show()
            image.save(images_dir + "/" + "image_" + str(i + 1) + ".png", "PNG")
            i += 1

        return images

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
                o_ps.append(merge_dumb([[k] + v]))

            return o_ps

    def calculate_scaling_factor(self, annot_xml):
        real_xml = annot_xml.replace("_annot.xml", ".xml")

        # read the real xml and get the resolutions from it
        bs_content = self.read_file_xml_to_bs4(xml_file=real_xml)

        pages = bs_content.find_all("Page")  ##########fix here

        shape_actual = list(
            map(lambda x: [float(x.get("HEIGHT")), float(x.get("WIDTH"))], pages)
        )

        # get_rendered_image_resolution()
        image_dir = annot_xml.rsplit("/", 1)[0] + "/images"

        image_files = [image_dir + "/" + element for element in os.listdir(image_dir)]

        def get_image_size(img_file):
            img = cv2.imread(img_file, 0)
            height, width = img.shape[:2]
            return [height, width]

        shape_rendered = list(map(lambda x: get_image_size(x), image_files))

        # print(len(shape_actual),len(shape_rendered))

        ratios = []
        for e1, e2 in zip(shape_actual, shape_rendered):
            height_ratio = e2[0] / e1[0]
            width_ratio = e2[1] / e1[1]
            ratios.append([height_ratio, width_ratio])

        return ratios

        # calculate the scaling factor

    def rescale_dict(self, dict_pages, scale_ratios):
        resclaled_dict = {}
        for page, coords in dict_pages.items():
            for coord in coords:
                c1 = (
                    coord[0][0] * scale_ratios[page - 2][0],
                    coord[0][1] * scale_ratios[page - 2][1],
                )
                c2 = (
                    coord[1][0] * scale_ratios[page - 2][0],
                    coord[1][1] * scale_ratios[page - 2][1],
                )
                val = [page, c1, c2, coord[2]]
                if page not in resclaled_dict:
                    resclaled_dict[page] = [val]
                else:
                    resclaled_dict[page].append(val)

        return resclaled_dict

    def get_contours_of_image(self, image, show_image=False):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Find Canny edges
        edged = cv2.Canny(gray, 30, 200)

        contours, hierarchy = cv2.findContours(
            edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if show_image:
            # Draw all contours
            # Use '-1' as the 3rd parameter to draw all
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
            cv2.imshow("Contours:" + str(len(contours)), image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return len(contours)

    def filter_empty_boxes_using_contours(self, dict_pages):
        new_dict = {}
        # print("filtering empty boxes or page_nos falsely detected") #basically low contour boxes

        threshold = 4  # every thing greater than or equal to this threshold of contours will be preserved

        for k, v in tqdm(dict_pages.copy().items()):
            image_file = self.path_images + f"/image_{k}.png"

            image = cv2.imread(image_file)
            # print(image.shape)

            temp = []
            for box in v.copy():
                crop_img = image[
                    math.floor(box[1][1]) : math.ceil(box[2][1]),
                    math.floor(box[1][0]) : math.ceil(box[2][0]),
                ]
                try:
                    contour_count = self.get_contours_of_image(
                        crop_img, show_image=False
                    )

                except:
                    continue
                if contour_count >= threshold:
                    temp.append(box)
            new_dict[k] = temp
        return new_dict

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

    def sub_boxes_single_update(self, dict_of_coords):
        for page, para_boxes in dict_of_coords.items():
            for box1 in para_boxes:
                for box2 in para_boxes:
                    if box1 == box2:
                        continue
                    if self.compare_blocks(box1, box2):
                        para_boxes.remove(box2)
                        # print(len(dict_of_coords[page]))
                        return dict_of_coords
                    elif self.compare_blocks(box2, box1):
                        para_boxes.remove(box1)
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

    def visualize_boxes(self, dict_coords, labels_xml):
        # visualize the boxes
        track = []
        texts = []
        coords = []

        # getting patches ready
        for k, v in tqdm(dict_coords.items()):
            for element in v:
                image, text = self.draw_rectangle(
                    labels_xml.rsplit("/", 1)[0] + "/images", box=element
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

    def merge_single_label(self, dict_pages):
        new_dict = {}
        for k, v in dict_pages.items():
            d = list(set([element[3] for element in v]))
            for name in d:
                that_ = self.merge_dumb(
                    [
                        [coord[0], coord[1][0], coord[1][1], coord[2][0], coord[2][1]]
                        for coord in dict_pages[k]
                        if (coord[3] == name)
                    ]
                )
                if k not in new_dict:
                    new_dict[k] = [that_ + [name]]
                else:
                    new_dict[k].append(that_ + [name])
        return new_dict

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

    def fit(self):
        self.check_path_or_create(self.labels_xml)

        bs_content = self.read_file_xml_to_bs4(self.labels_xml)

        skip = self.compute_skip(self.labels_xml)

        # get all the annotation boxes with labels that start with uri
        annots = list(bs_content.find_all("ANNOTATION"))  # ANNOTATION

        # filtering boxes
        filtered_boxes = list(filter(lambda x: self.filter_texts(x), annots))

        # convert tags to boxes
        boxes = list(map(lambda x: self.tags_to_boxes(x, skip), filtered_boxes))

        dict_pages = self.page_wise(boxes)

        # calculate scale ratios
        scale_ratios = self.calculate_scaling_factor(
            self.labels_xml
        )  ###error with sclaing factor

        dict_pages = self.rescale_dict(
            dict_pages, scale_ratios
        )  ### error with rescaing

        # filter too small boxes or with no contextual information blocks
        ################################################################
        dict_pages = self.filter_empty_boxes_using_contours(
            dict_pages
        )  ####### error #####
        ################################################################

        dict_pages = self.merge_single_label(dict_pages)

        dict_pages = self.filter_self_containing_boxes(dict_pages)
        # remove duplicate blocked

        dict_pages = self.sort_coordinates_in_dict(dict_pages)

        if self.show_images is True:
            self.visualize_boxes(dict_pages, self.labels_xml)

        return dict_pages
