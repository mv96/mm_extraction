from math import *
import PyPDF2
from bs4 import BeautifulSoup as bs
import lxml.etree as etree
import pandas as pd
import copy
import math
import os


class merge_using_pdfalto:
    def __init__(self, xml_main, sample_pdf, scales, table):
        self.xml_main = xml_main
        self.sample_pdf = sample_pdf
        self.scales = scales
        self.table = table
        self.end_of_font_indicator = "[~end_of_font~]"

    def compute_skip(self):
        xml_main = self.xml_main
        sample_pdf = self.sample_pdf
        # creating a pdf file object

        # creating a pdf reader object
        try:
            pdfFileObj = open(sample_pdf, "rb")
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
            true_page_count = pdfReader.numPages
            pdfFileObj.close()
        except:
            # alternate method to count the pages in pdf
            true_page_count = len(os.listdir(sample_pdf.rsplit("/", 1)[0] + "/images"))

        with open(xml_main, "r") as file:
            # Read each line in the file, readlines() returns a list of lines
            content = file.readlines()
            # Combine the lines in the list into a string
            # print(content)
            content = "".join(content)
            bs_content = bs(content, "xml")
            file.close()

        pages = bs_content.find_all("Page")
        if true_page_count == len(pages):
            return False, pages
        else:
            # remove the first page info
            pages.pop(0)
            return True, pages

    def get_page_wise(self, pages):
        """extract all page_wise information as textline"""
        page_wise_text = {}
        for i in range(len(pages)):
            txt_lines = pages[i].find_all("TextLine")

            lines = []
            for line in txt_lines:
                text = ""
                fonts_line = ""

                hpos = float(line.get("HPOS"))
                vpos = float(line.get("VPOS"))
                height = float(line.get("HEIGHT"))
                width = float(line.get("WIDTH"))

                top_left = [hpos, vpos]
                bot_right = [hpos + width, vpos + height]

                coordinates = [top_left, bot_right]
                for second in line:
                    if str(second).startswith("<String"):  # its a string
                        try:
                            content = second.get("CONTENT") + self.end_of_font_indicator
                            font = second.get("STYLEREFS")

                            # print(content)
                            # we can also extract the font

                            text += content
                            fonts_line += font + " "
                        except:
                            continue

                    if str(second).startswith("<SP"):
                        text += " "  # add a space
                        fonts_line += " "

                lines.append([text, coordinates, fonts_line])

            page_wise_text[i + 1] = lines

        return page_wise_text

    def rescales_lines(self, page_wise_text):
        scales = self.scales
        pages_lines = {}
        for page in page_wise_text:
            pdf_alto_info = page_wise_text[page]
            page_lines = []
            for line in pdf_alto_info:
                new_coords = [
                    (
                        math.floor(line[1][0][0] * scales[page - 1][0]),
                        math.floor(line[1][0][1] * scales[page - 1][1]),
                    ),
                    (
                        math.ceil(line[1][1][0] * scales[page - 1][0]),
                        math.ceil(line[1][1][1] * scales[page - 1][1]),
                    ),
                ]
                line[1] = new_coords
                page_lines.append(line)
            # now we have page lines
            pages_lines[page] = page_lines

        return pages_lines

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

    def merge(self):
        table = self.table
        xml_main = self.xml_main
        sample_pdf = self.sample_pdf
        skip, pages = self.compute_skip()
        page_wise_text = self.get_page_wise(
            pages
        )  # all the text present as textlines from the xml main

        res = []
        page_lines = self.rescales_lines(page_wise_text)
        for i, element in table.iterrows():
            top_left_annot = element["top_left"]
            bot_right_annot = element["bot_right"]
            page_no_annots = element["page_no"]
            text_grobid = element["text"]
            label = element["label"]
            # try to fit every line in the annotbox
            subselect = page_lines[page_no_annots]

            margin = 5
            # print(element)
            text_pdf_alto = []
            fonts_pdf_alto = []
            for box in subselect:
                top_left_small = box[1][0]
                bot_right_small = box[1][1]
                if (
                    top_left_annot[0] - margin
                    <= top_left_small[0]
                    <= bot_right_annot[0] + margin
                    and top_left_annot[1] - margin
                    <= top_left_small[1]
                    <= bot_right_annot[1] + margin
                    and top_left_annot[0] - margin
                    <= bot_right_small[0]
                    <= bot_right_annot[0] + margin
                    and top_left_annot[1] - margin
                    <= bot_right_small[1]
                    <= bot_right_annot[1] + margin
                ):
                    text_pdf_alto.append(box[0])
                    fonts_pdf_alto.append(box[-1])
                else:
                    # if there is some intersection even then consider the block
                    # some times the header of the file is also taken into account
                    bb1 = {
                        "x1": top_left_small[0],
                        "y1": top_left_small[1],
                        "x2": bot_right_small[0],
                        "y2": bot_right_small[1],
                    }
                    bb2 = {
                        "x1": top_left_annot[0],
                        "y1": top_left_annot[1],
                        "x2": bot_right_annot[0],
                        "y2": bot_right_annot[1],
                    }
                    try:
                        if self.get_iou(bb1, bb2) > 0:
                            text_pdf_alto.append(box[0])
                            fonts_pdf_alto.append(box[-1])
                    except:
                        print(box)

            final = [
                page_no_annots,
                top_left_annot,
                bot_right_annot,
                text_grobid,
                text_pdf_alto,
                fonts_pdf_alto,
                label,
            ]
            # print(final)
            res.append(final)

        df = pd.DataFrame(
            res,
            columns=[
                "page_no",
                "top_left",
                "bot_right",
                "grobid_text",
                "pdf_alto_text",
                "fonts",
                "label",
            ],
        )

        # df.to_csv("/Users/mv96/Desktop/temp.csv",index=False)
        return df
