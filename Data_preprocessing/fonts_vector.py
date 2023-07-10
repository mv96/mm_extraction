from bs4 import BeautifulSoup as bs
import pandas as pd
from tqdm import tqdm
import functools
import numpy as np
from sklearn.preprocessing import LabelEncoder
from colormap import rgb2hex
from colormap import hex2rgb
import string


class fonts2vec:
    def __init__(self, xml_main, excel_file):
        self.xml_main = xml_main
        self.excel_file = excel_file

    def get_styliing_information_df(self):
        """get mapping by font font0 ==>> cmrbx1"""
        xml_main = self.xml_main
        try:
            # open the xml file and get all the styling based tags
            with open(xml_main, "r") as file:
                # Read each line in the file, readlines() returns a list of lines
                content = file.readlines()
                # Combine the lines in the list into a string
                # print(content)
                content = "".join(content)
                bs_content = bs(content, "xml")
                file.close()
            style_info = bs_content.find_all("Styles")
            # convert this in pandas dataframe
            for element in style_info:
                text_lines = element.find_all("TextStyle")
            # make the dataframe using attributes inside the tags
            # store them in a list format
            lst = []

            for element in text_lines:
                font_color = element.get("FONTCOLOR")
                font_family = element.get("FONTFAMILY")
                font_size = element.get("FONTSIZE")
                font_type = element.get("FONTTYPE")
                font_width = element.get("FONTWIDTH")
                id_ = element.get("ID")
                # sometimes this attribute might be absent
                try:
                    font_style = element.get("FONTSTYLE")
                except:
                    font_style = "None"

                # file id since font can be different
                # file_name=sample_pdf.rsplit("/",1)[1]
                lst.append(
                    [
                        id_,
                        font_color,
                        font_family,
                        font_size,
                        font_type,
                        font_width,
                        font_style,
                    ]
                )

            df = pd.DataFrame(
                lst,
                columns=[
                    "id",
                    "font_color",
                    "font_family",
                    "font_size",
                    "font_type",
                    "font_width",
                    "font_style",
                ],
            )
            return df

        except FileNotFoundError:
            return print("File not found error")

    def fix_nans_in_font_style(self, df):
        df["font_style"] = df["font_style"].replace(np.nan, "Normal")
        return df

    def vectorize_font_style(self, df):
        final = []

        for val in df["font_style"]:
            if "Normal" in val:
                Normal = 1
            else:
                Normal = 0
            if "superscript" in val:
                Superscipt = 1
            else:
                Superscipt = 0

            if "subscript" in val:
                Subscript = 1
            else:
                Subscript = 0

            if "italics" in val:
                italics = 1
            else:
                italics = 0
            if "bold" in val:
                bold = 1
            else:
                bold = 0

            row = [Normal, Superscipt, Subscript, italics, bold]
            final.append(row)

        new = pd.DataFrame(
            final, columns=["Normal", "Superscipt", "Subscript", "italics", "bold"]
        )
        df = pd.concat([df, new], axis=1)
        df = df.drop(columns=["font_style"])
        return df

    def vectorize_fontwidth_fonttype(self, df):
        encoder1 = LabelEncoder()
        encoder1.classes_ = np.load("classes_font_width.npy", allow_pickle=True)
        encoder2 = LabelEncoder()
        encoder2.classes_ = np.load("classes_font_type.npy", allow_pickle=True)

        df["is_Proportional"] = encoder1.fit_transform(df["font_width"])
        df["is_Serif"] = encoder2.fit_transform(df["font_type"])
        df = df.drop(columns=["font_type", "font_width"])
        return df

    def vectorize_font_color(self, df):
        new = []
        for element in df["font_color"]:
            temp = list(np.array(hex2rgb("#" + element)) / 255)
            new.append(temp)
        new = pd.DataFrame(
            new, columns=["font_color_red", "font_color_green", "font_color_blue"]
        )
        df = pd.concat([df, new], axis=1)
        df = df.drop(columns=["font_color"])
        return df

    def fix_font_family(self, df):
        check = []
        for element in df["font_family"]:
            a = element.rstrip(
                string.digits
            )  # strip the last font size at the end of the string
            check.append(a)
        df["font_family"] = pd.DataFrame(check, columns=["font_family"])
        return df

    def vectorize_font_family(self, df):
        excel_file = self.excel_file
        font_excel = pd.read_excel(excel_file)
        new = []
        for font in df["font_family"]:
            font_contains = font_excel[font_excel["Font-family"] == font]
            if len(font_contains) == 1:
                bold = font_contains["Bold"].values[0]
                italic = font_contains["Italics"].values[0]
                ftype = font_contains["FType"].values[0]
                # for bold
                if bold == "Y":
                    is_bold = 1
                else:
                    is_bold = 0

                # for italics
                if italic == "Y":
                    is_italics = 1
                else:
                    is_italics = 0

                # for font type
                if ftype == "S":
                    is_serif = 1
                    is_math = 0
                if ftype == "SS":
                    is_serif = 0
                    is_math = 0
                if ftype == "M":
                    is_math = 1
                    is_serif = 0
                result = [is_bold, is_italics, is_serif, is_math]
                new.append(result)
            else:
                print(f"font{font} not present in the font types or multiple entry:")
                print(
                    "hence ignoring the font and returning all zeros for the manual labelling"
                )
                new.append(
                    [
                        0,
                        0,
                        0,
                        0,
                    ]
                )
        new = pd.DataFrame(
            new,
            columns=[
                "is_bold_manual",
                "is_italic_manual",
                "is_serif_manual",
                "is_math_manual",
            ],
        )
        df = pd.concat([df, new], axis=1)
        df["id"] = df["font_family"]
        df = df.drop(columns=["font_family"])
        return df

    def rescale_font_size(self, df):
        final = []
        # the max font size that can occur is 40
        for element in df["font_size"]:
            if float(element) > 40:
                new = 40
            new = (
                float(element) / 40
            )  # normalizing assuming the lowest possible font is 0 and max possible is 40
            final.append(new)
        new = pd.DataFrame(final, columns=["new_font_size"])
        df = pd.concat([df, new], axis=1)
        df = df.drop(columns=["font_size"])
        return df

    def get_dataframe(self):
        style_df = self.get_styliing_information_df()
        style_df = self.fix_nans_in_font_style(style_df)
        style_df = self.vectorize_font_style(style_df)
        style_df = self.vectorize_fontwidth_fonttype(style_df)
        style_df = self.vectorize_font_color(style_df)
        style_df = self.fix_font_family(style_df)
        style_df = self.vectorize_font_family(style_df)
        style_df = self.rescale_font_size(style_df)

        return style_df
