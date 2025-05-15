"""必要的PDF翻译功能，提取与翻译PDF内容"""

import asyncio
import io
import os
import urllib.request
from asyncio import CancelledError
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Dict

import numpy as np
import tqdm
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pymupdf import Document, Font

from .converter import TranslateConverter
from .doclayout import OnnxModel
from .pdfinterp import PDFPageInterpreterEx

NOTO_NAME = "noto"

noto_list = [
    "am", "ar", "bn", "bg", "chr", "el", "gu", "iw", "hi", "kn", 
    "ml", "mr", "ru", "sr", "ta", "te", "th", "ur", "uk"
]


def translate_patch(
    inf: BinaryIO,
    pages: Optional[list[int]] = None,
    vfont: str = "",
    vchar: str = "",
    thread: int = 0,
    doc_zh: Document = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    noto_name: str = "",
    noto: Font = None,
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: List = None,
    **kwarg: Any,
) -> None:
    rsrcmgr = PDFResourceManager()
    layout = {}
    device = TranslateConverter(
        rsrcmgr,
        vfont,
        vchar,
        thread,
        layout,
        lang_in,
        lang_out,
        service,
        noto_name,
        noto,
        envs,
        prompt,
    )

    assert device is not None
    obj_patch = {}
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch)
    if pages:
        total_pages = len(pages)
    else:
        total_pages = doc_zh.page_count

    parser = PDFParser(inf)
    doc = PDFDocument(parser)
    with tqdm.tqdm(total=total_pages) as progress:
        for pageno, page in enumerate(PDFPage.create_pages(doc)):
            if cancellation_event and cancellation_event.is_set():
                raise CancelledError("task cancelled")
            if pages and (pageno not in pages):
                continue
            progress.update()
            if callback:
                callback(progress)
            page.pageno = pageno
            pix = doc_zh[page.pageno].get_pixmap()
            image = np.fromstring(pix.samples, np.uint8).reshape(
                pix.height, pix.width, 3
            )[:, :, ::-1]
            page_layout = model.predict(image, imgsz=int(pix.height / 32) * 32)[0]
            # kdtree 是不可能 kdtree 的，不如直接渲染成图片，用空间换时间
            box = np.ones((pix.height, pix.width))
            h, w = box.shape
            vcls = ["abandon", "figure", "table", "isolate_formula", "formula_caption"]
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] not in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = i + 2
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = 0
            layout[page.pageno] = box
            # 新建一个 xref 存放新指令流
            page.page_xref = doc_zh.get_new_xref()  # hack 插入页面的新 xref
            doc_zh.update_object(page.page_xref, "<<>>")
            doc_zh.update_stream(page.page_xref, b"")
            doc_zh[page.pageno].set_contents(page.page_xref)
            interpreter.process_page(page)

    device.close()
    return obj_patch


def translate_stream(
    stream: bytes,
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: List = None,
    **kwarg: Any,
):
    font_list = [("tiro", None)]

    font_path = download_remote_fonts(lang_out.lower())
    noto_name = NOTO_NAME
    noto = Font(noto_name, font_path)
    font_list.append((noto_name, font_path))

    doc_en = Document(stream=stream)
    stream = io.BytesIO()
    doc_en.save(stream)
    doc_zh = Document(stream=stream)
    page_count = doc_zh.page_count
    # font_list = [("GoNotoKurrent-Regular.ttf", font_path), ("tiro", None)]
    font_id = {}
    for page in doc_zh:
        for font in font_list:
            font_id[font[0]] = page.insert_font(font[0], font[1])
    xreflen = doc_zh.xref_length()
    for xref in range(1, xreflen):
        for label in ["Resources/", ""]:  # 可能是基于 xobj 的 res
            try:  # xref 读写可能出错
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                if font_res[0] == "dict":
                    for font in font_list:
                        font_exist = doc_zh.xref_get_key(xref, f"{label}Font/{font[0]}")
                        if font_exist[0] == "null":
                            doc_zh.xref_set_key(
                                xref,
                                f"{label}Font/{font[0]}",
                                f"{font_id[font[0]]} 0 R",
                            )
            except Exception:
                pass

    fp = io.BytesIO()

    doc_zh.save(fp)
    obj_patch: dict = translate_patch(fp, **locals())

    for obj_id, ops_new in obj_patch.items():
        # ops_old=doc_en.xref_stream(obj_id)
        # print(obj_id)
        # print(ops_old)
        # print(ops_new.encode())
        doc_zh.update_stream(obj_id, ops_new.encode())

    doc_en.insert_file(doc_zh)
    for id in range(page_count):
        doc_en.move_page(page_count + id, id * 2 + 1)

    doc_zh.subset_fonts(fallback=True)
    doc_en.subset_fonts(fallback=True)
    return (
        doc_zh.write(deflate=True, garbage=3, use_objstms=1),
        doc_en.write(deflate=True, garbage=3, use_objstms=1),
    )




def download_remote_fonts(lang: str):
    URL_PREFIX = "https://github.com/timelic/source-han-serif/releases/download/main/"
    LANG_NAME_MAP = {
        **{la: "GoNotoKurrent-Regular.ttf" for la in noto_list},
        **{
            la: f"SourceHanSerif{region}-Regular.ttf"
            for region, langs in {
                "CN": ["zh-cn", "zh-hans", "zh"],
                "TW": ["zh-tw", "zh-hant"],
                "JP": ["ja"],
                "KR": ["ko"],
            }.items()
            for la in langs
        },
    }
    font_name = LANG_NAME_MAP.get(lang, "GoNotoKurrent-Regular.ttf")

    # docker
    models_dir = Path("./models/fonts")
    models_dir.mkdir(parents=True, exist_ok=True)
    font_path = models_dir / font_name
    if not font_path.exists():
        print(f"Downloading {font_name} to {font_path}...")
        urllib.request.urlretrieve(f"{URL_PREFIX}{font_name}", font_path)

    return font_path.as_posix()


def extract_and_translate(
    input_file: str,
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: List = None,
):
    """
    提取PDF内容并翻译，将翻译结果保存为JSON文件。
    
    Args:
        input_file: 输入的PDF文件路径。
        output_json_path: 输出翻译结果的JSON文件路径。
        pages: 要翻译的页面列表，默认为全部。
        lang_in: 源语言。
        lang_out: 目标语言。
        service: 翻译服务。
        thread: 线程数。
        vfont: 自定义字体规则。
        vchar: 自定义字符规则。
        model: 识别模型。
        envs: 翻译服务相关的环境变量。
        prompt: 翻译提示模板。
    """
    doc_raw = open(input_file, "rb")
    s_raw = doc_raw.read()
    doc_raw.close()

    # 调用翻译流
    _, translated_json = translate_stream(
        s_raw,
        pages=pages,
        lang_in=lang_in,
        lang_out=lang_out,
        service=service,
        thread=thread,
        vfont=vfont,
        vchar=vchar,
        model=model,
        envs=envs,
        prompt=prompt,
    )


def write_translated_result(
    input_file: str,
    output_dir: str = "result",
    pages: Optional[List[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: List = None,
    **kwargs: Any,
):
    """
    和 translate() 处理方式一致：先将原始 PDF 重新保存成一份干净的 PDF，
    再对其插入新字体、替换文本，最后输出翻译结果 PDF。
    """

    # 1) 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 2) 读取原始 PDF
    with open(input_file, "rb") as doc_raw:
        s_raw = doc_raw.read()

    font_list = [("tiro", None)]

    font_path = download_remote_fonts(lang_out.lower())
    noto_name = NOTO_NAME
    noto = Font(noto_name, font_path)
    font_list.append((noto_name, font_path))

    doc_en = Document(stream=s_raw)
    stream = io.BytesIO()
    doc_en.save(stream)
    doc_zh = Document(stream=stream)
    page_count = doc_zh.page_count

    font_id = {}
    for page in doc_zh:
        for font in font_list:
            font_id[font[0]] = page.insert_font(font[0], font[1])

    xreflen = doc_zh.xref_length()
    for xref in range(1, xreflen):
        for label in ["Resources/", ""]:
            try:
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                if font_res[0] == "dict":
                    for font in font_list:
                        font_exist = doc_zh.xref_get_key(xref, f"{label}Font/{font[0]}")
                        if font_exist[0] == "null":
                            doc_zh.xref_set_key(
                                xref,
                                f"{label}Font/{font[0]}",
                                f"{font_id[font[0]]} 0 R",
                            )
            except Exception:
                pass

    fp = io.BytesIO()
    doc_zh.save(fp)
    obj_patch: dict = translate_patch(fp, **locals())

    for obj_id, ops_new in obj_patch.items():
        # ops_old=doc_en.xref_stream(obj_id)
        # print(obj_id)
        # print(ops_old)
        # print(ops_new.encode())
        doc_zh.update_stream(obj_id, ops_new.encode())

    doc_en.insert_file(doc_zh)
    for id in range(page_count):
        doc_en.move_page(page_count + id, id * 2 + 1)

    doc_zh.subset_fonts(fallback=True)
    doc_en.subset_fonts(fallback=True)

    output_file = os.path.join(
            output_dir,f"{os.path.splitext(os.path.basename(input_file))[0]}_translated{os.path.splitext(input_file)[1]}",
        )

    with open(output_file, "wb") as f:
        f.write(doc_zh.write(deflate=True, garbage=3, use_objstms=1))

    return output_file
