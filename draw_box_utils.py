import matplotlib.font_manager as fm
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from matplotlib.transforms import Bbox
from PIL import ImageColor
from PIL.Image import Image, fromarray

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_text(draw,
              box: list,
              cls: int,
              score: float,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
    
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                  ds,
                  fill='black',
                  font=font)
        left += text_width

def draw_text1(draw,
              box: list,
              cls: int,
              score: float,
              pemi,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        # font = ImageFont.truetype(font, font_size)
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    image_width = right - left
    image_height = bottom - top
    normalize_au = (pemi[2]/image_width+pemi[3]/image_height+pemi[4]/image_width+pemi[5]/image_height)/4
  
    display_str = f"{category_index[str(cls)]}: {int(100 * score)}%  PE:{pemi[0]:.3f}  DropOut_var:{normalize_au:.3f}"
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                  ds,
                  fill='black',
                  font=font)
        left += text_width

def draw_text_au(draw,
              box: list,
              cls: int,
              score: float,
              sigma,
              pe,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        # font = ImageFont.truetype(font, font_size)
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),font_size)

    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    
    image_width = right - left
    image_height = bottom - top
    normalize_au = (sigma[0]/image_width+sigma[1]/image_height+sigma[2]/image_width+sigma[3]/image_height)/4 


    display_str = f"{category_index[str(cls)]}: {int(100 * score)}% PE:{pe:.3f} AU_var:{normalize_au:.3f}"
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                  ds,
                  fill='black',
                  font=font)
        left += text_width

def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))


def draw_objs(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              sigmas= None,
              pes=None,
              gt_box = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = False):
    """
    将目标边界框信息，类别信息，mask信息绘制在图片上
    Args:
        image: 需要绘制的图片
        boxes: 目标边界框信息
        classes: 目标类别信息
        scores: 目标概率信息
        masks: 目标mask信息
        category_index: 类别与名称字典
        box_thresh: 过滤的概率阈值
        mask_thresh:
        line_thickness: 边界框宽度
        font: 字体类型
        font_size: 字体大小
        draw_boxes_on_image:
        draw_masks_on_image:
    Returns:
    """

    # 过滤掉低概率的目标
    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    print("在图上绘制的物体数：", boxes.shape[0])
    classes = classes[idxs]
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        return image

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]

    if draw_boxes_on_image:
        # Draw all boxes onto image.
        draw = ImageDraw.Draw(image)
        i=0
        #  把实验结果绘制出来
        draw_experiment_result_x = 50
        draw_experiment_result_y = 50
        if len(boxes) % 2 ==0: column_len = len(boxes) // 2
        else : column_len = len(boxes) // 2 + 1
        for box, sigma,cls, score,color, pe in zip(boxes, sigmas,classes, scores, colors, pes):
            i+=1
            while_colunm = (i-1) // column_len
            while_lane = (i-1) % column_len
            draw_experiment_result_x = 50*while_lane 
            draw_experiment_result_y = 700*while_colunm 
            fontsize = 30
            fontx = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)
            left, top, right, bottom = box
            
            sigma*=3
            s1, s2, s3, s4=sigma
            tuoyuan1_zuo=left-s1
            tuoyuan1_you=left+s1
            tuoyuan1_top=top-s2
            tuoyuan1_xia=top+s2

            tuoyuan2_zuo=right-s3
            tuoyuan2_you=right+s3
            tuoyuan2_top=bottom-s4
            tuoyuan2_xia=bottom+s4

            # draw.ellipse((tuoyuan1_zuo,tuoyuan1_top, tuoyuan1_you,tuoyuan1_xia), fill=(0,255,0))
            # 如果都画上有点混乱，这里先只绘制右下角的
            # draw.ellipse((tuoyuan2_zuo,tuoyuan2_top, tuoyuan2_you,tuoyuan2_xia), fill=(0,0,255))
            draw.text([(left+right)/2, (top+bottom)/2],  str(i), fill=(255,0,0), font=fontx)  #绘制编号


            left, top, right, bottom = box
            image_width = right - left
            image_height = bottom - top
            normalize_au = (sigma[0]/image_width+sigma[1]/image_height+sigma[2]/image_width+sigma[3]/image_height)/4 
            display_str = f"Index: {i} {category_index[str(cls)]}: {int(100 *float(score))}% PE:{pe:.3f} AU_var:{normalize_au:.3f}"
            draw.text([draw_experiment_result_y, draw_experiment_result_x],  display_str, fill=(255,5,2), font=fontx)  
            # 上面这一行是为了将检测结果画在图片上
            # 绘制目标边界框
            draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], width=line_thickness, fill=color)
            # 绘制类别和概率信息
            draw_text_au(draw, box.tolist(), int(cls), float(score), sigma, pe ,category_index, color, font, font_size)

    if isinstance(gt_box, np.ndarray):

        for box in gt_box:
            #(yolo的数据格式是：中心点的坐标，宽 ，高)
            centerx, centery, w, h =box
            draw.line([(centerx- w/2, centery-h/2),(centerx- w/2, centery+h/2),(centerx+w/2, centery+h/2),(centerx+w/2, centery-h/2),(centerx- w/2, centery-h/2)],width=line_thickness, fill=(0,255,0))


    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks(image, masks, colors, mask_thresh)

    return image


def draw_objs_dropout(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              pemis= None,
              sigmas= None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = False):
    """
    将目标边界框信息，类别信息，mask信息绘制在图片上
    Args:
        image: 需要绘制的图片
        boxes: 目标边界框信息
        classes: 目标类别信息
        scores: 目标概率信息
        masks: 目标mask信息
        category_index: 类别与名称字典
        box_thresh: 过滤的概率阈值
        mask_thresh:
        line_thickness: 边界框宽度
        font: 字体类型
        font_size: 字体大小
        draw_boxes_on_image:
        draw_masks_on_image:
    Returns:
    """

    # 过滤掉低概率的目标
    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    print("在图上绘制的物体数：", boxes.shape[0])
    classes = classes[idxs]
    scores = scores[idxs]
    pemis = pemis[idxs]
    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        return image

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]

    if draw_boxes_on_image:
        # Draw all boxes onto image.
        draw = ImageDraw.Draw(image)
        i=0
        for box, sigma,cls, score,pemi,color in zip(boxes, sigmas,classes, scores, pemis,colors):
            i+=1
            fontsize = 35
            fontx = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)
            left, top, right, bottom = box
            # draw.text([(left+right)/2, (top+bottom)/2],  str(i), fill=(255,0,0), font=fontx)
            sigma*=3
            s1, s2, s3, s4=sigma
            tuoyuan1_zuo=left-s1
            tuoyuan1_you=left+s1
            tuoyuan1_top=top-s2
            tuoyuan1_xia=top+s2

            tuoyuan2_zuo=right-s3
            tuoyuan2_you=right+s3
            tuoyuan2_top=bottom-s4
            tuoyuan2_xia=bottom+s4

            # draw.ellipse((tuoyuan1_zuo,tuoyuan1_top, tuoyuan1_you,tuoyuan1_xia), fill=(0,255,0))
            # draw.ellipse((tuoyuan2_zuo,tuoyuan2_top, tuoyuan2_you,tuoyuan2_xia), fill=(0,0,255))
            # 绘制目标边界框
            draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], width=line_thickness, fill=color)
            # 绘制类别和概率信息
            draw_text1(draw, box.tolist(), int(cls), float(score), pemi, category_index, color, font, font_size)

    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks(image, masks, colors, mask_thresh)

    return image
