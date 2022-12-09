# このスクリプトのライセンスは、train_dreambooth.pyと同じくApache License 2.0とします
# (c) 2022 Kohya S. @kohya_ss

# 横長の画像から顔検出して正立するように回転し、そこを中心に正方形に切り出す

# v2: extract max face if multiple faces are found

import argparse
import math
import cv2
import glob
import os
from anime_face_detector import create_detector
from tqdm import tqdm
import numpy as np

KP_REYE = 11
KP_LEYE = 19

SCORE_THRES = 0.90


def detect_face(detector, image):
  preds = detector(image)                     # bgr
  # print(len(preds))
  if len(preds) == 0:
    return None, None, None, None, None

  index = -1
  max_score = 0
  max_size = 0
  for i in range(len(preds)):
    bb = preds[i]['bbox']
    score = bb[-1]
    size = max(bb[2]-bb[0], bb[3]-bb[1])
    if (score > max_score and max_score < SCORE_THRES) or (score >= SCORE_THRES and size > max_size):
      index = i
      max_score = score
      max_size = size

  left = preds[index]['bbox'][0]
  top = preds[index]['bbox'][1]
  right = preds[index]['bbox'][2]
  bottom = preds[index]['bbox'][3]
  cx = int((left + right) / 2)
  cy = int((top + bottom) / 2)
  fw = int(right - left)
  fh = int(bottom - top)

  lex, ley = preds[index]['keypoints'][KP_LEYE, 0:2]
  rex, rey = preds[index]['keypoints'][KP_REYE, 0:2]
  angle = math.atan2(ley - rey, lex - rex)
  angle = angle / math.pi * 180
  return cx, cy, fw, fh, angle


def rotate_image(image, angle, cx, cy):
  h, w = image.shape[0:2]
  rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

  # # 回転する分、すこし画像サイズを大きくする→とりあえず無効化
  # nh = max(h, int(w * math.sin(angle)))
  # nw = max(w, int(h * math.sin(angle)))
  # if nh > h or nw > w:
  #   pad_y = nh - h
  #   pad_t = pad_y // 2
  #   pad_x = nw - w
  #   pad_l = pad_x // 2
  #   m = np.array([[0, 0, pad_l],
  #                 [0, 0, pad_t]])
  #   rot_mat = rot_mat + m
  #   h, w = nh, nw
  #   cx += pad_l
  #   cy += pad_t

  result = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
  return result, cx, cy


def process(args):
  assert not (args.resize_fit and args.resize_face_size is not None), f"resize_fit and resize_face_size can't be specified both / resize_fitとresize_face_sizeはどちらか片方しか指定できません"

  # アニメ顔検出モデルを読み込む
  print("loading face detector.")
  detector = create_detector('yolov3')

  # cropの引数を解析する
  if args.crop_size is None:
    crop_width = crop_height = None
  else:
    tokens = args.crop_size.split(',')
    assert len(tokens) == 2, f"crop_size must be 'width,height' / crop_sizeは'幅,高さ'で指定してください"
    crop_width, crop_height = [int(t) for t in tokens]

  # 画像を処理する
  print("processing.")
  output_extension = ".png"

  os.makedirs(args.dst_dir, exist_ok=True)
  paths = glob.glob(os.path.join(args.src_dir, "*.png")) + glob.glob(os.path.join(args.src_dir, "*.jpg"))
  for path in tqdm(paths):
    basename = os.path.splitext(os.path.basename(path))[0]

    # image = cv2.imread(path)        # 日本語ファイル名でエラーになる
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 2:
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
      print(f"image has alpha. ignore / 画像の透明度が設定されているため無視します: {path}")
      image = image[:, :, :3].copy()                    # copyをしないと内部的に透明度情報が付いたままになるらしい

    h, w = image.shape[:2]

    cx, cy, fw, fh, angle = detect_face(detector, image)
    if cx is None:
      print(f"face not found, skip: {path}")
      # cx = cy = fw = fh = 0
      continue          # スキップする

    # オプション指定があれば回転する
    if args.rotate and cx != 0:
      image, cx, cy = rotate_image(image, angle, cx, cy)

    # オプション指定があれば顔を中心に切り出す
    if crop_width is not None:
      assert cx > 0, f"face not found for cropping: {path}"

      # リサイズを必要なら行う
      scale = 1.0
      if args.resize_face_size is not None:
        # 顔サイズを基準にリサイズする
        scale = args.resize_face_size / max(fw, fh)
        if scale < crop_width / w:
          print(
              f"image width too small in face size based resizing / 顔を基準にリサイズすると画像の幅がcrop sizeより小さい（顔が相対的に大きすぎる）ので顔サイズが変わります: {path}")
          scale = crop_width / w
        if scale < crop_height / h:
          print(
              f"image height too small in face size based resizing / 顔を基準にリサイズすると画像の高さがcrop sizeより小さい（顔が相対的に大きすぎる）ので顔サイズが変わります: {path}")
          scale = crop_height / h
      else:
        if w < crop_width:
          print(f"image width too small/ 画像の幅がcrop sizeより小さいので画質が劣化します: {path}")
          scale = crop_width / w
        if h < crop_height:
          print(f"image height too small/ 画像の高さがcrop sizeより小さいので画質が劣化します: {path}")
          scale = crop_height / h
        if args.resize_fit:
          scale = max(crop_width / w, crop_height / h)

      if scale != 1.0:
        w = int(w * scale + .5)
        h = int(h * scale + .5)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4)
        cx = int(cx * scale + .5)
        cy = int(cy * scale + .5)
        fw = int(fw * scale + .5)
        fh = int(fh * scale + .5)

      x = cx - crop_width // 2
      cx = crop_width // 2
      if x < 0:
        cx = cx + x
        x = 0
      elif x + crop_width > w:
        cx = cx + (x + crop_width - w)
        x = w - crop_width
      image = image[:, x:x+crop_width]

      y = cy - crop_height // 2
      cy = crop_height // 2
      if y < 0:
        cy = cy + y
        y = 0
      elif y + crop_height > h:
        cy = cy + (y + crop_height - h)
        y = h - crop_height
      image = image[y:y + crop_height]

    # # debug
    # print(path, cx, cy, angle)
    # crp = cv2.resize(image, (image.shape[1]//8, image.shape[0]//8))
    # cv2.imshow("image", crp)
    # if cv2.waitKey() == 27:
    #   break
    # cv2.destroyAllWindows()

    # debug
    if args.debug:
      cv2.rectangle(image, (cx-fw//2, cy-fh//2), (cx+fw//2, cy+fh//2), (255, 0, 255), fw//20)

    # cv2.imwrite(os.path.join(args.dst_dir, f"{basename}_{cx:04d}_{cy:04d}_{fw:04d}_{fh:04d}.{output_extension}"), image)
    _, buf = cv2.imencode(output_extension, image)
    with open(os.path.join(args.dst_dir, f"{basename}_{cx:04d}_{cy:04d}_{fw:04d}_{fh:04d}{output_extension}"), "wb") as f:
      buf.tofile(f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--src_dir", type=str, help="directory to load images / 画像を読み込むディレクトリ")
  parser.add_argument("--dst_dir", type=str, help="directory to save images / 画像を保存するディレクトリ")
  parser.add_argument("--rotate", action="store_true", help="rotate images to align faces / 顔が正立するように画像を回転する")
  parser.add_argument("--resize_fit", action="store_true",
                      help="resize to fit smaller side / 画像の短辺がcrop_sizeにあうようにリサイズする")
  parser.add_argument("--resize_face_size", type=int, default=None,
                      help="resize image before cropping by face size / 切り出し前に顔がこのサイズになるようにリサイズする")
  parser.add_argument("--crop_size", type=str, default=None,
                      help="crop images with 'width,height' pixels, face centered / 顔を中心として'幅,高さ'のサイズで切り出す")
  parser.add_argument("--debug", action="store_true", help="render rect for face / 処理後画像の顔位置に矩形を描画します")
  args = parser.parse_args()

  process(args)
