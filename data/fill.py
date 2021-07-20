import os
from shutil import rmtree, copy
from easydict import EasyDict
os.chdir(os.path.dirname(__file__))

root = "root"

if os.path.exists(root):
    rmtree(root)
    print("root folder cleared!")

os.makedirs(root)
os.makedirs(root + "/ImageSets/Main")
os.makedirs(root + "/Annotations") # xml
os.makedirs(root + "/JPEGImages") # jpg
# os.makedirs(root / "SegmentationClass")
print("root folder created!")

clas = EasyDict({
    "spitball": 0,
    "papercup": 1,
    "orangepeel": 2,
    "bottle": 3,
    "battery": 4,
})


# labels_txt = open(f"{root}/labels.txt", 'w')
# test_txt = open(f"{root}/ImageSets/Main/test.txt", 'w')
# trainval = f"{root}/ImageSets/Main/trainval.txt"

jpg_path = f"old/JPEGImages"
xml_path = f"old/Annotations"

count = 0
for xml in os.listdir(xml_path):
    jpg = os.path.join(jpg_path, xml.replace(".xml", ".jpg"))
    jpeg = os.path.join(jpg_path, xml.replace(".xml", ".jpeg"))
    
    text = f"img_{count:04}"
    newjpg = os.path.join(f"{root}/JPEGImages", f"{text}.jpg")
    newxml = os.path.join(f"{root}/Annotations", f"{text}.xml")
    # 复制图片
    if os.path.exists(jpg):
        copy(jpg, newjpg)
    elif os.path.exists(jpeg):
        copy(jpeg, newjpg)
    else:
        continue
    # 复制标注
    copy(os.path.join(xml_path, xml), newxml)
    # 保存文件名
    # test_txt.write(f"{text}\n")
    count += 1
# 保存分类名
for cla in clas:
    # labels_txt.write(f"{cla}\n")
    print(f"{cla}")

# labels_txt.close()
# test_txt.close()
print("Datasets updated!")