## Extract Text Recognition data from COCO (2014)



assume you have installed `coco_text` and downloaded `coco 2014` dataset

```python
## coco text dataset
ct = coco_text.COCO_Text('./COCO/cocotext.v2.json')
data_path = './COCO/train2014'
dataType = 'train2014'

# get available idx 
img_idx = ct.getImgIds(imgIds = ct.train, catIds=[('legibility', 'legible')])
```



let's  see some images `coco_text` chosen

```python
# randomly pick one 
img = ct.loadImgs(img_idx[np.random.randint(0, len(img_idx))])[0]
image_path = os.path.join(data_path, img['file_name'])
i = io.imread(image_path)
print(image_path)
# plt.figure(figsize = (20,40))
plt.imshow(i)
plt.xlabel(i.shape)

```



you can also check annotation on the image

```python
# get corresponding annotation idx
ann_idx = ct.getAnnIds(imgIds = img['id'])
anns = ct.loadAnns(ann_idx)
ct.showAnns(anns)
```



and see extracted text recognition image

```python
for idx, ann in enumerate(anns):
    if ann['utf8_string'] =='':
        continue
    top_left_x = abs(int(ann['bbox'][0] ))
    top_left_y = abs(int(ann['bbox'][1] ))
    width = abs(int(ann['bbox'][2]*1.1)) # multiply 1.1 for residual space from text
    height = abs(int(ann['bbox'][3]*1.1)) # multiply 1.1 for residual space from text
    cropped_image = i[top_left_y : top_left_y + height , top_left_x : top_left_x + width, :]
    
    if (cropped_image.shape[0] < 10) | (cropped_image.shape[1] < 10) :
        continue
    
    n_rows = len(anns)%4 if len(anns)//4==0 else len(anns)//4
    plt.subplot(((idx+1e-10)//n_rows)+1, 4, (idx%4)+1 )
    plt.imshow(cropped_image)
    plt.xlabel(f"{ann['utf8_string']}\nsize : {cropped_image.shape}")
plt.show()
```



now, you can make new dataset including only text recognition image