# A package of image restoration by leo

There are three methods in our package, which are 

```
# import the package
import image_restoration_tools
irt = image_restoration_tools.Agent()

## 1. input the images dir and the dir path where enhanced images save
irt.restore_dir(imgdir, savedir, path_length=3, isJPEG=False, isBLUR=False)

## 2. input the image path and the path where the enhanced image save
irt.restore_path(imgpath, savepath, path_length=3, isJPEG=False, isBLUR=False)

## 3. input the image and the path where the enhanced image save
import image_restoration_tools.utils
img = utils.imread_uint(imgpath, 3)
irt.restore_image(img, savepath, path_length=3, isJPEG=False, isBLUR=False)
```
You can set enhancement path by set $path_length$, the smaller the value, the shorter the recovery time will be.
if the degrading process is known as JPEG/Gaussian BLUR, set $isJPEG/isBLUR$ true separately

You can install our package by pip:
```
pip install image_restoration_tools
```
