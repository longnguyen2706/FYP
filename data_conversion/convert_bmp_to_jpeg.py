import os
from PIL import Image
from PIL import ImageMath
#current directory (string)
current_path = os.getcwd()

for root, dirs, files in os.walk(current_path, topdown=False):
    for name in files:
        print(os.path.join(root, name))

        if os.path.splitext(os.path.join(root, name))[1].lower()==".BMP":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                print
                "A jpeg file already exists for %s" % name
                # If a jpeg is *NOT* present, create one from the tiff.
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0]+".jpg"
                try:
                    im = Image.open(os.path.join(root, name))
                    print ("Generating jpeg for %s", name)
                    if (im.mode == 'I'):
                    	im = ImageMath.eval('im/256', {'im':im}).convert('L')
                    im.thumbnail(im.size)
                    im.save(outfile, "JPEG", quality=100)
                except Exception as e:
                    print (e)

