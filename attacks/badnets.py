from PIL import Image
import numpy as np


class Badnets(object):
    def __init__(self, img_size, num, triggers, mode=0, target=0):
        super(Badnets, self).__init__()
        self.img_size = img_size
        self.num = num
        self.mode = mode
        self.target = target
        self.trigger = triggers

    def __call__(self, img, target, backdoor, idx):
        if (self.mode == 0 and idx >= self.num) or self.mode == 2:
            target, backdoor = self.target, 1
            img = np.array(img)

            if img.shape[0] == 32:
                for w in range(24, 26):
                    for h in range(24, 26):
                        for c in range(0, 3):
                            img[w, h, c] = 255

            img = Image.fromarray(img)
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        else:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return img, target, backdoor

    def set_mode(self, mode):
        self.mode = mode
